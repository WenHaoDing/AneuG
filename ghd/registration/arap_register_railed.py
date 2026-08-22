"""arap_register_railed.py -- PARALLEL prototype to arap_register.py, NOT a
replacement. arap_register.py uses one rigid-body ring transform per branch
(tip only, no info about intermediate bending along the branch's length).
This script instead builds DENSE point handles running each branch's WHOLE
length, split into side-consistent "rails" -- fixing the original point-
snap bug (independent nearest-vertex snaps per centerline point have no
notion of which side of the tube's circular cross-section they land on,
so adjacent handles could snap to front/back-mismatched vertices and drag
them past each other, which is what produced the crease/twist artifact
this whole detour started from).

THE FRONT/BACK REFERENCE: a single consistent signed normal, shared by
every branch, built the same way on canonical and case:
  1. per branch, take the INITIAL tangent direction near the branch's
     start (NOT overall start->end, which can wander with individual
     vessel tortuosity -- the initial direction right at a bifurcation is
     anatomically far more stable across cases).
  2. SVD over these tangent vectors -> smallest singular vector = best-
     fit plane normal (sign-ambiguous).
  3. orientation fix: cyclic sum of cross products between consecutive
     tangents (cross(v0,v1)+cross(v1,v2)+...+cross(v_{n-1},v0) for n>=3;
     just cross(v0,v1) for n==2, since the cyclic sum degenerates to zero
     there) -- this has a definite sign fixed by the branch ORDER, unlike
     the SVD axis alone. Flip the SVD normal to agree with it.
  4. sign-match canonical's normal against the case's (after the existing
     rigid+scale alignment transform) via a dot-product check -- same
     branch ordering doesn't guarantee the same handedness survives
     alignment on its own.

RAILS: for each branch, at each retained arc-length sample (the earliest
samples near the dome are dropped -- same rationale as
tps_register.py's landmark_start_frac experiment: the branch/dome
boundary has no clean geometric edge, so near-dome candidates are noisy),
query the k nearest mesh vertices to that centerline point, split them by
sign(dot(vertex - centerline_point, n)), and take the nearest ONE within
each side -- never comparing across sides. Do this independently on
canonical (mesh vertices = ARAP handle indices) and on the case's own
closed_mesh (real surface positions = ARAP handle TARGETS), then pair
canonical_front[k,i] <-> case_front[k,i] and canonical_back[k,i] <->
case_back[k,i]. Both ends of every pair are now actual surface points at
a matched (arc-length, side) -- not "canonical surface point vs. the
case's side-less centerline point" like the original broken version.

conda activate new
python ghd/registration/arap_register_railed.py --case-dir /path/to/case
"""

import argparse
import sys
from pathlib import Path

import igl
import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ghd.fitting.alignment import (
    load_case_data, load_canonical_data, fit_alignment, _resample_branch_uniform, DOME_LOADERS,
)
from ghd.registration.tps_register import render_warped_vs_target


def _initial_tangent(seg, idx=None):
    """seg: (n_resample,3) dome-first resampled branch. Direction near the
    START (idx a small fraction in, not the raw first difference, for a
    little robustness to jitter right at the origin) -- see module
    docstring on why this beats the overall start->end direction."""
    n = len(seg)
    if idx is None:
        idx = max(2, n // 10)
    v = seg[idx] - seg[0]
    return v / np.linalg.norm(v)


def _consistent_normal(tangents):
    """tangents: list of unit vectors, one per branch, SAME order on both
    sides. Returns a normal with a definite, order-derived sign (not just
    an axis) -- see module docstring steps 2-3."""
    n_branches = len(tangents)
    T = np.stack(tangents, axis=0)
    if n_branches >= 3:
        _, _, Vt = np.linalg.svd(T - T.mean(axis=0, keepdims=True))
        n_svd = Vt[-1]
        n_svd /= np.linalg.norm(n_svd)
        cross_sum = np.zeros(3)
        for i in range(n_branches):
            cross_sum += np.cross(tangents[i], tangents[(i + 1) % n_branches])
        n_cross = cross_sum / np.linalg.norm(cross_sum)
    else:
        # 2 branches: any 2 vectors exactly span a plane -- no SVD needed,
        # and the cyclic cross-sum would cancel to zero (cross(v0,v1)+
        # cross(v1,v0)==0), so just use the single cross product directly.
        n_cross = np.cross(tangents[0], tangents[1])
        n_cross /= np.linalg.norm(n_cross)
        n_svd = n_cross.copy()
    if np.dot(n_svd, n_cross) < 0:
        n_svd = -n_svd
    return n_svd


def build_rail_handles(case, canonical, transform, device, V_canon, closed_mesh_case,
                       n_resample=64, start_frac=0.3, k_neighbors=12):
    """Returns (handle_idx, target_pos) for ARAP: handle_idx indexes INTO
    V_canon (canonical mesh vertices), target_pos are real 3D positions on
    the case's own closed_mesh, in the aligned frame."""
    n_branches = len(case["branch_segments"])

    canon_segs = [_resample_branch_uniform(canonical["branch_segments"][k], n_resample) for k in range(n_branches)]
    case_segs_raw = [_resample_branch_uniform(case["branch_segments"][k], n_resample) for k in range(n_branches)]
    with torch.no_grad():
        case_segs = [transform(torch.as_tensor(seg, dtype=torch.float32, device=device)).cpu().numpy()
                    for seg in case_segs_raw]

    canon_tangents = [_initial_tangent(seg) for seg in canon_segs]
    case_tangents = [_initial_tangent(seg) for seg in case_segs]
    n_canon = _consistent_normal(canon_tangents)
    n_case = _consistent_normal(case_tangents)
    if np.dot(n_canon, n_case) < 0:
        n_case = -n_case
    print(f"  consistent normal: canon={n_canon.round(3)} case={n_case.round(3)} "
          f"(dot={np.dot(n_canon, n_case):.3f})")

    canon_tree = cKDTree(V_canon)
    with torch.no_grad():
        case_verts_raw = torch.as_tensor(closed_mesh_case.vertices, dtype=torch.float32, device=device)
        case_verts = transform(case_verts_raw).cpu().numpy().astype(np.float64)
    case_tree = cKDTree(case_verts)

    start_idx = int(round(start_frac * n_resample))
    handle_idx_list, target_pos_list = [], []
    n_front_pairs = n_back_pairs = 0

    for k in range(n_branches):
        for i in range(start_idx, n_resample):
            c_canon = canon_segs[k][i]
            c_case = case_segs[k][i]

            # canonical side: k nearest mesh vertices, split by side, nearest-in-side
            dists, idxs = canon_tree.query(c_canon, k=k_neighbors)
            sides = np.sign(np.dot(V_canon[idxs] - c_canon, n_canon))
            front_cands = idxs[sides > 0]
            back_cands = idxs[sides < 0]

            # case side: same, on the case's real surface
            dists_c, idxs_c = case_tree.query(c_case, k=k_neighbors)
            sides_c = np.sign(np.dot(case_verts[idxs_c] - c_case, n_case))
            front_cands_c = idxs_c[sides_c > 0]
            back_cands_c = idxs_c[sides_c < 0]

            if len(front_cands) > 0 and len(front_cands_c) > 0:
                handle_idx_list.append(front_cands[0])  # cKDTree.query returns sorted by distance
                target_pos_list.append(case_verts[front_cands_c[0]])
                n_front_pairs += 1
            if len(back_cands) > 0 and len(back_cands_c) > 0:
                handle_idx_list.append(back_cands[0])
                target_pos_list.append(case_verts[back_cands_c[0]])
                n_back_pairs += 1

    print(f"  {n_front_pairs} front-rail pair(s), {n_back_pairs} back-rail pair(s) "
          f"across {n_branches} branch(es)")

    handle_idx = np.asarray(handle_idx_list, dtype=np.int64)
    target_pos = np.asarray(target_pos_list, dtype=np.float64)
    # dedupe: a canonical vertex could win "nearest in side" at consecutive
    # arc indices if resampling is denser than the mesh's own vertex spacing
    handle_idx, uniq_pos = np.unique(handle_idx, return_index=True)
    target_pos = target_pos[uniq_pos]
    return handle_idx, target_pos


def run_railed_registration(case_dir, canonical_root=None, out_dir=None, align_epochs=800,
                            device="cpu", n_resample=64, start_frac=0.3, k_neighbors=12):
    case_dir = Path(case_dir)
    out_dir = Path(out_dir) if out_dir else ROOT / "runtime" / "arap_railed_test" / case_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    canonical_root = Path(canonical_root) if canonical_root else ROOT / "dataset" / "canonical"

    case = load_case_data(case_dir, dome_points_loader=DOME_LOADERS["nrrd"])
    aneurysm_type = case["aneurysm_type"]
    canonical_dir = canonical_root / ("Bifurcated" if aneurysm_type == 0 else "Sidewall")
    canonical = load_canonical_data(canonical_dir)
    transform, _ = fit_alignment(case, canonical, epochs=align_epochs, device=device, log_every=align_epochs)

    canon_mesh = trimesh.load(canonical_dir / "mesh.obj", process=False)
    V = np.asarray(canon_mesh.vertices, dtype=np.float64)
    F = np.asarray(canon_mesh.faces, dtype=np.int64)

    b, bc = build_rail_handles(case, canonical, transform, device, V, case["closed_mesh"],
                               n_resample=n_resample, start_frac=start_frac, k_neighbors=k_neighbors)

    b32 = b.astype(np.int32)
    arap_data = igl.ARAPData()
    igl.arap_precomputation(V, F, 3, b32, arap_data)
    U = igl.arap_solve(bc, arap_data, V.copy())

    warped_mesh = trimesh.Trimesh(vertices=U, faces=F, process=False)
    try:
        vol_ratio = warped_mesh.volume / canon_mesh.volume
        print(f"  volume ratio (warped/canonical) = {vol_ratio:.4f}")
    except Exception as e:
        print(f"  volume ratio: could not compute ({e})")

    with torch.no_grad():
        closed_verts = torch.as_tensor(case["closed_mesh"].vertices, dtype=torch.float32, device=device)
        target_verts = transform(closed_verts).cpu().numpy()

    np.savez(out_dir / "arap_railed_warp_result.npz", V=V, F=F, U=U, target_verts=target_verts, b=b, bc=bc)
    warped_mesh.export(out_dir / "canonical_arap_railed_warped.obj")
    render_warped_vs_target(out_dir / "warped_vs_target.png", U, F, target_verts)
    print(f"  Done -> {out_dir}")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--canonical-root", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--align-epochs", type=int, default=800)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-resample", type=int, default=64)
    parser.add_argument("--start-frac", type=float, default=0.3,
                        help="Drop this fraction of each branch's proximal (near-dome) "
                             "arc-length samples from rail construction.")
    parser.add_argument("--k-neighbors", type=int, default=12,
                        help="How many nearest mesh vertices to consider per side-split at "
                             "each arc-length sample.")
    args = parser.parse_args()
    run_railed_registration(args.case_dir, args.canonical_root, args.out_dir, args.align_epochs,
                            args.device, args.n_resample, args.start_frac, args.k_neighbors)


if __name__ == "__main__":
    main()
