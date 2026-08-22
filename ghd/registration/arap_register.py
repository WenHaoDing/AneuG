"""arap_register.py -- prototype: As-Rigid-As-Possible (ARAP) handle-based
deformation as an alternative to TPS for the Stage A warm-start target.

Unlike TPS (a global affine + unbounded-RBF interpolant, which forces the
whole shape -- including the unconstrained dome -- to inflate in order to
reach stretched branch landmarks), ARAP alternates:
  (a) per-vertex-neighbourhood optimal local rotation (SVD, cotangent-
      weighted Procrustes -- exactly what GHDRigidLoss already computes
      as a SOFT loss term in Stage A/B),
  (b) a global sparse Poisson solve for new positions consistent with
      those rotations, SUBJECT TO a set of HANDLE vertices being pinned
      to exact target positions.

Handles are each branch's OPENING RING (canonical_topology.npy's boundary
loop of vertices at the tube's cut end), moved as ONE RIGID BODY per
branch (rotate by the minimal/no-twist rotation aligning canonical->target
tangent direction, translate centroid to target tip) -- NOT independent
per-point nearest-vertex snaps along the centerline. Point-snapping was
tried first and produced local twisting/pinching: a centerline point has
no notion of which side of the tube's circular cross-section a nearby
mesh vertex sits on, so adjacent handles could snap to front/back-
mismatched vertices and drag them past each other. A whole ring rotating
coherently has no such ambiguity -- there's no per-vertex correspondence
search at all.

No dome anchor: an earlier version also hard-pinned canonical_topology's
dome_vertex_indices to their own resting position, which fixed volume
inflation (ratio ~0.82 vs TPS's ~1.42) but fought the branch handles hard
enough to crease the mesh at the dome/neck boundary. Dropped in favor of
branch mesh health -- some dome drift is an acceptable trade.

conda activate new
python ghd/registration/arap_register.py --case-dir /path/to/case
"""

import argparse
import sys
from pathlib import Path

import igl
import numpy as np
import torch
import trimesh

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ghd.fitting.alignment import (
    load_case_data, load_canonical_data, fit_alignment, _resample_branch_uniform, DOME_LOADERS,
    save_stage1_checkpoints,
)
from ghd.registration.tps_register import render_warped_vs_target


def _rotation_aligning(a, b):
    """Minimal (no-twist) rotation matrix taking unit vector a to unit vector b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s < 1e-10:
        return np.eye(3) if c > 0 else _rotation_180(a)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))


def _rotation_180(a):
    # a and b anti-parallel: rotate 180 deg about any axis perpendicular to a
    perp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    axis = np.cross(a, perp)
    axis /= np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + 2 * K @ K


def _ordered_loop_normal(pts, mesh_centroid_aligned):
    """Outward-oriented normal of an ORDERED boundary loop -- same Newell's-
    method formula as record_topology.py's cross_vector()/
    orient_opening_outward(), applied to the TARGET's own real opening
    ring instead of canonical's. pts: (n,3) in the shared aligned frame,
    in their natural boundary-loop cyclic order."""
    c = pts.mean(axis=0)
    rel = pts - c
    rel_next = np.roll(rel, -1, axis=0)
    cv = np.cross(rel, rel_next).sum(axis=0)
    cv /= np.linalg.norm(cv)
    if np.dot(cv, c - mesh_centroid_aligned) < 0:
        cv = -cv
    return cv


def _slerp_rotation(R1, R2, frac):
    """frac=0 -> R1, frac=1 -> R2, smoothly in between (quaternion slerp)."""
    from scipy.spatial.transform import Rotation, Slerp
    key_rots = Rotation.from_matrix([R1, R2])
    slerp = Slerp([0, 1], key_rots)
    return slerp([frac])[0].as_matrix()


def build_ring_handles(case, canonical, transform, device, canonical_dir, V,
                       use_ring_normal=False, out_dir=None, ring_normal_blend=1.0):
    """Per-branch RIGID handle groups instead of independent point-snaps.
    Each branch's opening ring (canonical_topology.npy's boundary loop of
    vertices at the tube's cut end) moves as ONE rigid body: rotate,
    translate its centroid to the target branch tip. A whole ring
    rotating coherently can't mismatch front/back the way independent
    nearest-vertex point snaps can -- there's no per-vertex correspondence
    search at all, just one rotation applied to the group.

    use_ring_normal: DEFAULT FALSE, preserving the original behavior --
    rotation is the minimal (no-twist) rotation aligning the branch's
    CENTERLINE TANGENT direction canonical->target. Observed to leave the
    ring's cap facing a visibly wrong direction sometimes: the tangent
    (a coarse 2-point finite difference near the tip) is only a PROXY for
    the ring's own true cut-plane orientation, and real anatomical clip
    planes aren't always exactly perpendicular to the local centerline.
    When True, uses each ring's own ACTUAL outward-facing normal instead --
    canonical_topology.npy's precomputed cross_vector for canonical, and
    the same Newell's-method computation applied to the TARGET's own real
    opening ring (out_dir/landmarks.npz's opening_ring_idx_k, written by
    save_stage1_checkpoints -- must have already been called). Requires
    out_dir. Falls back to the tangent method for any branch missing
    target ring data (n_openings < n_branches for that case)."""
    topo = np.load(canonical_dir / "canonical_topology.npy", allow_pickle=True).item()
    n_branches = len(case["branch_segments"])

    target_ring_normals = {}
    if use_ring_normal:
        assert out_dir is not None, "use_ring_normal requires out_dir (reads landmarks.npz)"
        lm = np.load(Path(out_dir) / "landmarks.npz", allow_pickle=True)
        n_openings = int(lm["n_openings"]) if "n_openings" in lm else 0
        with torch.no_grad():
            mesh_centroid_raw = np.asarray(case["closed_mesh"].vertices, dtype=np.float32).mean(axis=0)
            mesh_centroid_aligned = transform(
                torch.as_tensor(mesh_centroid_raw[None], device=device)).cpu().numpy()[0]
        for k in range(n_openings):
            loop_idx = lm[f"opening_ring_idx_{k}"]
            pts_raw = np.asarray(case["closed_mesh"].vertices)[loop_idx]
            with torch.no_grad():
                pts = transform(torch.as_tensor(pts_raw, dtype=torch.float32, device=device)).cpu().numpy()
            target_ring_normals[k] = _ordered_loop_normal(pts, mesh_centroid_aligned)
        print(f"  use_ring_normal: target ring-normal data available for "
              f"{len(target_ring_normals)}/{n_branches} branch(es)")

    handle_idx_list, target_pos_list = [], []
    for k in range(n_branches):
        canon_seg = _resample_branch_uniform(canonical["branch_segments"][k], 8)
        case_seg_raw = _resample_branch_uniform(case["branch_segments"][k], 8)
        with torch.no_grad():
            case_seg = transform(torch.as_tensor(case_seg_raw, dtype=torch.float32, device=device)).cpu().numpy()

        canon_tangent = canon_seg[-1] - canon_seg[-2]
        case_tangent = case_seg[-1] - case_seg[-2]
        R_tangent = _rotation_aligning(canon_tangent, case_tangent)

        if use_ring_normal and k in target_ring_normals:
            canon_normal = np.asarray(topo["openings"][k]["cross_vector"], dtype=np.float64)
            R_ring_normal = _rotation_aligning(canon_normal, target_ring_normals[k])
            R = (R_ring_normal if ring_normal_blend >= 1.0 else
                _slerp_rotation(R_tangent, R_ring_normal, ring_normal_blend))
        else:
            R = R_tangent

        ring_idx = np.asarray(topo["openings"][k]["indices"], dtype=np.int64)
        ring_centroid = V[ring_idx].mean(axis=0)
        target_centroid = case_seg[-1]  # branch tip, in the aligned frame
        new_pos = (V[ring_idx] - ring_centroid) @ R.T + target_centroid

        handle_idx_list.append(ring_idx)
        target_pos_list.append(new_pos)

    return np.concatenate(handle_idx_list), np.concatenate(target_pos_list)


def run_arap_registration(case_dir, canonical_root=None, out_dir=None, align_epochs=800,
                          device="cpu", use_ring_normal=False, ring_normal_blend=1.0):
    case_dir = Path(case_dir)
    out_dir = Path(out_dir) if out_dir else ROOT / "runtime" / "arap_test" / case_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    canonical_root = Path(canonical_root) if canonical_root else ROOT / "dataset" / "canonical"

    case = load_case_data(case_dir, dome_points_loader=DOME_LOADERS["nrrd"])
    aneurysm_type = case["aneurysm_type"]
    canonical_dir = canonical_root / ("Bifurcated" if aneurysm_type == 0 else "Sidewall")
    canonical = load_canonical_data(canonical_dir)
    transform, _ = fit_alignment(case, canonical, epochs=align_epochs, device=device, log_every=align_epochs)
    if use_ring_normal:
        save_stage1_checkpoints(out_dir, transform, case, canonical)

    canon_mesh = trimesh.load(canonical_dir / "mesh.obj", process=False)
    V = np.asarray(canon_mesh.vertices, dtype=np.float64)
    F = np.asarray(canon_mesh.faces, dtype=np.int64)

    b, bc = build_ring_handles(case, canonical, transform, device, canonical_dir, V,
                               use_ring_normal=use_ring_normal, out_dir=out_dir,
                               ring_normal_blend=ring_normal_blend)

    print(f"  {len(b)} ring handle vertex(es) across {len(canonical['branch_endpoints'])} branch(es), "
          f"NO dome anchor (dropped per instruction)")

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

    np.savez(out_dir / "arap_warp_result.npz", V=V, F=F, U=U, target_verts=target_verts,
             b=b, bc=bc)
    warped_mesh.export(out_dir / "canonical_arap_warped.obj")
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
    parser.add_argument("--use-ring-normal", action="store_true",
                        help="Orient each branch's ring transform by the ring's own outward "
                             "normal (canonical's precomputed cross_vector vs. the target's "
                             "real opening ring) instead of the coarse centerline-tangent "
                             "estimate. Default off, preserving the original behavior.")
    parser.add_argument("--ring-normal-blend", type=float, default=1.0,
                        help="SLERP blend between the tangent-based rotation (0.0) and the "
                             "ring-normal-based rotation (1.0), only used with --use-ring-normal. "
                             "Full ring-normal correction can over-twist the cap relative to what "
                             "the tube surface behind it can smoothly absorb; a partial blend "
                             "trades some direction accuracy for less local distortion.")
    args = parser.parse_args()
    run_arap_registration(args.case_dir, args.canonical_root, args.out_dir, args.align_epochs,
                          args.device, use_ring_normal=args.use_ring_normal,
                          ring_normal_blend=args.ring_normal_blend)


if __name__ == "__main__":
    main()
