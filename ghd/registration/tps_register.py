"""tps_register.py -- diagnostic prototype: thin-plate-spline (TPS) landmark
warp from the canonical mesh onto a specific case, to see whether a dense
per-vertex "where should this node go" field is a viable way to guide GHD
fitting past the local-optimum failure mode discussed in the AneuX_stable/
ImperialNHS tuning sessions (chamfer/occupancy alone let a dome vertex bulge
into a false stub instead of the real branch bending to reach it).

WHY TPS, BRIEFLY (see conversation for the fuller discussion): geodesic-
distance-based coordinates can't break a shape's own bilateral symmetry
(front/back look identical intrinsically), so they can't reliably tell a
mesh optimizer "which way is which". TPS sidesteps this entirely by using
the ALREADY-SOLVED correspondence this pipeline has via Stage 1 alignment
(canonical branch k <-> case's real branch k, canonical neck <-> case's
aneurysm centroid) as landmark pairs, and fits the smoothest possible
R^3 -> R^3 warp that hits every pair exactly:

    f(x) = a0 + A @ x + sum_i w_i * U(|x - p_i|),   U(r) = r  (3D kernel)

("smoothest" = minimum bending energy, closed form -- solve one small
linear system for w/a0/A, no iteration). This is the fitting done here;
nothing about this script trains anything or touches ghd_fit.py -- it only
produces a field + a picture to eyeball before deciding whether to wire it
into the actual loss.

CAVEAT worth watching for: with N landmarks in 3D, the system is only
MORE expressive than a plain affine map once N > 4 (an affine map already
has 12 free parameters -- enough to hit any 4 non-degenerate points
exactly on its own, leaving nothing for the bending term to explain). This
case's landmark set (openings + neck centroid) may be right at or near
that boundary depending on branch count -- see the printed
`rbf_weight_norm` diagnostic: near-zero means the fit is effectively pure
affine, not a real bend.

conda activate new
python ghd/registration/tps_register.py --case-dir /path/to/case
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ghd.fitting.alignment import (
    load_case_data, load_canonical_data, fit_alignment, DOME_LOADERS,
    _resample_branch_uniform,
)


# ── TPS core ─────────────────────────────────────────────────────────────────

def fit_tps(source_pts, target_pts, reg=0.0):
    """Closed-form 3D thin-plate spline. source_pts/target_pts: (N,3) arrays,
    corresponding landmark pairs already expressed in the SAME frame the
    returned warp will be evaluated in (source_pts are where landmarks
    START, target_pts are where they should END UP). Returns a callable
    warp(x) -> warped x, exact at the landmarks (reg=0.0), smooth (minimum
    bending energy) everywhere else.

    reg: small positive value relaxes exactness for smoothness if landmarks
    are noisy correspondences. 0.0 here since these come from an already-
    solved discrete correspondence (Stage 1's branch/neck matching), not
    noisy detections.
    """
    source_pts = np.asarray(source_pts, dtype=np.float64)
    target_pts = np.asarray(target_pts, dtype=np.float64)
    n = len(source_pts)

    diffs = source_pts[:, None, :] - source_pts[None, :, :]
    K = np.linalg.norm(diffs, axis=-1)          # U(r) = r, the 3D biharmonic kernel
    K += reg * np.eye(n)
    P = np.hstack([np.ones((n, 1)), source_pts])  # n x 4: [1, x, y, z]

    A = np.zeros((n + 4, n + 4))
    A[:n, :n] = K
    A[:n, n:] = P
    A[n:, :n] = P.T
    b = np.zeros((n + 4, 3))
    b[:n] = target_pts

    sol = np.linalg.solve(A, b)
    w = sol[:n]        # (n,3) RBF weights -- the actual "bending" part
    affine = sol[n:]   # (4,3): row 0 = translation, rows 1:4 = 3x3 linear map

    def warp(x):
        x = np.asarray(x, dtype=np.float64)
        d = np.linalg.norm(x[:, None, :] - source_pts[None, :, :], axis=-1)  # (M,n)
        return d @ w + affine[0] + x @ affine[1:]

    warp.rbf_weight_norm = float(np.linalg.norm(w))
    warp.affine_norm = float(np.linalg.norm(affine))
    return warp


# ── landmark assembly ────────────────────────────────────────────────────────

def build_landmarks(case_dir, canonical_root=None, align_epochs=800, device="cpu",
                    n_resample=64, landmark_start_frac=0.0):
    """Runs a fresh DEFAULT (no branch-focus, no experimental weighting)
    Stage 1 alignment to bring the case into canonical-comparable space,
    then returns (canonical_landmarks, case_landmarks, case, canonical,
    transform).

    n_resample: landmarks per branch. n_resample=1 reduces to the single
    endpoint (the degenerate 4-landmark case that came out purely affine --
    see module docstring). n_resample>1 resamples each branch's FULL
    centerline to n_resample points at uniform arc-length (reusing
    _resample_branch_uniform, the same helper built for point2point
    centerline matching in alignment.py) on both canonical and case sides,
    dome-first order already known/consistent so index k on one side
    matches index k on the other. This constrains the warp ALONG each
    branch, not just at its tip, and pushes well past the affine-DOF
    threshold (n_branches * n_resample + 1 total landmarks) so the RBF/
    bending term actually has something to do.
    """
    ROOT_ = Path(__file__).resolve().parents[2]
    canonical_root = Path(canonical_root) if canonical_root else ROOT_ / "dataset" / "canonical"

    case = load_case_data(case_dir, dome_points_loader=DOME_LOADERS["nrrd"])
    canonical_dir = (canonical_root / "Bifurcated" if case["aneurysm_type"] == 0
                     else canonical_root / "Sidewall")
    canonical = load_canonical_data(canonical_dir)

    transform, _ = fit_alignment(case, canonical, epochs=align_epochs, device=device, log_every=align_epochs)

    n_branches = len(case["branch_segments"])
    if n_resample <= 1:
        canon_branch_list = [canonical["branch_endpoints"][k:k + 1] for k in range(n_branches)]
        with torch.no_grad():
            case_ep_all = transform(torch.as_tensor(
                case["branch_endpoints"], dtype=torch.float32, device=device)).cpu().numpy()
        case_branch_list = [case_ep_all[k:k + 1] for k in range(n_branches)]
    else:
        canon_branch_list = [_resample_branch_uniform(canonical["branch_segments"][k], n_resample)
                             for k in range(n_branches)]
        case_branch_raw_list = [_resample_branch_uniform(case["branch_segments"][k], n_resample)
                                for k in range(n_branches)]
        with torch.no_grad():
            case_branch_list = [transform(torch.as_tensor(
                seg, dtype=torch.float32, device=device)).cpu().numpy() for seg in case_branch_raw_list]

        if landmark_start_frac > 0.0:
            # branches are dome-first (index 0 near dome/neck, last index the
            # far tip -- see _resample_branch_uniform's docstring). Dropping
            # the proximal fraction excludes points close to/inside the dome
            # from the landmark set, so TPS's affine+RBF only has to explain
            # the distal (tip-side) part of each branch -- tests whether
            # near-dome landmarks are what's dragging the dome outward.
            cut = int(round(landmark_start_frac * n_resample))
            canon_branch_list = [seg[cut:] for seg in canon_branch_list]
            case_branch_list = [seg[cut:] for seg in case_branch_list]

    with torch.no_grad():
        case_neck = transform(torch.as_tensor(case["aneurysm_centroid"][None], dtype=torch.float32, device=device)).cpu().numpy()

    canonical_landmarks = np.vstack(canon_branch_list + [canonical["neck_centroid"][None]])
    case_landmarks = np.vstack(case_branch_list + [case_neck])
    return canonical_landmarks, case_landmarks, case, canonical, transform


# ── visualization ────────────────────────────────────────────────────────────

def render_registration_sanity(save_path, canon_verts, warped_verts, target_verts,
                                canonical_landmarks, case_landmarks, n_angles=6,
                                n_arrows=250):
    """One combined multi-angle image: canonical mesh (grey, its rest pose),
    TPS-warped canonical (blue), the actual aligned target surface (green,
    for reference), thin black "node-node mapping" arrows from a subsample
    of canonical vertices to their warped positions, and the landmark pairs
    themselves as large markers (orange=canonical landmark, magenta=case
    landmark) so exact-interpolation can be visually confirmed."""
    import pyvista as pv

    if pv.system_supports_plotting() is False or not os.environ.get("DISPLAY"):
        pv.start_xvfb()

    rng = np.random.default_rng(0)
    idx = rng.choice(len(canon_verts), size=min(n_arrows, len(canon_verts)), replace=False)

    all_pts = np.vstack([canon_verts, warped_verts, target_verts])
    focal = all_pts.mean(axis=0)
    diag = np.linalg.norm(all_pts.max(0) - all_pts.min(0))
    cam_dist = diag * 1.6

    n_cols = min(3, n_angles)
    n_rows = int(np.ceil(n_angles / n_cols))
    plotter = pv.Plotter(off_screen=True, shape=(n_rows, n_cols),
                         window_size=(480 * n_cols, 480 * n_rows), border=True)

    lines = np.hstack([canon_verts[idx], warped_verts[idx]]).reshape(-1, 3)
    n_seg = len(idx)
    line_conn = np.hstack([np.full((n_seg, 1), 2),
                           np.arange(0, 2 * n_seg, 2)[:, None],
                           np.arange(1, 2 * n_seg, 2)[:, None]])
    arrows_pd = pv.PolyData(lines, lines=line_conn.astype(np.int64))
    landmark_point_size = 14 if len(canonical_landmarks) <= 10 else 6

    for i in range(n_angles):
        row, col = divmod(i, n_cols)
        plotter.subplot(row, col)
        angle_deg = round(360 * i / n_angles)
        angle_rad = np.deg2rad(angle_deg)
        cam_pos = focal + cam_dist * np.array([np.cos(angle_rad), np.sin(angle_rad), 0.3])

        plotter.add_points(canon_verts, color="lightgrey", point_size=2, opacity=0.35,
                          label="canonical (rest)" if i == 0 else None)
        plotter.add_points(target_verts, color="green", point_size=2, opacity=0.35,
                          label="target (aligned)" if i == 0 else None)
        plotter.add_points(warped_verts, color="blue", point_size=3, opacity=0.6,
                          label="canonical (TPS-warped)" if i == 0 else None)
        plotter.add_mesh(arrows_pd, color="black", line_width=1, opacity=0.5,
                        label="node-node mapping" if i == 0 else None)
        plotter.add_points(canonical_landmarks, color="orange", point_size=landmark_point_size,
                          label="canonical landmark" if i == 0 else None)
        plotter.add_points(case_landmarks, color="magenta", point_size=landmark_point_size,
                          label="case landmark (target)" if i == 0 else None)

        plotter.add_text(f"{angle_deg} deg", font_size=10, position="upper_edge")
        if i == 0:
            plotter.add_legend(size=(0.4, 0.35), bcolor="white")
        plotter.set_background("white")
        plotter.camera.position = cam_pos
        plotter.camera.focal_point = focal
        plotter.camera.up = (0.0, 0.0, 1.0)

    plotter.screenshot(str(save_path))
    plotter.close()
    print(f"  Registration sanity saved: {save_path}")


def render_warped_vs_target(save_path, warped_verts, faces, target_verts, n_angles=6):
    """Clean two-way comparison, no landmarks/arrows -- TPS-warped canonical
    AS A REAL MESH SURFACE (blue) vs. the aligned target surface (green
    points), same visual convention as ghd_fit.py's render_fit_sanity, so
    it's directly comparable to the eventual GHD-fitted result."""
    import pyvista as pv

    if pv.system_supports_plotting() is False or not os.environ.get("DISPLAY"):
        pv.start_xvfb()

    warped_pd = pv.PolyData(warped_verts, faces=np.concatenate(
        [np.full((faces.shape[0], 1), 3), faces], axis=1))
    all_pts = np.vstack([warped_verts, target_verts])
    focal = all_pts.mean(axis=0)
    diag = np.linalg.norm(all_pts.max(0) - all_pts.min(0))
    cam_dist = diag * 1.6

    n_cols = min(3, n_angles)
    n_rows = int(np.ceil(n_angles / n_cols))
    plotter = pv.Plotter(off_screen=True, shape=(n_rows, n_cols),
                         window_size=(480 * n_cols, 480 * n_rows), border=True)

    for i in range(n_angles):
        row, col = divmod(i, n_cols)
        plotter.subplot(row, col)
        angle_deg = round(360 * i / n_angles)
        angle_rad = np.deg2rad(angle_deg)
        cam_pos = focal + cam_dist * np.array([np.cos(angle_rad), np.sin(angle_rad), 0.3])

        plotter.add_points(target_verts, color="lightgreen", point_size=3, opacity=0.5,
                          label="target (aligned)" if i == 0 else None)
        plotter.add_mesh(warped_pd, color="royalblue", opacity=0.6, show_edges=True,
                         edge_color="navy", label="canonical (TPS-warped)" if i == 0 else None)

        plotter.add_text(f"{angle_deg} deg", font_size=10, position="upper_edge")
        if i == 0:
            plotter.add_legend(size=(0.35, 0.3), bcolor="white")
        plotter.set_background("white")
        plotter.camera.position = cam_pos
        plotter.camera.focal_point = focal
        plotter.camera.up = (0.0, 0.0, 1.0)

    plotter.screenshot(str(save_path))
    plotter.close()
    print(f"  Warped-vs-target saved: {save_path}")


# ── driver ───────────────────────────────────────────────────────────────────

def run_registration(case_dir, canonical_root=None, out_dir=None, align_epochs=800, device="cpu",
                     n_resample=64, tps_reg=0.0, landmark_start_frac=0.0):
    case_dir = Path(case_dir)
    out_dir = Path(out_dir) if out_dir else ROOT / "runtime" / "registration_test" / case_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    canonical_landmarks, case_landmarks, case, canonical, transform = build_landmarks(
        case_dir, canonical_root=canonical_root, align_epochs=align_epochs, device=device,
        n_resample=n_resample, landmark_start_frac=landmark_start_frac)

    n_landmarks = len(canonical_landmarks)
    n_branches = len(case["branch_segments"])
    print(f"  {n_landmarks} landmark pair(s): {n_branches} branch(es) x {n_resample} point(s) "
          f"+ 1 neck centroid")
    if n_landmarks <= 4:
        print(f"  NOTE: {n_landmarks} landmarks <= 4 -- in 3D an affine map alone has 12 free "
              f"parameters, enough to hit up to 4 non-degenerate points exactly on its own. "
              f"Check rbf_weight_norm below: near-zero means this fit is effectively pure "
              f"affine, not a real bend.")

    warp = fit_tps(canonical_landmarks, case_landmarks, reg=tps_reg)
    print(f"  tps_reg={tps_reg:.4g}  rbf_weight_norm={warp.rbf_weight_norm:.4f}  affine_norm={warp.affine_norm:.4f}")

    aneurysm_type = case["aneurysm_type"]
    canonical_dir = ((Path(canonical_root) if canonical_root else ROOT / "dataset" / "canonical")
                     / ("Bifurcated" if aneurysm_type == 0 else "Sidewall"))
    canon_mesh = trimesh.load(canonical_dir / "mesh.obj", process=False)
    canon_verts = np.asarray(canon_mesh.vertices, dtype=np.float64)
    warped_verts = warp(canon_verts)

    warped_mesh_vol = trimesh.Trimesh(vertices=warped_verts, faces=canon_mesh.faces, process=False)
    try:
        vol_ratio = warped_mesh_vol.volume / canon_mesh.volume
        print(f"  volume ratio (warped/canonical) = {vol_ratio:.4f}")
    except Exception as e:
        print(f"  volume ratio: could not compute ({e})")

    with torch.no_grad():
        closed_verts = torch.as_tensor(case["closed_mesh"].vertices, dtype=torch.float32, device=device)
        target_verts = transform(closed_verts).cpu().numpy()

    np.savez(out_dir / "tps_warp_result.npz",
            canonical_landmarks=canonical_landmarks, case_landmarks=case_landmarks,
            canon_verts=canon_verts, warped_verts=warped_verts, target_verts=target_verts,
            rbf_weight_norm=warp.rbf_weight_norm, affine_norm=warp.affine_norm)

    warped_mesh = trimesh.Trimesh(vertices=warped_verts, faces=canon_mesh.faces, process=False)
    warped_mesh.export(out_dir / "canonical_tps_warped.obj")

    render_registration_sanity(out_dir / "registration_sanity.png", canon_verts, warped_verts,
                               target_verts, canonical_landmarks, case_landmarks)
    render_warped_vs_target(out_dir / "warped_vs_target.png", warped_verts, canon_mesh.faces, target_verts)
    print(f"  Done -> {out_dir}")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--canonical-root", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--align-epochs", type=int, default=800)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-resample", type=int, default=64,
                        help="Landmarks per branch (resampled uniformly along its full "
                             "centerline, not just its endpoint). 1 = endpoint only (the "
                             "degenerate all-affine case).")
    parser.add_argument("--tps-reg", type=float, default=0.0,
                        help="TPS smoothing regularization -- relaxes exact landmark "
                             "interpolation for a gentler, less-inflated warp.")
    parser.add_argument("--landmark-start-frac", type=float, default=0.0,
                        help="Drop this fraction of each branch's proximal (near-dome) "
                             "resampled points from the landmark set, keeping only the "
                             "distal/tip-side remainder. 0.0 = full branch (default).")
    args = parser.parse_args()
    run_registration(args.case_dir, args.canonical_root, args.out_dir, args.align_epochs, args.device,
                     args.n_resample, tps_reg=args.tps_reg, landmark_start_frac=args.landmark_start_frac)


if __name__ == "__main__":
    main()
