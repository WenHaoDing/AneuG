"""
check_opening_chamfer.py -- diagnostic: visualize exactly which points the
EXPERIMENTAL opening-chamfer loss samples, on both the canonical (rest
pose) and target sides, so the cap-face identification (fan-cap faces on
the target's closed mesh, mesh.obj-minus-mesh_trimmed.obj faces on
canonical) can be visually confirmed before trusting the loss in a real fit.

conda activate new
python ghd/fitting/check_opening_chamfer.py --stage1-dir runtime/ghd_fit_test/<case-name>
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.multi_canonical_ghd_reconstruct import MultiCanonicalGHDReconstruct
from ghd.fitting.ghd_fit import load_target, setup_experimental_losses


def render_opening_chamfer_sanity(save_path, can_verts_phys, can_faces, can_cap_pts_phys,
                                  tgt_verts_phys, tgt_faces, tgt_cap_pts_phys, n_angles=6):
    import pyvista as pv

    # DISPLAY merely being SET makes system_supports_plotting() true, so over
    # an `ssh -X` forward with no GLX this guard never fired and VTK called
    # abort() -- uncatchable. Always render offscreen on xvfb instead.
    os.environ.pop("DISPLAY", None)
    try:
        pv.start_xvfb()
    except Exception:
        pass

    can_pd = pv.PolyData(can_verts_phys, faces=np.concatenate(
        [np.full((can_faces.shape[0], 1), 3), can_faces], axis=1))
    tgt_pd = pv.PolyData(tgt_verts_phys, faces=np.concatenate(
        [np.full((tgt_faces.shape[0], 1), 3), tgt_faces], axis=1))

    all_pts = np.vstack([can_verts_phys, tgt_verts_phys])
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

        plotter.add_mesh(can_pd, color="royalblue", opacity=0.25, show_edges=False,
                         label="canonical (rest)" if i == 0 else None)
        plotter.add_points(can_cap_pts_phys, color="blue", point_size=6, opacity=0.9,
                           label="canonical cap pts" if i == 0 else None)
        plotter.add_points(tgt_verts_phys, color="lightgreen", point_size=2, opacity=0.15,
                           label="target surface" if i == 0 else None)
        plotter.add_points(tgt_cap_pts_phys, color="darkgreen", point_size=6, opacity=0.9,
                           label="target cap pts" if i == 0 else None)

        plotter.add_text(f"{angle_deg} deg", font_size=10, position="upper_edge")
        if i == 0:
            plotter.add_legend(size=(0.4, 0.35), bcolor="white")
        plotter.set_background("white")
        plotter.camera.position = cam_pos
        plotter.camera.focal_point = focal
        plotter.camera.up = (0.0, 0.0, 1.0)

    plotter.screenshot(str(save_path))
    plotter.close()
    print(f"  Opening-chamfer sanity saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage1-dir", required=True)
    parser.add_argument("--canonical-root", default=str(ROOT / "dataset" / "canonical"))
    parser.add_argument("--aneurysm-type", type=int, default=None)
    parser.add_argument("--out", default=None, help="Default: <stage1-dir>/opening_chamfer_check.png")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-pts", type=int, default=3000)
    args = parser.parse_args()

    stage1_dir = Path(args.stage1_dir)
    device = torch.device(args.device)

    lm = np.load(stage1_dir / "landmarks.npz", allow_pickle=True)
    atype = int(lm["aneurysm_type"]) if args.aneurysm_type is None else args.aneurysm_type
    mc = MultiCanonicalGHDReconstruct(args.canonical_root, device=device)
    recon = mc.get(atype)
    norm_canonical = recon.norm_canonical

    target = load_target(stage1_dir, norm_canonical)
    if target["n_openings"] == 0:
        print(f"  FAILED: {stage1_dir} has no saved opening-ring data (re-run Stage 1 alignment.py first).")
        return

    experimental = setup_experimental_losses(mc, atype, args.canonical_root, target, device,
                                              n_opening_chamfer_pts=args.n_pts)
    if "opening_chamfer" not in experimental:
        print(f"  FAILED: could not build opening-chamfer data for {stage1_dir} "
              f"(no usable cap faces on one or both sides).")
        return
    oc = experimental["opening_chamfer"]

    V0 = recon.canonical_Meshes.verts_packed()
    F_can = recon.canonical_Meshes.faces_packed()
    can_verts_phys = (V0 * norm_canonical).detach().cpu().numpy()
    can_faces_np = F_can.detach().cpu().numpy()
    can_cap_faces_np = can_faces_np[oc["can_cap_face_idx"].cpu().numpy()]
    import trimesh
    can_cap_mesh = trimesh.Trimesh(vertices=can_verts_phys, faces=can_cap_faces_np, process=False)
    can_cap_pts_phys, _ = trimesh.sample.sample_surface(can_cap_mesh, args.n_pts)

    tgt_verts_phys = target["mesh_verts_norm"] * norm_canonical
    tgt_cap_pts_phys = oc["tgt_cap_pts"].squeeze(0).cpu().numpy() * norm_canonical

    print(f"  {stage1_dir.name}: canonical cap faces={len(can_cap_faces_np)}  "
          f"target cap faces={int(target['cap_face_mask'].sum())}  "
          f"n_pts sampled per side={args.n_pts}")

    out_path = Path(args.out) if args.out else stage1_dir / "opening_chamfer_check.png"
    render_opening_chamfer_sanity(
        out_path, can_verts_phys, can_faces_np, np.asarray(can_cap_pts_phys),
        tgt_verts_phys, target["mesh_faces"], tgt_cap_pts_phys,
    )


if __name__ == "__main__":
    main()
