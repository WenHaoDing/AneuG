"""
check_occupancy.py -- diagnostic: visualize the DVS occupancy ground-truth
point field (prepare_dvs_samples' interior/exterior samples) against the
Stage 1 target mesh, so mesh-closing artifacts (extrude/planarize/fan-cap in
alignment.py, baked into final_aligned.obj) can be visually validated before
trusting lambda_occupancy > 0 in a real fitting run.

conda activate new
python ghd/fitting/check_occupancy.py --stage1-dir runtime/ghd_fit_test/<case-name>
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

from models.multi_canonical_ghd_reconstruct import MultiCanonicalGHDReconstruct
from ghd.fitting.ghd_fit import load_target, prepare_dvs_samples


def render_occupancy_sanity(save_path, target_verts_phys, target_faces, pos_phys, neg_phys, n_angles=6):
    import pyvista as pv

    # DISPLAY merely being SET makes system_supports_plotting() true, so over
    # an `ssh -X` forward with no GLX this guard never fired and VTK called
    # abort() -- uncatchable. Always render offscreen on xvfb instead.
    os.environ.pop("DISPLAY", None)
    try:
        pv.start_xvfb()
    except Exception:
        pass

    target_pd = pv.PolyData(target_verts_phys, faces=np.concatenate(
        [np.full((target_faces.shape[0], 1), 3), target_faces], axis=1))
    all_pts = np.vstack([target_verts_phys, pos_phys, neg_phys])
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

        plotter.add_mesh(target_pd, color="lightgrey", opacity=0.15, show_edges=False,
                         label="target surface" if i == 0 else None)
        plotter.add_points(pos_phys, color="blue", point_size=4, opacity=0.8,
                           label="interior (positive)" if i == 0 else None)
        plotter.add_points(neg_phys, color="red", point_size=2, opacity=0.25,
                           label="exterior (negative)" if i == 0 else None)

        plotter.add_text(f"{angle_deg} deg", font_size=10, position="upper_edge")
        if i == 0:
            plotter.add_legend(size=(0.4, 0.3), bcolor="white")
        plotter.set_background("white")
        plotter.camera.position = cam_pos
        plotter.camera.focal_point = focal
        plotter.camera.up = (0.0, 0.0, 1.0)

    plotter.screenshot(str(save_path))
    plotter.close()
    print(f"  Occupancy sanity saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage1-dir", required=True)
    parser.add_argument("--canonical-root", default=str(ROOT / "dataset" / "canonical"))
    parser.add_argument("--aneurysm-type", type=int, default=None)
    parser.add_argument("--out", default=None, help="Default: <stage1-dir>/occupancy_check.png")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-dvs", type=int, default=20_000)
    args = parser.parse_args()

    stage1_dir = Path(args.stage1_dir)
    device = torch.device(args.device)

    lm = np.load(stage1_dir / "landmarks.npz", allow_pickle=True)
    atype = int(lm["aneurysm_type"]) if args.aneurysm_type is None else args.aneurysm_type
    mc = MultiCanonicalGHDReconstruct(args.canonical_root, device=device)
    norm_canonical = mc.get(atype).norm_canonical

    target = load_target(stage1_dir, norm_canonical)
    result = prepare_dvs_samples(target, device, n_dvs=args.n_dvs)
    if result is None:
        print(f"  FAILED to close/sample target mesh for {stage1_dir} -- "
              f"cannot produce occupancy ground truth (see prepare_dvs_samples).")
        return
    pos, neg, _, _ = result

    pos_phys = pos.cpu().numpy() * norm_canonical
    neg_phys = neg.cpu().numpy() * norm_canonical
    target_verts_phys = target["mesh_verts_norm"] * norm_canonical

    tgt_trimesh = trimesh.Trimesh(vertices=target["mesh_verts_norm"], faces=target["mesh_faces"], process=False)
    print(f"  {stage1_dir.name}: target watertight={tgt_trimesh.is_watertight}  "
          f"interior samples={len(pos_phys)}  exterior samples={len(neg_phys)}")

    out_path = Path(args.out) if args.out else stage1_dir / "occupancy_check.png"
    render_occupancy_sanity(out_path, target_verts_phys, target["mesh_faces"], pos_phys, neg_phys)


if __name__ == "__main__":
    main()
