"""Contact sheets for the Stage-A MSE-weight sweep, for visual comparison.

One PNG per case: a row per lambda_node_mse, showing
  [Stage A output vs clone target]   [final fit vs real target]
so the question "does the shape actually morph, and does it stay healthy?"
is answerable by eye across weights.

Usage: python scripts/fit/build_mse_sweep_sheet.py [sweep_root]
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh

ROOT = Path(sys.argv[1] if len(sys.argv) > 1
            else "/media/yaplab2/HDD Storage/wenhao/AneuG/runtime/fitting/mse_weight_sweep")
# name of the per-arm dir prefix and of the warm-start target mesh:
#   clone sweep -> w<weight>/  + clone_target.obj
#   ARAP  sweep -> mh<scale>/  + canonical_arap_warped.obj
PREFIX = sys.argv[2] if len(sys.argv) > 2 else "w"
REF_MESH = sys.argv[3] if len(sys.argv) > 3 else "clone_target.obj"
KNOB = "lambda_node_mse" if PREFIX == "w" else "mesh_health_scale"
VIEWS = (0, 90)          # azimuths per panel pair


def _pd(mesh):
    f = np.asarray(mesh.faces)
    return pv.PolyData(np.asarray(mesh.vertices), np.concatenate([np.full((len(f), 1), 3), f], 1))


def _cam(plotter, pts, azim):
    focal = pts.mean(0)
    dist = np.linalg.norm(pts.max(0) - pts.min(0)) * 1.7
    a = np.deg2rad(azim)
    plotter.camera.position = focal + dist * np.array([np.cos(a), np.sin(a), 0.3])
    plotter.camera.focal_point = focal
    plotter.camera.up = (0.0, 0.0, 1.0)


def build(case, weights):
    rows = []
    for w in weights:
        d = ROOT / f"{PREFIX}{w}" / "ImperialNHS" / case
        if not (d / "ghd_fitted.obj").exists():
            continue
        m = json.loads((d / "metrics.json").read_text()) if (d / "metrics.json").exists() else {}
        rows.append((w, d, m))
    if not rows:
        print(f"  {case}: nothing complete yet")
        return

    ncol = len(VIEWS) * 2
    p = pv.Plotter(off_screen=True, shape=(len(rows), ncol),
                   window_size=(420 * ncol, 420 * len(rows)), border=True)
    for r, (w, d, m) in enumerate(rows):
        fitted = trimesh.load(d / "ghd_fitted.obj", process=False)
        target = trimesh.load(d / "final_aligned.obj", process=False)
        clone = trimesh.load(d / REF_MESH, process=False)
        stage_a = d / "sanity_stage_a" / "final_vs_clone_target.png"   # existence only
        pairs = [(clone, fitted, "warm-start target vs FINAL"), (target, fitted, "real target vs FINAL")]
        c = 0
        for ref, live, label in pairs:
            for azim in VIEWS:
                p.subplot(r, c); c += 1
                p.add_mesh(_pd(ref), color="lightgreen", opacity=0.40)
                p.add_mesh(_pd(live), color="royalblue", opacity=0.45)
                p.set_background("white")
                _cam(p, np.vstack([ref.vertices, live.vertices]), azim)
                if r == 0:
                    p.add_text(f"{label} @{azim}deg", font_size=8, position="upper_edge")
                if c == 1:
                    cb = m.get("chamfer_best", float("nan"))
                    p.add_text(f"{KNOB}={w}\nchamfer_best={cb:.5f}",
                               font_size=10, position="lower_left", color="black")
    out = ROOT / f"sheet_{case}.png"
    p.screenshot(str(out)); p.close()
    print(f"  {case}: {len(rows)} weight(s) -> {out}")


if __name__ == "__main__":
    # DISPLAY merely being SET makes system_supports_plotting() true, so over
    # an `ssh -X` forward with no GLX this guard never fired and VTK called
    # abort() -- uncatchable. Always render offscreen on xvfb instead.
    os.environ.pop("DISPLAY", None)
    try:
        pv.start_xvfb()
    except Exception:
        pass
    weights = sorted({d.name[len(PREFIX):] for d in ROOT.glob(f"{PREFIX}*") if d.is_dir()},
                     key=lambda s: float(s))
    cases = sorted({c.name for d in ROOT.glob(f"{PREFIX}*/ImperialNHS") for c in d.iterdir() if c.is_dir()})
    print(f"weights={weights}  cases={cases}")

    summary = []
    for case in cases:
        build(case, weights)
        for w in weights:
            f = ROOT / f"{PREFIX}{w}" / "ImperialNHS" / case / "metrics.json"
            if f.exists():
                m = json.loads(f.read_text())
                summary.append((case, float(w), m.get("chamfer_best"), m.get("node_mse_mode")))

    if summary:
        print("\n%-24s %8s %12s %9s" % ("case", "lambda", "chamfer_best", "mse_mode"))
        print("-" * 58)
        for case, w, cb, mode in sorted(summary, key=lambda r: (r[0], r[1])):
            print("%-24s %8.1f %12.6f %9s" % (case, w, cb if cb is not None else float("nan"), mode))
