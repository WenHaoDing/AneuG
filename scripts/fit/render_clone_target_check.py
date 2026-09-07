"""Three-panel visual check of the clone_target pipeline, for one case dir.

Row 1  WORLD LEG      : clone registered onto the case's world-space mesh.
                        Does the surface-chamfer registration land?
Row 2  CANONICAL LEG  : reframed clone vs the real target, both in canonical
                        space. Did the case's own alignment transform put the
                        clone where the target sits?
Row 3  FIT            : final fitted mesh vs the reframed clone -- the thing
                        Stage B was actually asked to match.

Usage: python scripts/fit/render_clone_target_check.py <case_dir>
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh

CASE = Path(sys.argv[1] if len(sys.argv) > 1 else
            "runtime/fitting/clone_target_smoke/ImperialNHS/0ZafRQZ51s_aneurysm1")
VIEWS = (0, 120, 240)

ROWS = [
    ("WORLD: clone (blue) vs case world mesh (green)",
     "case_closed_world.obj", "clone_target_world.obj"),
    ("CANONICAL: reframed clone (blue) vs real target (green)",
     "final_aligned.obj", "clone_target.obj"),
    ("FIT: ghd_fitted (blue) vs reframed clone (green)",
     "clone_target.obj", "ghd_fitted.obj"),
]


def _pd(m):
    f = np.asarray(m.faces)
    return pv.PolyData(np.asarray(m.vertices), np.concatenate([np.full((len(f), 1), 3), f], 1))


# DISPLAY merely being SET makes system_supports_plotting() true, so over
# an `ssh -X` forward with no GLX this guard never fired and VTK called
# abort() -- uncatchable. Always render offscreen on xvfb instead.
os.environ.pop("DISPLAY", None)
try:
    pv.start_xvfb()
except Exception:
    pass

rows = []
for label, ref_name, live_name in ROWS:
    ref_p, live_p = CASE / ref_name, CASE / live_name
    if ref_p.exists() and live_p.exists():
        rows.append((label, trimesh.load(ref_p, process=False), trimesh.load(live_p, process=False)))
    else:
        print(f"  missing: {ref_name if not ref_p.exists() else live_name}")

if not rows:
    sys.exit("nothing to render")

p = pv.Plotter(off_screen=True, shape=(len(rows), len(VIEWS)),
               window_size=(520 * len(VIEWS), 520 * len(rows)), border=True)
for r, (label, ref, live) in enumerate(rows):
    allp = np.vstack([ref.vertices, live.vertices])
    focal = allp.mean(0)
    dist = np.linalg.norm(allp.max(0) - allp.min(0)) * 1.7
    d = np.abs(trimesh.proximity.ProximityQuery(ref).signed_distance(np.asarray(live.vertices)))
    for c, azim in enumerate(VIEWS):
        p.subplot(r, c)
        p.add_mesh(_pd(ref), color="lightgreen", opacity=0.40)
        p.add_mesh(_pd(live), color="royalblue", opacity=0.45)
        p.set_background("white")
        a = np.deg2rad(azim)
        p.camera.position = focal + dist * np.array([np.cos(a), np.sin(a), 0.3])
        p.camera.focal_point = focal
        p.camera.up = (0.0, 0.0, 1.0)
        if c == 0:
            p.add_text(f"{label}\nmean |dist| = {d.mean():.4f} mm   p95 = {np.percentile(d,95):.4f}",
                       font_size=9, position="lower_left", color="black")
        p.add_text(f"{azim} deg", font_size=8, position="upper_edge")
    print(f"  {label}: mean |dist| = {d.mean():.4f} mm  p95 = {np.percentile(d, 95):.4f} mm")

out = CASE / "clone_target_check.png"
p.screenshot(str(out)); p.close()
m = CASE / "metrics.json"
if m.exists():
    d = json.loads(m.read_text())
    print("\nmetrics:", {k: d[k] for k in
                         ("reframe_route", "fit_target", "skip_stage_a", "reframe_scale",
                          "reframe_mean_surface_dist", "chamfer_best") if k in d})
print("\nsaved", out)
