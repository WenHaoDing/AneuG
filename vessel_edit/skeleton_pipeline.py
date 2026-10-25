"""Dense 1D vessel graph (centerline + radius) with the aneurysm taken out, for conditioning.

Per case, in runtime_skeleton/<case>/ (the geometry dataset is only read, never written):
  1. copy label.nrrd, the meshes and endpoints into input/
  2. VMTK centerlines of the original mesh (only used to bridge the gap in step 3)
  3. place the case's GHD fit in image space; its reviewed dome gives the sac and the ostium
     (geometry/ostium.py -> ostium.npz). Remove that sac + 1 mm collar from the big mesh and
     patch the vessel (heal_vessel.heal_case); if the sac sits at the end of
     reconstruction4clipping.ply, redo on the larger reconstruction.ply.
     Cached: skipped when patched.ply exists with sac_source = ghd_fit.
  4. geometry/extract_skeleton.py (vmtk310 env): patched mesh + endpoints -> skeleton_graph.npy (edges)
     and skeleton_graph_points.npz (the point graph for the models: pos, radius, edge_index,
     branch ids with 0 = aneurysm, one-hot), radius interpolated across the aneurysm region
  5. skeleton.png (graph as radius tubes), graph_1d.png (graph coloured by radius, with a
     close-up of every bifurcation) and radius_profiles.png (edges through the aneurysm)

  python vessel_edit/skeleton_pipeline.py --cases 0bjVkRg2pM_aneurysm1 y3GooHGaVq_aneurysm5
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root: geometry/, models/
from geometry import heal_vessel as hv  # noqa: E402

if os.environ.get("DISPLAY", "").startswith("localhost:"):
    del os.environ["DISPLAY"]          # ssh-forwarded X has no OpenGL here; render off-screen

REPO = Path(__file__).resolve().parents[1]
GEOMETRY_ROOT = Path("/media/yaplab2/wd8tb/wenhao/angioflow/data/geometry/ImperialNHS")
OUT_ROOT = REPO / "runtime_skeleton"
PY_VMTK = "/home/yaplab2/miniconda3/envs/vmtk310/bin/python"
INPUT_FILES = ["label.nrrd", "reconstruction4clipping.ply", "reconstruction.ply",
               "endpoints_manual.npy"]


def copy_inputs(case, geometry_root, inp):
    inp.mkdir(parents=True, exist_ok=True)
    for f in INPUT_FILES:
        src = geometry_root / case / f
        if src.exists() and not (inp / f).exists():
            shutil.copy2(src, inp / f)


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[1]} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return r.stdout


def original_centerlines(inp, mesh_name, out):
    npz = out / f"centerlines_orig_{Path(mesh_name).stem}.npz"
    if not npz.exists():
        run([PY_VMTK, str(REPO / "geometry/vmtk_centerlines.py"), "--mesh", str(inp / mesh_name),
             "--endpoints", str(inp / "endpoints_manual.npy"), "--out", str(npz)])
    return hv.load_paths(npz)


def save_ostium(o, path):
    np.savez(path, **{k: o[k] for k in ("ring", "centroid", "normal", "spans", "area", "dome",
                                        "aneurysm_type", "n_rim_loops", "verts", "faces")})


def patch(case, inp, out, a):
    """Cut the sac (+ collar) out of the big mesh and patch it.

    The sac is the GHD fit's dome, as the generator defines it (GHD-VAE complex, dome cut by
    the morphology sensor): the fit is placed in image space and big-mesh faces lying on its
    reviewed dome patch are the sac. The same patch defines the ostium (ostium.npz)."""
    from geometry.ostium import clean_dome_faces, fitted_ostium
    if (out / "patch_info.json").exists() and (out / "patched.ply").exists():
        info = json.loads((out / "patch_info.json").read_text())
        if info.get("sac_source") == "ghd_fit":
            return info
    ost = fitted_ostium(case, a.dataset)
    save_ostium(ost, out / "ostium.npz")
    dome_faces = clean_dome_faces(ost["faces"], ost["dome"])
    for mesh_name in ("reconstruction4clipping.ply", "reconstruction.ply"):
        if not (inp / mesh_name).exists():
            continue
        c = hv.load_case(inp, mesh_name)
        paths = original_centerlines(inp, mesh_name, out)
        f_sac = hv.locate_sac_from_fit(c, ost["verts"], ost["faces"], dome_faces)
        healed, zone, bridges, notes, f_sac = hv.heal_case(
            c, paths, collar_mm=a.collar_mm, blend_mm=a.blend_mm, h=a.h, mode=a.tube,
            target_faces=a.face_factor * len(c["mesh"].faces), f_sac=f_sac)
        if not any("reaches a path end" in n for n in notes):
            break
    healed.export(out / "patched.ply")
    c["mesh"].export(out / "original.ply")
    hv.submesh(c["mesh"], f_sac, False).export(out / "sac.ply")
    np.save(out / "zone_pts.npy", zone.surface_pts)
    np.savez(out / "bridges.npz", **{f"bridge{i}": b["curve"] for i, b in enumerate(bridges)})
    info = dict(mesh=mesh_name, sac_source="ghd_fit", n_bridges=len(bridges), patch_notes=notes,
                patched_faces=len(healed.faces), watertight=bool(healed.is_watertight),
                ostium_spans_mm=[round(float(x), 2) for x in ost["spans"]])
    (out / "patch_info.json").write_text(json.dumps(info, indent=1))
    return info


def plot_profiles(case, g, path_png):
    """Radius along every edge that crosses the aneurysm region: raw, interpolated, smoothed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [e for e in g["edges"] if e["aneurysm"].any()]
    if not rows:
        return
    ncol = min(3, len(rows))
    nrow = int(np.ceil(len(rows) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.2 * nrow), squeeze=False)
    for ax, e in zip(axes.ravel(), rows):
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(e["pts"], axis=0), axis=1))]
        ax.plot(s, e["radius_raw"], "--", color="tab:orange", lw=1.2, label="raw (patched mesh)")
        ax.plot(s, e.get("radius_interp", e["radius"]), color="tab:green", lw=1.2, label="interpolated")
        ax.plot(s, e["radius"], color="tab:blue", lw=1.8, label="interpolated + smoothed (condition)")
        sr = s[e["aneurysm"]]
        ax.axvspan(sr.min(), sr.max(), color="tab:red", alpha=0.12, label="aneurysm region")
        ax.set_title(f"{e['kind']} group {e['group']}  ({s[-1]:.1f} mm)", fontsize=9)
        ax.set_xlabel("arc length (mm)", fontsize=8)
        ax.set_ylabel("radius (mm)", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(rows):]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=7)
    fig.suptitle(case, fontsize=10)
    fig.tight_layout()
    fig.savefig(path_png, dpi=130)
    plt.close(fig)


def render_skeleton(case, out, path_png):
    """Skeleton from the complete graph (bifurcation pieces included) as radius tubes."""
    import os
    import pyvista as pv
    import trimesh
    if not os.environ.get("DISPLAY"):
        pv.start_xvfb()
    g = np.load(out / "skeleton_graph.npy", allow_pickle=True).item()
    orig = trimesh.load(out / "original.ply", process=False)
    sac = trimesh.load(out / "sac.ply", process=False)
    patched = trimesh.load(out / "patched.ply", process=False)
    focal = sac.vertices.mean(0)
    dist = max(22.0, 6.0 * np.linalg.norm(sac.vertices - focal, axis=1).max())
    allc = np.vstack([e["pts"] for e in g["edges"]])
    outward = focal - allc[np.argmin(np.linalg.norm(allc - focal, axis=1))]
    outward /= np.linalg.norm(outward) + 1e-9
    side = np.cross(outward, [0, 0, 1.0])
    side /= np.linalg.norm(side) + 1e-9

    def tubes(key, color):
        items, ends = [], {}
        for e in g["edges"]:
            line = pv.lines_from_points(e["pts"])
            line["radius"] = e[key]
            items.append((line.tube(scalars="radius", absolute=True, n_sides=24),
                          dict(color=color, smooth_shading=True)))
            ends.setdefault(e["u"], []).append(e[key][0])
            ends.setdefault(e["v"], []).append(e[key][-1])
        # a ball at each junction closes the joint between the tubes meeting there
        for n, rs in ends.items():
            if len(rs) >= 2:
                items.append((pv.Sphere(radius=float(np.mean(rs)), center=g["nodes"][n]),
                              dict(color=color, smooth_shading=True)))
        return items + [(hv.to_pv(sac), dict(color="crimson", opacity=0.15))]

    panels = [("original + sac", [(hv.to_pv(orig), dict(color="lightgrey", smooth_shading=True)),
                                  (hv.to_pv(sac), dict(color="crimson", smooth_shading=True))]),
              ("patched mesh", [(hv.to_pv(patched), dict(color="lightgrey", smooth_shading=True))]),
              ("skeleton, raw radius", tubes("radius_raw", "tab:orange")),
              ("skeleton, interpolated radius", tubes("radius", "steelblue"))]
    # every view from the sac side, so no other vessel sits between the camera and the patch
    dirs = [outward, (outward + side) / np.sqrt(2), (outward - side) / np.sqrt(2)]
    hv.render_panels(path_png, panels, focal, dirs, dist=dist, title=case)
    pv.close_all()


def plot_graph(case, out, path_png):
    """The 1D graph coloured by radius: overview plus a close-up of every bifurcation node (own
    colour range, so any jump in radius shows), and the aneurysm region."""
    import os
    import pyvista as pv
    if not os.environ.get("DISPLAY"):
        pv.start_xvfb()
    g = np.load(out / "skeleton_graph.npy", allow_pickle=True).item()
    nodes = g["nodes"]
    deg = np.bincount([x for e in g["edges"] for x in (e["u"], e["v"])], minlength=len(nodes))
    allp = np.vstack([e["pts"] for e in g["edges"]])
    allr = np.concatenate([e["radius"] for e in g["edges"]])

    def coloured(clim, lw, bar_title):
        # pyvista shares one scalar bar per title across subplots, so every panel gets its own
        items = []
        for k, e in enumerate(g["edges"]):
            ln = pv.lines_from_points(e["pts"])
            ln[bar_title] = e["radius"]
            items.append((ln, dict(scalars=bar_title, cmap="turbo", clim=clim, line_width=lw,
                                   show_scalar_bar=k == 0,
                                   scalar_bar_args=dict(vertical=True, n_labels=4, fmt="%.2f",
                                                        label_font_size=10, title_font_size=10))))
        for d, col in ((1, "black"), (3, "white")):
            sel = (deg == 1) if d == 1 else (deg >= 3)
            if sel.any():
                items.append((pv.PolyData(nodes[sel]), dict(color=col, point_size=10 if d == 1 else 13,
                                                             render_points_as_spheres=True)))
        return items

    bif = np.flatnonzero(deg >= 3)
    center = nodes.mean(0)
    span = float(np.linalg.norm(nodes - center, axis=1).max())
    pca = np.linalg.svd(allp - allp.mean(0), full_matrices=False)[2]
    shape = (1 + int(np.ceil(len(bif) / 3)), 3)
    pl = pv.Plotter(off_screen=True, shape=shape, window_size=(620 * shape[1], 560 * shape[0]), border=True)

    def place(c, dist, lab):
        pl.camera.focal_point = c
        pl.camera.position = c + dist * pca[2]
        pl.camera.up = pca[0]
        pl.add_text(lab, font_size=9)

    full = (float(allr.min()), float(allr.max()))
    pl.subplot(0, 0)
    for obj, kw in coloured(full, 5, "radius (mm)"):
        pl.add_mesh(obj, **kw)
    place(center, 2.6 * span, f"{case}\nradius (condition: interpolated + smoothed); white = bifurcation, black = end")
    pl.subplot(0, 1)
    for e in g["edges"]:
        pl.add_mesh(pv.lines_from_points(e["pts"]), color="darkgrey", line_width=4)
        if e["aneurysm"].sum() > 1:
            pl.add_mesh(pv.lines_from_points(e["pts"][e["aneurysm"]]), color="red", line_width=7)
    place(center, 2.6 * span, "red = aneurysm region (radius interpolated)")
    pl.subplot(0, 2)
    for e in g["edges"]:
        ln = pv.lines_from_points(e["pts"])
        ln["raw radius (mm)"] = e["radius_raw"]
        pl.add_mesh(ln, scalars="raw radius (mm)", cmap="turbo", clim=full, line_width=5,
                    show_scalar_bar=e is g["edges"][0],
                    scalar_bar_args=dict(vertical=True, n_labels=4, fmt="%.2f", label_font_size=10,
                                         title_font_size=10))
    place(center, 2.6 * span, "raw radius (before interpolation), same colours")
    for k, n in enumerate(bif):
        pl.subplot(1 + k // 3, k % 3)
        near = np.linalg.norm(allp - nodes[n], axis=1) < 3.0
        local = (float(allr[near].min()), float(allr[near].max()))
        for obj, kw in coloured(local, 8, f"radius near node {n} (mm)"):
            pl.add_mesh(obj, **kw)
        place(nodes[n], 7.0, f"bifurcation node {n}: 3 mm around it, colour range of this area")
    pl.screenshot(str(path_png))
    pl.close()


def skeleton(out, inp):
    return run([PY_VMTK, str(REPO / "geometry/extract_skeleton.py"), "--mesh", str(out / "patched.ply"),
                "--endpoints", str(inp / "endpoints_manual.npy"), "--zone", str(out / "zone_pts.npy"),
                "--out", str(out / "skeleton_graph.npy")]).strip().splitlines()[-1]


def plot_skeleton_sanity(case, out, path_png):
    """1D graph (radius) + ground-truth ostium ring + dome (transparent): one overview of the
    whole tree, then three close-ups around the ostium."""
    import pyvista as pv
    import trimesh
    from geometry import load_point_graph
    if not os.environ.get("DISPLAY"):
        pv.start_xvfb()
    pg = load_point_graph(out / "skeleton_graph_points.npz")
    o = np.load(out / "ostium.npz", allow_pickle=True)
    ring, c, n = o["ring"], o["centroid"], o["normal"]
    F = o["faces"]
    dome = hv.to_pv(trimesh.Trimesh(o["verts"], F[o["dome"][F].all(1)], process=False))
    ei = pg["edge_index"][:, pg["edge_index"][0] < pg["edge_index"][1]]
    clim = (float(pg["radius"].min()), float(pg["radius"].max()))
    loop = pv.lines_from_points(np.vstack([ring, ring[:1]]))

    def scene(pl, bar):
        graph = pv.PolyData(pg["pos"].astype(float))
        graph.lines = np.hstack([np.full((ei.shape[1], 1), 2), ei.T]).ravel()
        graph["radius (mm)"] = pg["radius"]
        pl.add_mesh(graph, scalars="radius (mm)", cmap="turbo", clim=clim, line_width=6,
                    render_lines_as_tubes=True, show_scalar_bar=bar,
                    scalar_bar_args=dict(vertical=True, n_labels=4, fmt="%.2f"))
        pl.add_mesh(dome, color="darkorange", opacity=0.3, smooth_shading=True)
        pl.add_mesh(loop, color="black", line_width=6, render_lines_as_tubes=True)
        pl.add_mesh(pv.Arrow(start=c, direction=n, scale=2.0), color="black")
        for sel, col in ((pg["is_junction"], "white"), (pg["is_end"], "black")):
            if sel.any():
                pl.add_mesh(pv.PolyData(pg["pos"][sel].astype(float)), color=col, point_size=11,
                            render_points_as_spheres=True)

    side = np.cross(n, [0, 0, 1.0])
    side /= np.linalg.norm(side) + 1e-9
    allp = pg["pos"].astype(float)
    pca = np.linalg.svd(allp - allp.mean(0), full_matrices=False)[2]
    span = float(np.linalg.norm(allp - allp.mean(0), axis=1).max())
    close = max(14.0, 4.0 * float(o["spans"][0]))
    views = [("whole graph", allp.mean(0), pca[2], 2.6 * span, pca[0]),
             ("ostium, from outside the sac", c, n, close, side),
             ("ostium, side", c, side, close, n),
             ("ostium, other side", c, -side, close, n)]
    pl = pv.Plotter(off_screen=True, shape=(2, 2), window_size=(1400, 1260), border=True)
    for k, (lab, focal, d, dist, up) in enumerate(views):
        pl.subplot(k // 2, k % 2)
        scene(pl, bar=k == 0)
        pl.camera.focal_point = focal
        pl.camera.position = focal + dist * d
        pl.camera.up = up
        pl.add_text(f"{case}\n{lab}" if k == 0 else lab, font_size=9)
    pl.screenshot(str(path_png))
    pl.close()


def figures(case, out):
    g = np.load(out / "skeleton_graph.npy", allow_pickle=True).item()
    render_skeleton(case, out, out / "skeleton.png")
    plot_graph(case, out, out / "graph_1d.png")
    plot_profiles(case, g, out / "radius_profiles.png")
    if (out / "ostium.npz").exists():
        plot_skeleton_sanity(case, out, out / "skeleton_sanity.png")


def run_case(case, a):
    t0 = time.time()
    out = Path(a.out_root) / case
    inp = out / "input"
    copy_inputs(case, Path(a.geometry_root), inp)
    info = dict(case=case)
    info["aneurysm_type"] = int(np.load(inp / "endpoints_manual.npy", allow_pickle=True).item()
                                .get("aneurysm_type", -1))
    info.update(patch(case, inp, out, a))
    info["graph"] = json.loads(skeleton(out, inp))
    figures(case, out)
    info["seconds"] = round(time.time() - t0, 1)
    (out / "skeleton_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cases", nargs="+", required=True)
    p.add_argument("--geometry_root", default=str(GEOMETRY_ROOT))
    p.add_argument("--dataset", default="ImperialNHS", help="assembled dataset of the GHD fits")
    p.add_argument("--out_root", default=str(OUT_ROOT))
    p.add_argument("--collar_mm", type=float, default=1.0)
    p.add_argument("--blend_mm", type=float, default=1.5)
    p.add_argument("--h", type=float, default=0.1)
    p.add_argument("--tube", default="ellipse", choices=["ellipse", "circle", "profile"])
    p.add_argument("--face_factor", type=float, default=3.0,
                   help="decimate the patched mesh to this many times the original face count")
    p.add_argument("--plot_only", action="store_true", help="only redraw the figures")
    a = p.parse_args()
    for case in a.cases:
        try:
            if a.plot_only:
                figures(case, Path(a.out_root) / case)
            else:
                run_case(case, a)
        except Exception as e:  # keep going; report at the end
            print(json.dumps(dict(case=case, error=repr(e)[:800])), flush=True)


if __name__ == "__main__":
    main()
