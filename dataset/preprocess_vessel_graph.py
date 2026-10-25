"""Preprocess fitted aneurysm cases into a 1D vessel graph + the ground-truth ostium.

Design: TASK_vessel_conditioned_aneurysm.md, sections 4 and 5.2. Run in the `new` env (torch,
igl, open3d); VMTK steps run as subprocesses in the vmtk310 env.

Reads (never writes) per case:
  geometry folder  <geometry_dir>/  (from assembled_from.json)
                   big mesh: reconstruction.ply, the mesh the endpoints were picked on
                   (AneuX's is ~10x denser and is decimated first)
                   endpoints_manual.npy (VMTK seeds)
  runtime_dataset/assembled/<dataset>/<case>/   GHD fit + alignment_result.npy
  runtime_dataset/AneuG_morpho/<case>.npy       reviewed dome label (manual_dome)
No label.nrrd: the sac and the ostium come from the GHD fit's dome.

Writes one folder per case, <out_root>/<dataset>/<case>/vessel_graph/ (default out_root:
runtime_dataset/assembled, i.e. next to the final fitting results):
  vessel_graph.npz      the point graph (pos, radius, edge_index, branch, branch_id,
                        branch_onehot, aneurysm, is_junction, is_end, extracted), the raw
                        ostium rim, ostium centroid / normal / spans, site_point (graph point
                        nearest the ostium centre), geo_dist_from_site (mm along the graph from
                        the site; for cropping only -- it reveals the site, never a model
                        input), and meta (JSON)
  failed.json           instead, when a check fails: a degenerate VMTK centerline (<= 10
                        points), no bridge across the sac, or a graph that is not one tree
  skeleton_sanity.png   graph (radius) + raw rim + H=5 Fourier ring + dome
  patch.png             big mesh with the sac (from the fit) | patched mesh
Bulky by-products (decimated mesh, VMTK centerlines, patched mesh, zone) go to
<work_root>/<dataset>/<case>/ and can be deleted.

Steps per case:
  1. place the GHD fit in image space; its cleaned dome patch = sac; its rim = ostium
  2. big mesh (+ decimation for dense meshes); drop endpoints that would seed on the sac
     (safety net, see filter_endpoints); VMTK centerlines of the big mesh (to bridge the gap)
  3. cut the sac (big-mesh faces on the fit's dome) + 1 mm collar, patch the vessel
  4. extract_skeleton on the patched mesh: graph, radius interpolated across the aneurysm
     region, smoothed (positions 0.5 mm, radius 1 mm)
  5. save + figures

  python dataset/preprocess_vessel_graph.py --dataset ImperialNHS --cases 0bjVkRg2pM_aneurysm1
  python dataset/preprocess_vessel_graph.py --dataset AneuX --n_random 4 --seed 0 \
      --out_root runtime_scouting --work_root runtime_scouting/_work
"""
import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

if os.environ.get("DISPLAY", "").startswith("localhost:"):
    del os.environ["DISPLAY"]          # ssh-forwarded X has no OpenGL here; render off-screen

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from geometry import heal_vessel as hv  # noqa: E402
from geometry import load_point_graph  # noqa: E402
from geometry.ostium import (clean_dome_faces, decode_ostium, encode_ostium,  # noqa: E402
                             fitted_ostium, points_in_dome, rim_points)

ASSEMBLED = REPO / "runtime_dataset" / "assembled"
PY_VMTK = "/home/yaplab2/miniconda3/envs/vmtk310/bin/python"
TARGET_EDGE_MM = 0.35        # ImperialNHS meshes: mean edge 0.25-0.41 mm
DENSE_EDGE_MM = 0.25         # decimate meshes finer than this
H = 5


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{Path(cmd[1]).name} failed:\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
    return r.stdout


def mesh_candidates(dataset):
    """reconstruction.ply for both datasets: the endpoints in endpoints_manual.npy were picked on
    it (they sit on it to 0.00 mm). The smaller reconstruction4clipping.ply of ImperialNHS does
    not reach them, so VMTK snapped far endpoints onto whatever surface was nearest -- in
    RcbTSbBFsL_aneurysm1 the sac, which left a stub next to the ostium."""
    return ["reconstruction.ply"]


def prepare_mesh(geo, mesh_name, work):
    """Load the big mesh; decimate it to ~TARGET_EDGE_MM when it is much denser. Returns
    (mesh, path VMTK reads, info)."""
    import trimesh
    m = trimesh.load(geo / mesh_name, process=True)
    edge = float(m.edges_unique_length.mean())
    info = dict(mesh=mesh_name, faces_in=len(m.faces), mean_edge_in=round(edge, 3))
    path = geo / mesh_name
    if edge < DENSE_EDGE_MM:
        target = int(m.area / (TARGET_EDGE_MM ** 2 * np.sqrt(3) / 4))
        m = hv.decimate(m, target)
        path = work / f"{Path(mesh_name).stem}_decimated.ply"
        m.export(path)
        info.update(decimated_to=len(m.faces), mean_edge=round(float(m.edges_unique_length.mean()), 3))
    return m, path, info


class CaseFailed(RuntimeError):
    """A check failed; the case is recorded in failed.json instead of being saved."""


def original_paths(mesh_path, endpoints_path, work, tag):
    npz = work / f"centerlines_orig_{tag}.npz"
    if not npz.exists():
        run([PY_VMTK, str(REPO / "geometry/vmtk_centerlines.py"), "--mesh", str(mesh_path),
             "--endpoints", str(endpoints_path), "--out", str(npz)])
    # guard: VMTK can return a degenerate line -- a straight jump from source to target
    # ("Target not reached") -- without an error. Real lines are resampled every 0.1 mm, so a
    # degenerate one shows as a mean point spacing far above that. Very short real lines (an
    # endpoint next to the source: 0.2-0.4 mm, 0.1 mm spacing) are fine.
    z = np.load(npz)
    lines = [z[f"path{i}_pts"] for i in range(int(z["n_paths"]))]
    bad = [i for i, l in enumerate(lines) if len(l) < 2 or line_spacing(l) > 0.5]
    if bad or not any(line_length(l) > 1.0 for l in lines):
        raise CaseFailed(f"degenerate VMTK centerline on {Path(mesh_path).name}: lines {bad} of "
                         f"{len(lines)} (points {[len(l) for l in lines]})")
    return hv.load_paths(npz)


def line_length(l):
    return float(np.linalg.norm(np.diff(l, axis=0), axis=1).sum()) if len(l) > 1 else 0.0


def line_spacing(l):
    return line_length(l) / max(len(l) - 1, 1)


def filter_endpoints(eps_path, mesh, f_sac, ost, work):
    """Drop endpoints that would seed VMTK on the sac.

    An endpoint is dropped when it lies inside the closed GHD dome (or within 0.5 mm of it), or
    when its nearest big-mesh face is a sac face -- VMTK places each endpoint on the nearest
    surface point, so such a seed lands on the sac (after patching: on the patch, which gave a
    1.6 mm stub next to the ostium in RcbTSbBFsL_aneurysm1). Returns (path to use, removed list).
    The original file is used unchanged when nothing is dropped."""
    import trimesh
    rec = np.load(eps_path, allow_pickle=True).item()
    ep = np.asarray(rec["endpoints"], float)
    in_dome, _, _ = points_in_dome(ep, ost["verts"], ost["faces"], ost["dome"])
    _, _, tri = trimesh.proximity.closest_point(mesh, ep)
    on_sac = np.asarray(f_sac)[tri]
    drop = in_dome | on_sac
    if not drop.any():
        return eps_path, []
    keep = np.flatnonzero(~drop)
    if len(keep) < 2:
        raise CaseFailed(f"fewer than 2 endpoints left after dropping those on the sac ({drop.tolist()})")
    removed = [dict(index=int(i), source=bool(i == 0), inside_dome=bool(in_dome[i]),
                    nearest_face_on_sac=bool(on_sac[i])) for i in np.flatnonzero(drop)]
    out = dict(rec, endpoints=ep[keep], dropped_endpoints=ep[drop], kept_index=keep)
    path = work / "endpoints_filtered.npy"
    np.save(path, out, allow_pickle=True)              # source = first kept endpoint
    return path, removed


def geodesic_from(pg, source):
    """Distance along the graph (mm) from point `source` to every point."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import dijkstra
    pos = np.asarray(pg["pos"], float)
    ei = np.asarray(pg["edge_index"])
    w = np.linalg.norm(pos[ei[0]] - pos[ei[1]], axis=1)
    A = sp.csr_matrix((w, (ei[0], ei[1])), shape=(len(pos), len(pos)))
    return dijkstra(A, indices=int(source))


def process_case(dataset, case, out_root, work_root, figures=True, args=None):
    t0 = time.time()
    a_dir = ASSEMBLED / dataset / case
    geo = Path(json.loads((a_dir / "assembled_from.json").read_text())["geometry_dir"])
    out = Path(out_root) / dataset / case / "vessel_graph"
    work = Path(work_root) / dataset / case
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)

    # 1. GHD fit -> sac (dome patch) and ostium (its rim)
    ost = fitted_ostium(case, dataset)
    dome_faces = clean_dome_faces(ost["faces"], ost["dome"])
    rim = rim_points(ost["verts"], ost["faces"], ost["dome"])

    # 2-3. big mesh, cut the sac, patch; fall back to the larger mesh when needed
    for mesh_name in mesh_candidates(dataset):
        if not (geo / mesh_name).exists():
            continue
        mesh, mesh_path, minfo = prepare_mesh(geo, mesh_name, work)
        c = hv.load_case(geo, mesh_name, mesh=mesh)
        f_sac = hv.locate_sac_from_fit(c, ost["verts"], ost["faces"], dome_faces)
        eps, removed = filter_endpoints(geo / "endpoints_manual.npy", mesh, f_sac, ost, work)
        minfo["endpoints_removed"] = removed
        # bridge centerlines from the ORIGINAL mesh: on a decimated AneuX mesh VMTK returned a
        # degenerate 2-point line (p480_..._LICA_cut2), on the full mesh a proper one. Cached;
        # re-extracted (separate cache) only when endpoints were dropped.
        tag = Path(mesh_name).stem + ("_filtered" if removed else "")
        paths = original_paths(geo / mesh_name, eps, work, tag)
        healed, zone, bridges, notes, f_sac = hv.heal_case(
            c, paths, collar_mm=args.collar_mm, blend_mm=args.blend_mm, h=args.h, mode="ellipse",
            target_faces=3 * len(mesh.faces), f_sac=f_sac)
        if not any("reaches a path end" in n for n in notes):
            break
    healed.export(work / "patched.ply")
    if figures:
        patch_figure(case, mesh, f_sac, healed, ost, out / "patch.png")
    if len(bridges) == 0:
        raise CaseFailed("no bridge: the vessel under the sac would only be carved, not rebuilt")
    np.save(work / "zone_pts.npy", zone.surface_pts)

    # 4. skeleton of the patched mesh (radius interpolated across the aneurysm, smoothed)
    graph_npy = work / "skeleton_graph.npy"
    try:
        cont = json.loads(run([PY_VMTK, str(REPO / "geometry/extract_skeleton.py"),
                               "--mesh", str(work / "patched.ply"), "--endpoints", str(eps),
                               "--zone", str(work / "zone_pts.npy"), "--out", str(graph_npy)])
                          .strip().splitlines()[-1])
    except RuntimeError as e:
        if "degenerate VMTK centerline" in str(e):
            # seen when the patch itself failed (thin sheets), e.g. p518_..._RICA_1_cut2
            raise CaseFailed("degenerate VMTK centerline on the patched mesh (check patch.png)")
        raise
    pg = load_point_graph(work / "skeleton_graph_points.npz")
    if cont["components"] != 1 or cont["cycles"] != 0 or cont["n_edges"] < 1:
        raise CaseFailed(f"graph is not one tree: {cont}")

    # sanity of the ostium code on this graph
    code = encode_ostium(rim, pg, H=H)
    ring = decode_ostium(code, pg, M=256)
    d = np.linalg.norm(ring[:, None] - rim[None], axis=-1)
    fourier_err = float(np.r_[d.min(1), d.min(0)].mean())

    meta = dict(dataset=dataset, case=case, aneurysm_type=int(ost["aneurysm_type"]),
                geometry_dir=str(geo), sac_source="ghd_fit", dome_label="AneuG_morpho manual_dome",
                **minfo, n_bridges=len(bridges), patch_notes=notes,
                patched_watertight=bool(healed.is_watertight), graph=cont,
                ostium_spans_mm=[float(x) for x in ost["spans"]], rim_points=int(len(rim)),
                fourier_H=H, fourier_mean_err_mm=round(fourier_err, 3),
                site_point=int(code["p"]), scale_mm=round(code["scale"], 3),
                units="mm", frame="mesh frame of the geometry folder (origin + ijk * spacing)",
                created=time.strftime("%Y-%m-%d %H:%M:%S"))
    geo = geodesic_from(pg, code["p"])
    meta.update(geo_max_mm=round(float(geo.max()), 2),
                geo_nearest_end_mm=round(float(geo[np.asarray(pg["is_end"])].min()), 2),
                graph_length_mm=round(float(np.linalg.norm(
                    np.asarray(pg["pos"])[pg["edge_index"][0]] - np.asarray(pg["pos"])[pg["edge_index"][1]],
                    axis=1).sum() / 2), 2))
    arrays = {k: v for k, v in pg.items() if k != "meta"}
    np.savez(out / "vessel_graph.npz", **arrays, rim=rim.astype(np.float32),
             site_point=np.int64(code["p"]), geo_dist_from_site=geo.astype(np.float32),
             ostium_centroid=ost["centroid"], ostium_normal=ost["normal"],
             ostium_spans=ost["spans"], meta=np.array(json.dumps(meta)))

    if figures:
        sanity_figure(case, pg, rim, ring, ost, dome_faces, out / "skeleton_sanity.png")
    meta["seconds"] = round(time.time() - t0, 1)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


# ------------------------------------------------------------------ figures

def _start_xvfb():
    import pyvista as pv
    if not os.environ.get("DISPLAY"):
        pv.start_xvfb()
    return pv


def sanity_figure(case, pg, rim, ring, ost, dome_faces, path):
    """Graph (radius) + raw rim (grey) + H=5 Fourier ring (black) + dome (transparent)."""
    import trimesh
    pv = _start_xvfb()
    pos = np.asarray(pg["pos"], float)
    ei = pg["edge_index"][:, pg["edge_index"][0] < pg["edge_index"][1]]
    clim = (float(pg["radius"].min()), float(pg["radius"].max()))
    dome = hv.to_pv(trimesh.Trimesh(ost["verts"], ost["faces"][dome_faces], process=False))
    c, n = ring.mean(0), ost["normal"]
    side = np.cross(n, [0, 0, 1.0])
    side /= np.linalg.norm(side) + 1e-9
    pca = np.linalg.svd(pos - pos.mean(0), full_matrices=False)[2]
    span = float(np.linalg.norm(pos - pos.mean(0), axis=1).max())
    close = max(14.0, 4.0 * float(ost["spans"][0]))
    views = [("whole graph", pos.mean(0), pca[2], 2.6 * span, pca[0]),
             ("ostium, from outside the sac", c, n, close, side),
             ("ostium, side", c, side, close, n), ("ostium, other side", c, -side, close, n)]
    pl = pv.Plotter(off_screen=True, shape=(2, 2), window_size=(1400, 1260), border=True)
    for k, (lab, focal, dvec, dist, up) in enumerate(views):
        pl.subplot(k // 2, k % 2)
        g = pv.PolyData(pos)
        g.lines = np.hstack([np.full((ei.shape[1], 1), 2), ei.T]).ravel()
        g["radius (mm)"] = pg["radius"]
        pl.add_mesh(g, scalars="radius (mm)", cmap="turbo", clim=clim, line_width=6,
                    render_lines_as_tubes=True, show_scalar_bar=k == 0,
                    scalar_bar_args=dict(vertical=True, n_labels=4, fmt="%.2f"))
        pl.add_mesh(dome, color="darkorange", opacity=0.3, smooth_shading=True)
        pl.add_mesh(pv.lines_from_points(np.vstack([rim, rim[:1]])), color="grey", line_width=2)
        pl.add_mesh(pv.lines_from_points(np.vstack([ring, ring[:1]])), color="black", line_width=6,
                    render_lines_as_tubes=True)
        for sel, col in ((pg["is_junction"], "white"), (pg["is_end"], "black")):
            if sel.any():
                pl.add_mesh(pv.PolyData(pos[sel]), color=col, point_size=11, render_points_as_spheres=True)
        pl.camera.focal_point = focal
        pl.camera.position = focal + dist * dvec
        pl.camera.up = up
        pl.add_text(f"{case}\n{lab}" if k == 0 else lab, font_size=9)
    pl.subplot(0, 1)
    pl.add_text("\n\nblack: ostium (Fourier H=5), grey: raw rim, orange: dome", font_size=8)
    pl.screenshot(str(path))
    pl.close()


def patch_figure(case, mesh, f_sac, healed, ost, path):
    _start_xvfb()
    c, n = ost["centroid"], ost["normal"]
    side = np.cross(n, [0, 0, 1.0])
    side /= np.linalg.norm(side) + 1e-9
    grey = dict(color="lightgrey", smooth_shading=True)
    panels = [("big mesh + sac from the GHD fit (red)",
               [(hv.to_pv(hv.submesh(mesh, ~f_sac, False)), grey),
                (hv.to_pv(hv.submesh(mesh, f_sac, False)), dict(color="crimson", smooth_shading=True))]),
              ("patched mesh", [(hv.to_pv(healed), grey)])]
    dirs = [n, (n + side) / np.sqrt(2), (n - side) / np.sqrt(2)]
    hv.render_panels(path, panels, c, dirs, dist=max(18.0, 5.0 * float(ost["spans"][0])), title=case)


# ------------------------------------------------------------------ main

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=["ImperialNHS", "AneuX"])
    p.add_argument("--cases", nargs="*", default=None)
    p.add_argument("--n_random", type=int, default=None, help="pick this many cases at random")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_root", default=str(ASSEMBLED))
    p.add_argument("--work_root", default=str(REPO / "runtime_vessel_graph_work"))
    p.add_argument("--collar_mm", type=float, default=1.0)
    p.add_argument("--blend_mm", type=float, default=1.5)
    p.add_argument("--h", type=float, default=0.1)
    p.add_argument("--no_figures", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()

    cases = a.cases or sorted(x for x in os.listdir(ASSEMBLED / a.dataset) if (ASSEMBLED / a.dataset / x).is_dir())
    if a.n_random:
        cases = list(np.random.default_rng(a.seed).choice(cases, a.n_random, replace=False))
    for case in cases:
        done = Path(a.out_root) / a.dataset / case / "vessel_graph" / "vessel_graph.npz"
        if done.exists() and not a.overwrite:
            print(json.dumps(dict(case=case, skipped="exists")), flush=True)
            continue
        failed = Path(a.out_root) / a.dataset / case / "vessel_graph" / "failed.json"
        try:
            meta = process_case(a.dataset, case, a.out_root, a.work_root, not a.no_figures, a)
            failed.unlink(missing_ok=True)
            print(json.dumps({k: meta[k] for k in ("dataset", "case", "aneurysm_type", "mesh",
                                                   "n_bridges", "patched_watertight",
                                                   "fourier_mean_err_mm", "seconds")}
                             | dict(graph=meta["graph"], notes=meta["patch_notes"],
                                    decimated_to=meta.get("decimated_to"))), flush=True)
        except Exception as e:
            rec = dict(dataset=a.dataset, case=case, error=repr(e)[:600],
                       check_failed=isinstance(e, CaseFailed), trace=traceback.format_exc()[-1200:])
            failed.parent.mkdir(parents=True, exist_ok=True)
            done.unlink(missing_ok=True)                 # never leave a graph that failed a check
            failed.write_text(json.dumps(rec, indent=1))
            print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
