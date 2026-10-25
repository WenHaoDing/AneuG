"""Export each case's parent vessel with its aneurysm sac cut out, for hole filling.

WHY. The healthy vessel a sac grew from is defined here as: the expert's segmentation
with the sac removed and the hole closed. The sac label (2) was drawn with its boundary
where the expert imagined the healthy wall, so the hole left behind already outlines the
healthy vessel; filling it only interpolates a patch whose edge the expert fixed. The
filled patch is also the OSTIUM footprint on the healthy vessel -- the label for site
prediction.

SOURCE. ImperialNHS_v2/<case>/label.nrrd: label 1 = arteries, label 2 = THIS case's sac
only (multi-aneurysm scans are split one sac per case folder; other sacs are background).
Everything is in the scan's own world frame (mm), from the nrrd header.

WHAT IS WRITTEN, per case, into <out>/<case>/:

    parent_with_hole.obj / .stl   the vessel around the sac, sac faces removed: FILL THIS.
                                  The vessel ends at the crop box are capped flat by the
                                  meshing, so the sac hole is the ONLY open hole.
    sac.obj                       the removed sac surface, for reference
    arteries_only_closed.obj      the arteries label alone, meshed. It closes over the
                                  drawn label-1/label-2 interface, so it shows the
                                  expert's healthy wall exactly as drawn (voxel-stepped)
                                  -- a comparison for the fill
    hole_info.json                which open boundary is the sac hole (centroid, perimeter,
                                  neck diameter), sac size, neck ratio, crop, smoothing
    sanity.png                    the vessel (grey), the removed sac (red), the hole rim

HOW. Crop a box around the sac (+ margin), keep the vessel component touching the sac,
mesh label1|label2 with marching cubes, smooth lightly (Taubin, volume-preserving) so
the hole rim is not a voxel staircase, then classify each face by sampling the label just
inside the surface along its inward normal: faces over the sac go, the rest stay.

    python vessel_edit/export_parent_with_hole.py --cases 0bjVkRg2pM_aneurysm1 ...
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

GEOMETRY_ROOT = "/media/yaplab2/wd8tb/wenhao/angioflow/data/geometry/ImperialNHS_v2"
OUT_DEFAULT = Path(__file__).resolve().parents[1] / "runtime_debug" / "healthy_vessel_fill"
VESSEL, SAC = 1, 2


def load_label(case_dir):
    import nrrd
    data, hdr = nrrd.read(str(Path(case_dir) / "label.nrrd"))
    M = np.asarray(hdr["space directions"], float)       # rows: world step per index axis
    origin = np.asarray(hdr.get("space origin", np.zeros(3)), float)
    return data, M, origin


def crop_around_sac(data, M, margin_mm):
    sac = np.argwhere(data == SAC)
    if len(sac) == 0:
        raise ValueError("no sac (label 2) in this label map")
    spacing = np.linalg.norm(M, axis=1)
    pad = np.ceil(margin_mm / spacing).astype(int)
    lo = np.maximum(sac.min(0) - pad, 0)
    hi = np.minimum(sac.max(0) + pad + 1, data.shape)
    return tuple(slice(a, b) for a, b in zip(lo, hi)), lo


def component_touching(mask, seed_mask):
    """Keep only the connected component(s) of `mask` that contain seed voxels."""
    from scipy import ndimage
    lab, n = ndimage.label(mask, structure=np.ones((3, 3, 3)))
    keep = np.unique(lab[seed_mask & (lab > 0)])
    return np.isin(lab, keep[keep > 0])


def mesh_mask(mask, M, origin, offset):
    from skimage import measure
    import trimesh
    padded = np.pad(mask.astype(np.uint8), 1)
    v, f, _, _ = measure.marching_cubes(padded, level=0.5)
    v = origin + (v - 1.0 + offset) @ M                   # index -> world (mm)
    m = trimesh.Trimesh(v, f, process=True)
    if m.volume < 0:
        m.invert()                                        # outward normals
    return m


def boundary_loops(mesh):
    """[(vertex_ids, centroid, perimeter)] for every open boundary loop."""
    import trimesh
    e = mesh.edges_sorted
    uniq, counts = np.unique(e, axis=0, return_counts=True)
    bnd = uniq[counts == 1]
    if len(bnd) == 0:
        return []
    import networkx as nx
    g = nx.Graph()
    g.add_edges_from(bnd.tolist())
    out = []
    for comp in nx.connected_components(g):
        ids = np.array(sorted(comp))
        sub = g.subgraph(comp)
        per = sum(np.linalg.norm(mesh.vertices[a] - mesh.vertices[b]) for a, b in sub.edges())
        out.append((ids, mesh.vertices[ids].mean(0), float(per)))
    return out


def clean_face_labels(mesh, is_sac, passes=3):
    """Faces right at the rim are sometimes misclassified: they leave pinholes in the
    vessel and specks of 'sac' on it. A few rounds of majority vote over neighbouring
    faces settle the rim, and only the largest connected sac region is kept."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    adj = mesh.face_adjacency
    n = len(mesh.faces)
    A = sp.coo_matrix((np.ones(len(adj) * 2), (np.r_[adj[:, 0], adj[:, 1]],
                                               np.r_[adj[:, 1], adj[:, 0]])), shape=(n, n)).tocsr()
    lab = is_sac.copy()
    for _ in range(passes):
        votes = A @ lab.astype(float)
        deg = np.asarray(A.sum(1)).ravel()
        lab = np.where(deg > 0, votes * 2 > deg, lab) | (lab & (votes * 2 == deg))
    idx = np.where(lab)[0]
    if len(idx):
        ncomp, comp = connected_components(A[idx][:, idx], directed=False)
        keep = idx[comp == np.bincount(comp).argmax()]
        lab = np.zeros(n, bool)
        lab[keep] = True
    return lab


def export_case(case, root, out_root, margin_mm, smooth_iters, render=True):
    import trimesh
    case_dir = Path(root) / case
    out = Path(out_root) / case
    out.mkdir(parents=True, exist_ok=True)

    data, M, origin = load_label(case_dir)
    sl, lo = crop_around_sac(data, M, margin_mm)
    crop = data[sl]
    sac = crop == SAC
    union = component_touching((crop == VESSEL) | sac, sac)
    vessel_only = union & ~sac

    m = mesh_mask(union, M, origin, lo)
    if smooth_iters:
        trimesh.smoothing.filter_taubin(m, lamb=0.5, nu=-0.53, iterations=smooth_iters)

    # classify faces: sample the label half a voxel INSIDE the surface
    Minv = np.linalg.inv(M)
    step = 0.6 * np.linalg.norm(M, axis=1).min()
    c = m.triangles_center - step * m.face_normals
    idx = np.rint((c - origin) @ Minv - lo).astype(int)
    idx = np.clip(idx, 0, np.array(crop.shape) - 1)
    lab_in = crop[idx[:, 0], idx[:, 1], idx[:, 2]]
    is_sac = clean_face_labels(m, lab_in == SAC)

    parent = m.submesh([np.where(~is_sac)[0]], append=True)
    parts = parent.split(only_watertight=False)
    parent = max(parts, key=lambda p: len(p.faces))       # drop stray fragments
    sac_mesh = m.submesh([np.where(is_sac)[0]], append=True)

    # The sac hole is the LARGEST open boundary whose rim is shared with the removed sac.
    # (The crop-box ends are capped by the meshing, so normally it is the only one.)
    loops = boundary_loops(parent)
    from scipy.spatial import cKDTree
    sac_tree = cKDTree(sac_mesh.vertices)
    touching = [(ids, cen, per) for ids, cen, per in loops
                if (sac_tree.query(parent.vertices[ids])[0] < 1e-6).mean() > 0.5]
    if not touching:
        raise ValueError("no open boundary is shared with the removed sac")
    hole_ids, hole_c, hole_per = max(touching, key=lambda t: len(t[0]))
    others = [t for t in loops if t[0] is not hole_ids]

    sac_vol = float(sac.sum() * abs(np.linalg.det(M)))
    sac_eqd = 2 * (3 * sac_vol / (4 * np.pi)) ** (1 / 3)
    # neck size from the rim's extent across its own plane: the perimeter of a rim traced
    # on a voxel-derived mesh zigzags and overstates the size ~1.8x
    P = parent.vertices[hole_ids] - parent.vertices[hole_ids].mean(0)
    _, _, vt = np.linalg.svd(P, full_matrices=False)
    spans = np.ptp(P @ vt[:2].T, axis=0)
    neck_d = float(spans.mean())
    info = {
        "case": case, "source": str(case_dir / "label.nrrd"),
        "frame": "scan world frame (mm), from the nrrd header",
        "crop_voxels": [[int(s.start), int(s.stop)] for s in sl], "margin_mm": margin_mm,
        "taubin_iterations": smooth_iters,
        "sac_volume_mm3": sac_vol, "sac_equiv_diameter_mm": sac_eqd,
        "sac_hole": {"centroid_mm": hole_c.tolist(), "neck_diameter_mm": neck_d,
                     "neck_spans_mm": spans.tolist(), "rim_perimeter_mm": hole_per,
                     "n_rim_vertices": int(len(hole_ids))},
        "neck_to_sac_ratio": neck_d / sac_eqd,
        "crop_ends": "capped flat by the meshing -- the sac hole is the only real hole",
        "other_open_boundaries": [{"centroid_mm": c_.tolist(), "perimeter_mm": p_}
                                  for _, c_, p_ in others],
        "n_faces": {"parent_with_hole": int(len(parent.faces)), "sac": int(len(sac_mesh.faces))},
    }
    parent.export(out / "parent_with_hole.obj")
    parent.export(out / "parent_with_hole.stl")
    sac_mesh.export(out / "sac.obj")
    vm = mesh_mask(vessel_only, M, origin, lo)
    if smooth_iters:
        trimesh.smoothing.filter_taubin(vm, lamb=0.5, nu=-0.53, iterations=smooth_iters)
    vm.split(only_watertight=False)
    vm.export(out / "arteries_only_closed.obj")
    (out / "hole_info.json").write_text(json.dumps(info, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    if render:
        render_sanity(out, parent, sac_mesh, hole_ids, case, info)
    return info


def render_sanity(out, parent, sac_mesh, hole_ids, case, info):
    import pyvista as pv
    if not os.environ.get("DISPLAY"):
        pv.start_xvfb()
    to_pv = lambda m: pv.PolyData(np.asarray(m.vertices),
                                   np.hstack([np.full((len(m.faces), 1), 3), m.faces]).ravel())
    pl = pv.Plotter(off_screen=True, shape=(1, 2), window_size=(1800, 850))
    cen = np.asarray(info["sac_hole"]["centroid_mm"])
    for i, show_sac in enumerate((True, False)):
        pl.subplot(0, i)
        pl.add_mesh(to_pv(parent), color="lightgrey", smooth_shading=True)
        if show_sac:
            pl.add_mesh(to_pv(sac_mesh), color="crimson", opacity=0.45)
        pl.add_points(parent.vertices[hole_ids], color="black", point_size=6,
                      render_points_as_spheres=True)
        pl.camera.focal_point = cen
        pl.camera.position = cen + 25 * (cen - parent.vertices.mean(0)) / (
            np.linalg.norm(cen - parent.vertices.mean(0)) + 1e-9) + np.array([0, 0, 10])
        pl.add_text(("with the removed sac (red)" if show_sac else "the hole to fill (rim in black)"),
                    font_size=10)
    pl.subplot(0, 0)
    pl.add_text("\n%s  sac %.1f mm, neck %.1f mm (ratio %.2f)"
                % (case, info["sac_equiv_diameter_mm"], info["sac_hole"]["neck_diameter_mm"],
                   info["neck_to_sac_ratio"]), font_size=9, position="lower_left")
    pl.screenshot(str(out / "sanity.png"))
    pl.close()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cases", nargs="+", required=True)
    p.add_argument("--root", default=GEOMETRY_ROOT)
    p.add_argument("--out", default=str(OUT_DEFAULT))
    p.add_argument("--margin_mm", type=float, default=12.0,
                   help="vessel kept around the sac's bounding box")
    p.add_argument("--smooth_iters", type=int, default=10,
                   help="Taubin iterations on the marching-cubes surface (0 = raw voxel steps)")
    a = p.parse_args()
    for c in a.cases:
        try:
            i = export_case(c, a.root, a.out, a.margin_mm, a.smooth_iters)
            print("%-24s sac %.1f mm | neck %.1f mm (%.1f x %.1f, ratio %.2f) | rim %d verts | "
                  "other open boundaries %d"
                  % (c, i["sac_equiv_diameter_mm"], i["sac_hole"]["neck_diameter_mm"],
                     *i["sac_hole"]["neck_spans_mm"], i["neck_to_sac_ratio"],
                     i["sac_hole"]["n_rim_vertices"], len(i["other_open_boundaries"])), flush=True)
        except Exception as exc:
            print("%-24s FAILED %r" % (c, exc), flush=True)


if __name__ == "__main__":
    main()
