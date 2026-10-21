"""Regenerate the cap regions of generated (AneuGv2) shapes and save them per case.

WHY. gen_merged_cohort.py asks the morphology sensor where each opening's cap is
(SensorUncapper.uncap -> per-branch vertex masks on the GHD topology), uses the masks to
cut the dome open and attach a tube per branch, and then discards them. Downstream work on
dome.obj needs to know which part of the dome is a cap and WHICH opening each cap is, so
this script recomputes the masks and writes them to `cap_masks.npz` in each case folder.

WHY IT REPRODUCES THE ORIGINAL. uncap() is called exactly as the generator calls it
(sample=False, default region settings), on the same coefficients (ghd.npy), aneurysm type
and sensor checkpoint (both recorded in meta.json). Nothing random is involved.

THE CORRESPONDENCE. The generator handles openings in the sensor's BRANCH ORDER throughout:
cap b is lofted into tube b, and openings.npy's `roles[b]` names it -- branch 0 is always the
inlet. So cap_masks[b] belongs to roles[b]. The same opening carries different names in
different files, and all of them are written out so nothing has to be inferred:

    branch b   roles (openings.npy)   volume-mesh / Fluent id   wall_data.pt zone
    0          inlet                  4                         inlet
    1          outlet1                5                         outlet_0
    2          outlet2                6                         outlet_1       (bifurcated only)

VERIFIED PER CASE, against data the generator saved, before anything is written:

  * neck check -- the centroid of each regenerated cap's rim on dome.obj must sit on the
    stored `neck_centroids[b]` of the SAME branch, and be clearly closer to it than to any
    other branch's neck. (The stored necks were measured after a small rim planarization,
    so they agree to a fraction of a millimetre, not exactly.)
  * merged-mesh check -- mesh.obj is the dome with these caps cut out and tubes attached,
    and kept dome vertices are copied into it unchanged except within a few rings of a rim
    (seam smoothing). So every kept vertex 6+ rings from a rim must have an EXACT twin in
    mesh.obj, and no cap-interior vertex may have one. A proximity test does not work here:
    the tube wall starts at the rim and passes within 0.25 mm of much of the cap.

A case failing either check is reported and NOT written.

OUTPUT -- <case>/cap_masks.npz (see docs/cap_masks.md for how to use it):

    roles               (n,)      'inlet', 'outlet1'[, 'outlet2']   -- branch order
    fluent_ids          (n,)      4, 5[, 6]
    wall_zones          (n,)      'inlet', 'outlet_0'[, 'outlet_1']
    vertex_masks        (n, V)    bool, True = dome.obj vertex inside cap b
    face_masks          (n, F)    bool, True = dome.obj face removed for cap b
                                  (all three vertices in the cap -- the generator's rule)
    wall_vertex_mask      (V,)    bool, the clean wall: outside every cap, within
                                  off_surface_tol_mm of mesh.obj's surface, and only the
                                  SINGLE LARGEST connected piece (no stray islands)
    wall_face_mask        (F,)    bool, faces whose three vertices are all wall
    excluded_vertex_masks (n, V)  bool, every non-wall vertex, assigned to its nearest cap b
                                  by mesh-edge distance (cap + off-surface band + islands)
    excluded_face_masks   (n, F)  bool, non-wall faces touching zone b
    surface_distance_mm   (V,)    every dome.obj vertex's distance to mesh.obj's surface,
                                  so a different threshold can be applied downstream
    off_surface_tol_mm            the threshold used (default 0.2)
    rim_loops           (n,)      object: ordered dome.obj vertex indices around cap b
    rim_centroids       (n, 3)    centroid of each rim on dome.obj
    neck_centroids      (n, 3)    the generator's own, copied from openings.npy
    neck_distance_mm    (n,)      |rim_centroid - neck_centroid|, the verification
    dome_n_vertices, dome_n_faces, aneurysm_type, sensor, ...      provenance

    python scripts/generate/regen_cap_masks.py --cases cohort_650_0000_t0 --sanity
    python scripts/generate/regen_cap_masks.py                 # every case under --root
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_ROOT = Path("/media/yaplab2/wd8tb/wenhao/angioflow/cfd/AneuGv2")
CANONICAL_ROOT = ROOT / "dataset" / "canonical"
OUT_NAME = "cap_masks.npz"
SANITY_NAME = "uncap_sanity.png"

# role -> (volume-mesh / Fluent boundary id, wall_data.pt zone name)
ROLE_IDS = {"inlet": (4, "inlet"), "outlet1": (5, "outlet_0"), "outlet2": (6, "outlet_1")}

# verification thresholds
MAX_NECK_MM = 1.0          # rim centroid vs stored neck centroid, same branch
MIN_NECK_MARGIN = 2.0      # nearest OTHER neck must be at least this many times further
TWIN_TOL_MM = 1e-4         # "exact twin" in mesh.obj (copies are bit-identical; 0 found
                           # within 1e-3 mm for any deleted vertex on the test cases)
FAR_RING = 6               # seam smoothing moves rings 1-5 from a rim; beyond, exact copies
MIN_AGREEMENT = 0.999      # share that must behave as expected on each side
OFF_SURFACE_TOL_MM = 0.2   # a kept dome vertex further than this from mesh.obj's surface
                           # (seam smoothing near a rim) joins that cap's exclusion zone


def rings_from_rims(faces, rim, in_cap, n):
    """Edge-ring distance of every non-cap vertex from the nearest rim (rim = 0)."""
    import scipy.sparse as sp
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    A = sp.coo_matrix((np.ones(len(e) * 2), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
                      shape=(n, n)).tocsr()
    ring = np.full(n, -1)
    ring[rim] = 0
    front, k = rim.copy(), 0
    while front.any():
        k += 1
        nxt = (A @ front.astype(np.int8)) > 0
        nxt &= (ring < 0) & ~in_cap
        ring[nxt] = k
        front = nxt
    ring[ring < 0] = 10 ** 6                      # unreachable = far
    return ring


def read_obj(path):
    V, F = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                V.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                F.append([int(t.split("/")[0]) - 1 for t in line.split()[1:4]])
    return np.asarray(V, float), np.asarray(F, np.int64)


def write_mask(case_dir, rec):
    tmp = case_dir / "cap_masks.writing.npz"
    np.savez(tmp, **rec)
    os.replace(tmp, case_dir / OUT_NAME)        # atomic: never a half-written file


class Regenerator:
    """Holds the reconstruction and one SensorUncapper per checkpoint seen."""

    def __init__(self, device="cpu"):
        import torch
        from models.multi_canonical_ghd_reconstruct import MultiCanonicalGHDReconstruct
        self.torch = torch
        self.device = torch.device(device)
        self.multi_recon = MultiCanonicalGHDReconstruct(canonical_root=CANONICAL_ROOT,
                                                        device=self.device)
        self._uncappers = {}

    def uncapper(self, sensor_path):
        from models.morphoformer import SensorUncapper
        if sensor_path not in self._uncappers:
            self._uncappers[sensor_path] = SensorUncapper.from_checkpoint(
                Path(sensor_path), self.multi_recon, device=self.device)
        return self._uncappers[sensor_path]

    def masks_for(self, phi, atype, sensor_path):
        """Exactly the generator's call: uncap(phi, type) with default arguments."""
        unc = self.uncapper(sensor_path)
        with self.torch.no_grad():
            u = unc.uncap(self.torch.as_tensor(phi, dtype=self.torch.float32,
                                               device=self.device), atype)
        faces = self.multi_recon.get(atype).canonical_Meshes.faces_packed().cpu().numpy()
        caps = np.asarray(u["cap_masks"], bool)
        loops = [np.asarray(unc.rim_loop(caps[b], faces), np.int64) for b in range(len(caps))]
        return caps, loops, faces, np.asarray(u["rim_ok"], bool)


def _adjacency(faces, n):
    import scipy.sparse as sp
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    return sp.coo_matrix((np.ones(len(e) * 2), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
                         shape=(n, n)).tocsr()


def wall_and_zones(caps, faces, surf_d, off_tol):
    """Split dome.obj into one clean wall region and one exclusion zone per opening.

    wall  = the SINGLE LARGEST connected piece of {vertices outside every cap and within
            off_tol of mesh.obj's surface}. Off-surface vertices (seam smoothing near a
            rim) are dropped, and so is any small island the drop leaves cut off, so the
            wall is always one connected surface with no stray fragments.
    zones = everything else, each vertex assigned to its NEAREST cap by mesh-edge distance
            (multi-source BFS over the whole dome), so every excluded vertex keeps the
            role -- inlet / outlet1 / outlet2 -- of the opening it belongs to.
    """
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    n = caps.shape[1]
    A = _adjacency(faces, n)
    cand = ~caps.any(0) & (surf_d <= off_tol)
    idx = np.where(cand)[0]
    sub = A[idx][:, idx]
    ncomp, lab = connected_components(sub, directed=False)
    wall = np.zeros(n, bool)
    if len(idx):
        wall[idx[lab == np.bincount(lab).argmax()]] = True
    dropped_islands = int(ncomp - 1) if len(idx) else 0

    owner = np.full(n, -1)
    for b in range(len(caps)):
        owner[caps[b]] = b
    frontier = [caps[b].copy() for b in range(len(caps))]
    while any(f.any() for f in frontier):
        claimed = np.zeros(n, bool)
        nxt = []
        for b, f in enumerate(frontier):
            reach = ((A @ f.astype(np.int8)) > 0) & (owner < 0) & ~claimed
            claimed |= reach
            nxt.append(reach)
        for b, r in enumerate(nxt):
            owner[r] = b
        frontier = nxt
    zones = np.stack([(owner == b) & ~wall for b in range(len(caps))])
    return wall, zones, dropped_islands


def process_case(regen, case_dir, overwrite=False, sanity=False, off_tol=None):
    off_tol = OFF_SURFACE_TOL_MM if off_tol is None else off_tol
    case_dir = Path(case_dir)
    row = {"case": case_dir.name}
    t0 = time.time()
    if (case_dir / OUT_NAME).exists() and not overwrite:
        return {**row, "status": "exists"}
    meta = json.loads((case_dir / "meta.json").read_text())
    atype = int(meta["aneurysm_type"])
    phi = np.load(case_dir / "ghd.npy")
    op = np.load(case_dir / "openings.npy", allow_pickle=True).item()
    roles = [str(r) for r in op["roles"]]
    necks = np.asarray(op["neck_centroids"], float)

    caps, loops, faces, rim_ok = regen.masks_for(phi, atype, meta["sensor"])
    dV, dF = read_obj(case_dir / "dome.obj")
    n = len(caps)
    if len(roles) != n:
        return {**row, "status": "branch_count_mismatch",
                "error": "%d caps regenerated, %d roles stored" % (n, len(roles))}
    if caps.shape[1] != len(dV) or len(faces) != len(dF) or not np.array_equal(faces, dF):
        return {**row, "status": "topology_mismatch",
                "error": "masks on %d verts / %d faces, dome.obj has %d / %d"
                         % (caps.shape[1], len(faces), len(dV), len(dF))}

    # --- check 1: each regenerated rim lands on the SAME branch's stored neck
    rim_c = np.array([dV[lp].mean(0) for lp in loops])
    D = np.linalg.norm(rim_c[:, None] - necks[None], axis=2)          # rim b vs neck j
    own = np.diag(D)
    others = np.array([np.min(np.delete(D[b], b)) if n > 1 else np.inf for b in range(n)])
    neck_ok = bool((np.argmin(D, axis=1) == np.arange(n)).all()
                   and (own <= MAX_NECK_MM).all() and (others >= MIN_NECK_MARGIN * own).all())

    # --- check 2: the split agrees with what survives in the merged mesh, by EXACT twins.
    # A dome vertex the generator kept is copied into mesh.obj unchanged, unless seam
    # smoothing moved it (only within a few rings of a rim). A cap vertex was deleted, so it
    # has no exact twin anywhere -- even though the tube wall can pass within 0.25 mm of it,
    # which is why a proximity test cannot tell the two apart.
    from scipy.spatial import cKDTree
    mV, _ = read_obj(case_dir / "mesh.obj")
    d, _ = cKDTree(mV).query(dV)
    twin = d < TWIN_TOL_MM
    in_cap = caps.any(0)
    rim_v = np.zeros(len(dV), bool)
    for lp in loops:
        rim_v[lp] = True
    interior = in_cap & ~rim_v                    # rim vertices themselves are kept
    ring = rings_from_rims(dF, rim_v, in_cap, len(dV))
    far_kept = ~in_cap & (ring >= FAR_RING)       # beyond seam smoothing's reach
    kept_ok = float(twin[far_kept].mean()) if far_kept.any() else 1.0
    cut_ok = float(1.0 - twin[interior].mean()) if interior.any() else 1.0
    agreement = min(kept_ok, cut_ok)
    mesh_ok = kept_ok >= MIN_AGREEMENT and cut_ok >= MIN_AGREEMENT

    face_masks = np.stack([caps[b][dF].all(1) for b in range(n)])

    # --- wall + exclusion zones. Kept dome vertices near a rim were moved by seam
    # smoothing during generation, up to ~0.6 mm off the merged surface, decaying to zero
    # ~6 rings out; anything sampled onto them from the real surface would be sampled off
    # target. The wall keeps only on-surface vertices, as ONE connected piece.
    import trimesh
    _, surf_d, _ = trimesh.proximity.closest_point(
        trimesh.Trimesh(mV, read_obj(case_dir / "mesh.obj")[1], process=False), dV)
    wall, zones, islands = wall_and_zones(caps, dF, surf_d, off_tol)
    wall_faces = wall[dF].all(1)
    zone_faces = np.stack([~wall_faces & zones[b][dF].any(1) for b in range(n)])
    row.update(n_caps=n, roles="+".join(roles), neck_mm_max=float(own.max()),
               other_neck_mm_min=float(others.min()), kept_agree=float(kept_ok),
               cut_agree=float(cut_ok), agreement=agreement, rim_ok=bool(rim_ok.all()),
               cap_vertices="/".join(str(int(c.sum())) for c in caps))
    if not (neck_ok and mesh_ok):
        return {**row, "status": "verification_failed",
                "error": "neck_ok=%s mesh_ok=%s" % (neck_ok, mesh_ok),
                "seconds": round(time.time() - t0, 1)}

    rec = {
        "roles": np.array(roles),
        "fluent_ids": np.array([ROLE_IDS[r][0] for r in roles], np.int64),
        "wall_zones": np.array([ROLE_IDS[r][1] for r in roles]),
        "vertex_masks": caps,
        "face_masks": face_masks,
        "wall_vertex_mask": wall,
        "wall_face_mask": wall_faces,
        "excluded_vertex_masks": zones,
        "excluded_face_masks": zone_faces,
        "surface_distance_mm": surf_d.astype(np.float32),
        "off_surface_tol_mm": np.float64(off_tol),
        "rim_loops": np.array(loops + [None], dtype=object)[:-1],
        "rim_centroids": rim_c,
        "neck_centroids": necks,
        "neck_distance_mm": own,
        "dome_n_vertices": np.int64(len(dV)),
        "dome_n_faces": np.int64(len(dF)),
        "aneurysm_type": np.int64(atype),
        "sensor": np.array(str(meta["sensor"])),
        "verification": np.array(json.dumps(
            {"neck_mm_max": float(own.max()), "other_neck_mm_min": float(others.min()),
             "far_kept_with_exact_twin": kept_ok, "cap_interior_without_twin": cut_ok,
             "twin_tol_mm": TWIN_TOL_MM, "far_ring": FAR_RING})),
        "generated_by": np.array("scripts/generate/regen_cap_masks.py"),
    }
    write_mask(case_dir, rec)
    row.update(zone_vertices="/".join(str(int(z.sum())) for z in zones),
               max_kept_off_surface_mm=float(surf_d[wall].max()),
               wall_vertices=int(wall.sum()), dropped_islands=islands)
    if sanity:
        render_sanity(case_dir, dV, dF, caps, zones, loops, roles, necks, own, off_tol)
    return {**row, "status": "written", "seconds": round(time.time() - t0, 1)}


def render_sanity(case_dir, dV, dF, caps, zones, loops, roles, necks, own, off_tol):
    """dome.obj coloured by opening: the generator's exact cap in a strong colour (inlet
    red, outlet1 blue, outlet2 green) and the seam-smoothed band added to its exclusion
    zone in a pale shade of the same colour. Rims in black, the generator's neck centroids
    as black spheres, mesh.obj translucent -- each tube should sprout from its own colour."""
    import pyvista as pv
    if not os.environ.get("DISPLAY"):
        pv.start_xvfb()
    col = {"inlet": "red", "outlet1": "royalblue", "outlet2": "limegreen"}
    pale = {"inlet": "lightpink", "outlet1": "lightskyblue", "outlet2": "palegreen"}
    dome = pv.PolyData(dV, np.hstack([np.full((len(dF), 1), 3), dF]).ravel())
    mV, mF = read_obj(case_dir / "mesh.obj")
    merged = pv.PolyData(mV, np.hstack([np.full((len(mF), 1), 3), mF]).ravel())
    pl = pv.Plotter(off_screen=True, shape=(1, 3), window_size=(2100, 760))
    for i, view in enumerate(("iso", "xy", "yz")):
        pl.subplot(0, i)
        wall_f = ~zones.any(0)[dF].any(1)
        pl.add_mesh(dome.extract_cells(np.where(wall_f)[0]), color="wheat", opacity=0.9)
        for b, r in enumerate(roles):
            cap_cells = np.where(caps[b][dF].all(1))[0]
            band = np.where(zones[b][dF].any(1) & ~caps[b][dF].all(1))[0]
            if len(band):
                pl.add_mesh(dome.extract_cells(band), color=pale[r],
                            label=(r + " band") if i == 0 else None)
            if len(cap_cells):
                pl.add_mesh(dome.extract_cells(cap_cells), color=col[r],
                            label=r if i == 0 else None)
            lp = np.append(loops[b], loops[b][0])
            pl.add_mesh(pv.lines_from_points(dV[lp]), color="black", line_width=3)
            pl.add_mesh(pv.Sphere(radius=0.25, center=necks[b]), color="black")
        pl.add_mesh(merged, color="grey", opacity=0.15)
        pl.view_isometric() if view == "iso" else getattr(pl, "view_" + view)()
        pl.camera.zoom(1.2)
    pl.subplot(0, 0)
    pl.add_legend(size=(0.25, 0.18), bcolor="white")
    pl.add_text("%s  caps (strong) + off-surface band > %.2f mm (pale) on dome.obj\n"
                "rim-to-neck %s mm | caps %s verts, zones %s verts"
                % (case_dir.name, off_tol, " / ".join("%.2f" % x for x in own),
                   "/".join(str(int(c.sum())) for c in caps),
                   "/".join(str(int(z.sum())) for z in zones)), font_size=9)
    pl.screenshot(str(case_dir / SANITY_NAME))
    pl.close()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    p.add_argument("--cases", nargs="*", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--log", default=None,
                   help="per-case CSV log (default <root>/cap_masks_log.csv); give each "
                        "parallel shard its own")
    p.add_argument("--off_surface_tol", type=float, default=OFF_SURFACE_TOL_MM,
                   help="mm: kept dome vertices further than this from mesh.obj's surface "
                        "join their cap's exclusion zone")
    p.add_argument("--sanity", action="store_true",
                   help="also write uncap_sanity.png into each case folder")
    a = p.parse_args()

    root = Path(a.root)
    cases = a.cases or sorted(d.name for d in root.iterdir()
                              if d.is_dir() and (d / "dome.obj").exists())
    regen = Regenerator(a.device)
    rows = []
    for i, c in enumerate(cases, 1):
        try:
            r = process_case(regen, root / c, overwrite=a.overwrite, sanity=a.sanity,
                             off_tol=a.off_surface_tol)
        except Exception as exc:
            r = {"case": c, "status": "failed", "error": repr(exc)}
        rows.append(r)
        print("[%d/%d] %-24s %-20s %s" % (
            i, len(cases), c, r["status"],
            ("rim-neck max %.2f mm (others >= %.1f) | twin check %.1f%% | caps %s -> zones %s"
             % (r["neck_mm_max"], r["other_neck_mm_min"], 100 * r["agreement"],
                r["cap_vertices"], r.get("zone_vertices", "-")))
            if "agreement" in r else r.get("error", "")), flush=True)

    log = Path(a.log) if a.log else root / "cap_masks_log.csv"
    with open(log, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["case", "status", "roles", "n_caps",
                                           "cap_vertices", "zone_vertices",
                                           "max_kept_off_surface_mm", "wall_vertices",
                                           "dropped_islands",
                                           "neck_mm_max", "other_neck_mm_min",
                                           "kept_agree", "cut_agree", "agreement", "rim_ok",
                                           "seconds", "error"], extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    import collections
    print("\n%s | log: %s" % (dict(collections.Counter(r["status"] for r in rows)), log))


if __name__ == "__main__":
    main()
