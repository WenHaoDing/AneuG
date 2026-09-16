"""Stage fix cases that have no CFD source folder, rebuilding the mesher's opening file.

stage_fix_cases.py copies each case's CFD source from angioflow/cfd/ImperialNHS. Nine
flagged cases are AneuX cases carried in the ImperialNHS registry, and have no folder
there. What the volume mesher (cfd_mesher/generate_cfd_volume_meshes.py) needs can be
rebuilt from the case's own backup folder instead:

    geomagic_processed.obj        the surface          copied as-is
    ghd_forward_fusion_info.npz   which opening is     written here, from
                                  inlet and outlets    wall_mesh_registration.pt

HOW THE OPENINGS ARE FOUND. wall_mesh_registration.pt marks the CFD mesh's boundary rings
by node_type: 2 is the inlet (Fluent zone 4), 3 the first outlet (zone 5), 4 the second
(zone 6). Its positions are in metres, in the same frame as the surface, which is in
millimetres. Checked on all nine cases: each ring, scaled to mm, lands 0.01-0.15 mm from
an open boundary of geomagic_processed.obj, with the next boundary at least 6.4 mm away.

The ring only says WHICH boundary is the inlet or an outlet. The centroid, radius and
normal written to the npz are measured on the surface's own boundary loop, because the
volume mesher labels caps against that surface, not against the registration mesh.

WHAT IS WRITTEN. The keys the volume mesher reads, in its convention (index 0 = inlet):

    opening_centroids   (n, 3) mm   centroid of each open boundary of the surface
    opening_normals     (n, 3)      unit normal, pointing out of the vessel
    opening_radii       (n,)   mm   mean distance of the boundary loop from its centroid
    cpcd_glo            object (n,) two-point STUB per branch, ending exactly on the
                                    centroid. The mesher only reads the last point, to
                                    label caps; there is no real centerline behind it.
    cpcd_glo_tangent    object (n,) the normal, repeated
    roles               (n,)        inlet, outlet1[, outlet2]
    source              str         where the openings came from

A one-outlet case has two rings and gets two entries; the mesher and fluent_setup.py
treat that as a sidewall configuration.

    python fix_cfd/stage_from_registration.py --dry_run
    python fix_cfd/stage_from_registration.py
"""

import argparse
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

BACKUP_ROOT = "/media/yaplab2/wd8tb/wenhao/datasets/angioflowv2_merged_fix_backup"
DEST_ROOT = "/media/yaplab2/wd8tb/wenhao/angioflow/cfd/AneuX_fix"
MANIFEST = "fix_manifest.csv"

SURFACE = "geomagic_processed.obj"
REGISTRATION = "wall_mesh_registration.pt"
FUSION_INFO = "ghd_forward_fusion_info.npz"

RING_TYPES = [(2, "inlet"), (3, "outlet1"), (4, "outlet2")]
M_TO_MM = 1000.0


def surface_boundaries(mesh):
    """Every open boundary loop of the surface: (centroid, points)."""
    fe = mesh.extract_feature_edges(boundary_edges=True, feature_edges=False,
                                    manifold_edges=False, non_manifold_edges=False).connectivity()
    loops = []
    for g in np.unique(fe.point_data["RegionId"]):
        pts = np.asarray(fe.points[fe.point_data["RegionId"] == g], dtype=float)
        loops.append((pts.mean(0), pts))
    return loops


def outward_normal(loop_pts, centroid, surface_pts, radius):
    """Plane normal of a boundary loop, turned to point out of the vessel.

    The vessel wall next to an opening lies on its inner side, so the surface points
    near the loop (the loop's own vertices excluded) tell which way is inward.
    """
    _, _, vt = np.linalg.svd(loop_pts - centroid, full_matrices=False)
    n = vt[-1]
    near = surface_pts[np.linalg.norm(surface_pts - centroid, axis=1) < 3.0 * radius]
    off_plane = (near - centroid) @ n
    inside = off_plane[np.abs(off_plane) > 0.05 * radius]
    if len(inside) and np.median(inside) > 0:
        n = -n
    return n / np.linalg.norm(n)


def build_openings(case_dir):
    import pyvista as pv
    import torch

    reg = torch.load(os.path.join(case_dir, REGISTRATION), map_location="cpu", weights_only=False)
    nt = reg["node_type"].numpy()
    pos = reg["pos"].numpy().astype(float) * M_TO_MM

    mesh = pv.read(os.path.join(case_dir, SURFACE)).clean().triangulate()
    if mesh.n_points == 0:
        raise ValueError("%s is empty" % SURFACE)
    surf_pts = np.asarray(mesh.points, dtype=float)
    loops = surface_boundaries(mesh)

    rings = [(k, role, pos[nt == k]) for k, role in RING_TYPES if (nt == k).any()]
    if len(rings) != len(loops):
        raise ValueError("%d registration rings but %d open boundaries on the surface"
                         % (len(rings), len(loops)))

    used, out = set(), []
    for k, role, ring in rings:
        ring_c = ring.mean(0)
        ring_r = float(np.linalg.norm(ring - ring_c, axis=1).mean())
        d = [np.linalg.norm(c - ring_c) for c, _ in loops]
        j = int(np.argmin(d))
        if j in used or d[j] > ring_r:
            raise ValueError("%s ring does not match a unique surface boundary (nearest %.2f mm, "
                             "ring radius %.2f mm)" % (role, d[j], ring_r))
        used.add(j)
        c, pts = loops[j]
        radius = float(np.linalg.norm(pts - c, axis=1).mean())
        n = outward_normal(pts, c, surf_pts, radius)
        out.append({"role": role, "centroid": c, "normal": n, "radius": radius,
                    "match_mm": float(d[j])})
    return out


def write_fusion_info(path, openings):
    stub = [np.stack([o["centroid"] - min(1.0, o["radius"]) * o["normal"], o["centroid"]])
            for o in openings]
    np.savez(path,
             opening_centroids=np.array([o["centroid"] for o in openings]),
             opening_normals=np.array([o["normal"] for o in openings]),
             opening_radii=np.array([o["radius"] for o in openings]),
             cpcd_glo=np.array(stub + [None], dtype=object)[:-1],     # keep a 1-D object array
             cpcd_glo_tangent=np.array([np.stack([o["normal"]] * 2) for o in openings] + [None],
                                       dtype=object)[:-1],
             roles=np.array([o["role"] for o in openings]),
             source=np.array("rebuilt from %s rings; openings measured on %s"
                             % (REGISTRATION, SURFACE)),
             allow_pickle=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cases", nargs="*", default=None,
                   help="case names (default: every case the staging manifest could not copy)")
    p.add_argument("--dest", default=DEST_ROOT)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    if args.cases:
        cases = args.cases
    else:
        m = pd.read_csv(os.path.join(BACKUP_ROOT, MANIFEST)).drop_duplicates("case", keep="last")
        cases = m[m["copy"].astype(str).str.startswith("missing")]["case"].tolist()
    print("%d case(s) to stage into %s" % (len(cases), args.dest))

    rows = []
    for case in cases:
        src = os.path.join(BACKUP_ROOT, case)
        dst = os.path.join(args.dest, case)
        try:
            openings = build_openings(src)
        except Exception as exc:
            print("  ! %s: %s" % (case, exc))
            rows.append({"case": case, "status": "failed: %s" % exc})
            continue
        desc = ", ".join("%s r%.2f (%.3f mm)" % (o["role"], o["radius"], o["match_mm"])
                         for o in openings)
        status = "staged"
        if os.path.exists(os.path.join(dst, FUSION_INFO)) and not args.overwrite:
            status = "already staged"
        elif not args.dry_run:
            os.makedirs(dst, exist_ok=True)
            shutil.copy2(os.path.join(src, SURFACE), os.path.join(dst, SURFACE))
            write_fusion_info(os.path.join(dst, FUSION_INFO), openings)
        print("  %-16s %-14s %d openings: %s" % (case, status, len(openings), desc))
        rows.append({"case": case, "status": status, "n_openings": len(openings),
                     "roles": "+".join(o["role"] for o in openings),
                     "surface_from": os.path.join(src, SURFACE), "staged_to": dst})

    if args.dry_run:
        print("--dry_run: nothing written")
        return
    df = pd.DataFrame(rows)
    df.insert(0, "staged_at", time.strftime("%Y-%m-%d %H:%M:%S"))
    path = os.path.join(args.dest, MANIFEST)
    os.makedirs(args.dest, exist_ok=True)
    df.to_csv(path, mode="a", index=False, header=not os.path.exists(path))
    print("manifest: %s" % path)
    if any(str(r["status"]).startswith("failed") for r in rows):
        sys.exit(1)


if __name__ == "__main__":
    main()
