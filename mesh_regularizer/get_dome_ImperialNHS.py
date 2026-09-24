"""Measure each ImperialNHS aneurysm dome from its segmentation, for scale normalization.

The AneuX reference cases carry a dome.ply; the ImperialNHS cases do not. What they do
carry is label.nrrd, the segmentation the geometry was reconstructed from, in which

    label 1   parent vasculature
    label 2   the aneurysm (dome)          <- what this script measures

For every case this writes, into that case's folder in the geometry tree:

    dome.ply        the dome surface, marching cubes on the label-2 mask
    dome_size.npy   a dict of size measures (np.load(..., allow_pickle=True).item())

WHY. A roughness scale that is fixed in millimetres means different things on a 3 mm dome
and a 9 mm one: the same 1 mm bump is gross on the first and slight on the second. To
compare shapes of different calibre, the scales have to be relative to the shape's own
size, and that needs a size per case. dome_size.npy is that number, measured the same way
on every case.

WHAT IS IN dome_size.npy.

    volume_mm3          dome volume, voxel count x voxel volume (the mask, not the mesh)
    equiv_diameter_mm   diameter of the sphere of that volume -- the recommended size,
                        being the least sensitive to a lobulated or elongated sac
    pca_extents_mm      (3,) extent along the dome's own principal axes, largest first
    max_extent_mm       pca_extents_mm[0], the longest straight span of the sac
    bbox_diagonal_mm    world-axis bounding box diagonal (frame-dependent; for reference)
    surface_area_mm2    area of the marching-cubes surface (staircase-biased; reference)
    centroid_mm         (3,) dome centroid in the nrrd's world frame
    voxel_volume_mm3    one voxel, for judging how well the sac is resolved
    n_voxels            label-2 voxel count; a sac of a few hundred voxels is coarse

GEOMETRY. Vertices come out of marching cubes in voxel index coordinates and are mapped to
world by  world = space_origin + index @ space_directions,  which carries the anisotropic
spacing and the oblique scan rotation exactly, so lengths are true millimetres. The nrrd's
frame is left-posterior-superior, the same frame the reconstructions were built in.

The mask is used as segmented, with no smoothing or hole filling: sizes come from voxel
counts rather than the mesh, so a staircase surface does not bias them.

    python mesh_regularizer/get_dome_ImperialNHS.py --dry_run          # what it would do
    python mesh_regularizer/get_dome_ImperialNHS.py --workers 4        # measure everything
    python mesh_regularizer/get_dome_ImperialNHS.py --cases A B        # just these
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

import numpy as np

GEOMETRY_ROOT = "/media/yaplab2/wd8tb/wenhao/angioflow/data/geometry/ImperialNHS_v2"
LABEL_FILE = "label.nrrd"
DOME_LABEL = 2                      # 1 is the parent vasculature
DOME_MESH = "dome.ply"
DOME_SIZE = "dome_size.npy"
SUMMARY = "dome_sizes.csv"


def measure_dome(case_dir, write_mesh=True, overwrite=False):
    """Measure the label-2 mask in <case_dir>/label.nrrd. Returns the size dict."""
    import nrrd

    size_path = os.path.join(case_dir, DOME_SIZE)
    if os.path.exists(size_path) and not overwrite:
        out = np.load(size_path, allow_pickle=True).item()
        out["status"] = "exists"
        return out

    label_path = os.path.join(case_dir, LABEL_FILE)
    data, header = nrrd.read(label_path)
    mask = data == DOME_LABEL
    n_voxels = int(mask.sum())
    if n_voxels == 0:
        return {"status": "no_dome_label", "n_voxels": 0}

    # rows of space directions are the world vectors of one step along each index axis
    M = np.asarray(header["space directions"], dtype=float)
    origin = np.asarray(header.get("space origin", np.zeros(3)), dtype=float)
    voxel_volume = float(abs(np.linalg.det(M)))

    idx = np.argwhere(mask).astype(float)
    pts = origin + idx @ M                       # voxel centres in world mm
    centroid = pts.mean(0)
    # extent along the sac's own axes, which is what "how big is this dome" means for a
    # lobulated sac; the world-axis bbox would depend on how the patient lay in the scanner
    _, s, vt = np.linalg.svd(pts - centroid, full_matrices=False)
    proj = (pts - centroid) @ vt.T
    pca_extents = np.sort(proj.max(0) - proj.min(0))[::-1]

    volume = n_voxels * voxel_volume
    out = {
        "status": "ok",
        "volume_mm3": volume,
        "equiv_diameter_mm": 2.0 * (3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0),
        "pca_extents_mm": pca_extents,
        "max_extent_mm": float(pca_extents[0]),
        "bbox_diagonal_mm": float(np.linalg.norm(pts.max(0) - pts.min(0))),
        "centroid_mm": centroid,
        "voxel_volume_mm3": voxel_volume,
        "n_voxels": n_voxels,
        "label": DOME_LABEL,
        "source": label_path,
        "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    if write_mesh and (overwrite or not os.path.exists(os.path.join(case_dir, DOME_MESH))):
        from skimage import measure as skmeasure
        import trimesh
        # pad so a sac touching the volume edge still closes
        padded = np.pad(mask.astype(np.uint8), 1)
        verts, faces, _, _ = skmeasure.marching_cubes(padded, level=0.5)
        verts = origin + (verts - 1.0) @ M
        mesh = trimesh.Trimesh(verts, faces, process=True)
        mesh.export(os.path.join(case_dir, DOME_MESH))
        out["surface_area_mm2"] = float(mesh.area)
        out["mesh_volume_mm3"] = float(abs(mesh.volume))

    np.save(size_path, out, allow_pickle=True)
    return out


def _job(args):
    case, case_dir, write_mesh, overwrite = args
    t0 = time.time()
    try:
        out = measure_dome(case_dir, write_mesh=write_mesh, overwrite=overwrite)
    except Exception as exc:
        return case, {"status": "error: %r" % exc}, time.time() - t0
    return case, out, time.time() - t0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=GEOMETRY_ROOT)
    p.add_argument("--cases", nargs="*", default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no_mesh", action="store_true",
                   help="write dome_size.npy only, skipping the dome.ply surface")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    cases = args.cases or sorted(c for c in os.listdir(args.root)
                                 if os.path.isdir(os.path.join(args.root, c)))
    jobs = []
    missing = []
    for c in cases:
        d = os.path.join(args.root, c)
        if os.path.exists(os.path.join(d, LABEL_FILE)):
            jobs.append((c, d, not args.no_mesh, args.overwrite))
        else:
            missing.append(c)
    print("%d case(s) with %s, %d without" % (len(jobs), LABEL_FILE, len(missing)), flush=True)
    if missing[:5]:
        print("  no label.nrrd: %s%s" % (", ".join(missing[:5]),
                                         " ..." if len(missing) > 5 else ""))
    if args.dry_run:
        print("--dry_run: nothing written")
        return

    rows, bad = [], []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
        futs = [pool.submit(_job, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            case, out, secs = fut.result()
            if out["status"] not in ("ok", "exists"):
                bad.append((case, out["status"]))
                print("[%d/%d] %s %s" % (i, len(jobs), case, out["status"]), flush=True)
                continue
            rows.append({"case": case, "equiv_diameter_mm": round(out["equiv_diameter_mm"], 3),
                         "max_extent_mm": round(out["max_extent_mm"], 3),
                         "volume_mm3": round(out["volume_mm3"], 2),
                         "n_voxels": out["n_voxels"],
                         "voxel_volume_mm3": round(out["voxel_volume_mm3"], 5)})
            if i % 25 == 0 or i == len(jobs):
                print("[%d/%d] %s dome equiv diameter %.2f mm (%.0f s)"
                      % (i, len(jobs), case, out["equiv_diameter_mm"], secs), flush=True)

    if rows:
        rows.sort(key=lambda r: r["equiv_diameter_mm"])
        path = os.path.join(args.root, SUMMARY)
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        d = np.array([r["equiv_diameter_mm"] for r in rows])
        print("\nmeasured %d dome(s): equivalent diameter median %.2f mm, range %.2f-%.2f mm"
              % (len(d), np.median(d), d.min(), d.max()))
        print("ratio largest / smallest: %.1fx" % (d.max() / d.min()))
        print("summary -> %s" % path)
    if bad:
        print("failed or no dome label: %d" % len(bad))
        for c, s in bad[:10]:
            print("  %s: %s" % (c, s))
        sys.exit(1)


if __name__ == "__main__":
    main()
