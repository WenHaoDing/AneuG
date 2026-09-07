#!/usr/bin/env python3
"""Does re-clipping the centerline to a MANUAL mesh fix its caps?

A "_manual" clip is hand-cut, so its boundary openings sit somewhere else than
the automatic clip's -- on p171, 3.5-5.6 mm away. But resolve_case_paths only
swaps the MESH to the manual variant; clipped_centerline/branch_ranking still
come from the automatic cut. _build_closed_case_mesh then extrudes from
centerline endpoints that don't lie on the mesh's actual openings.

This re-clips the existing merged centerline onto the manual mesh's own
openings (vessel_clipping.clip_centerline_to_mesh -- the mesh is never
touched), re-caps with it, and compares against the stale-centerline cap.

Works on a COPY of the case; the real geometry dir is never written to.
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "AneuSeg"))

from ghd.fitting.alignment import resolve_case_paths, _build_closed_case_mesh  # noqa: E402
from IAgents.tools.vessel_clipping import clip_centerline_to_mesh  # noqa: E402


def health(m, tag):
    V, F = np.asarray(m.vertices), np.asarray(m.faces)
    used = np.unique(F) if len(F) else np.array([], dtype=int)
    comps = m.split(only_watertight=False)
    degen = int((np.array([len(set(f)) for f in F]) < 3).sum()) if len(F) else 0
    print(f"  {tag:<34} V={len(V):6d} F={len(F):6d} comps={len(comps):3d} "
          f"degen={degen:3d} orphans={len(V)-len(used):5d} watertight={m.is_watertight}")
    return dict(components=len(comps), degenerate=degen, watertight=m.is_watertight)


def cl_endpoints(cl):
    """Endpoint of each branch in a clipped-centerline array."""
    out = []
    arr = cl.item() if isinstance(cl, np.ndarray) and cl.dtype == object and cl.shape == () else cl
    if isinstance(arr, dict):
        for k, v in arr.items():
            v = np.asarray(v)
            if v.ndim == 2 and len(v):
                out.append((str(k), v[0], v[-1], len(v)))
    else:
        for i, v in enumerate(arr):
            v = np.asarray(v)
            if v.ndim == 2 and len(v):
                out.append((str(i), v[0], v[-1], len(v)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True)
    ap.add_argument("--dataset", default="AneuX")
    ap.add_argument("--geometry-root",
                    default="/media/yaplab2/wd8tb/wenhao/angioflow/data/geometry")
    ap.add_argument("--out", default=str(ROOT / "runtime" / "debug_mesh_fix"))
    ap.add_argument("--extrude-length", type=float, default=0.25)
    args = ap.parse_args()

    src_case = Path(args.geometry_root) / args.dataset / args.case
    work = Path(args.out) / args.case
    work.mkdir(parents=True, exist_ok=True)
    sandbox = work / "geometry"
    if not sandbox.exists():
        print(f"copying case geometry -> {sandbox}  (original never written to)")
        shutil.copytree(src_case, sandbox)

    paths = resolve_case_paths(sandbox)
    mesh_name = paths["clipped_mesh"].name
    cl_name = paths["clipped_centerline"].name
    merged_name = paths["merged_centerline"].name
    print(f"\nmesh       : {mesh_name}   (manual={paths['used_manual_mesh']})")
    print(f"centerline : {cl_name}   <- clipped to the AUTOMATIC cut")
    print(f"merged     : {merged_name}\n")

    # ── BEFORE ────────────────────────────────────────────────────────────
    before_cl = np.load(sandbox / cl_name, allow_pickle=True)
    np.save(work / "centerline_before.npy", before_cl, allow_pickle=True)
    print("=== BEFORE: stale centerline ===")
    for name, p0, p1, n in cl_endpoints(before_cl):
        print(f"    branch {name}: {n:4d} pts   end={np.round(p1, 2)}")
    capped_before = _build_closed_case_mesh(paths, extrude_length=args.extrude_length)
    capped_before.export(work / "capped_before.obj")
    b = health(capped_before, "capped_before.obj")

    # ── REPAIR: re-clip the merged centerline to the MANUAL mesh's openings ─
    print("\n=== repairing: clip_centerline_to_mesh onto the manual mesh ===")
    repaired_name = "clipped_centerline_repaired.npy"
    clip_centerline_to_mesh(
        str(sandbox),
        r_clipped_mesh_filename=mesh_name,
        r_centerline_filename=merged_name,
        r_endpoints_filename="endpoints_manual.npy",
        w_clipped_cl_filename=repaired_name,
        opening_min_size=3,
    )
    after_cl = np.load(sandbox / repaired_name, allow_pickle=True)
    np.save(work / "centerline_after.npy", after_cl, allow_pickle=True)
    print("\n=== AFTER: repaired centerline ===")
    for name, p0, p1, n in cl_endpoints(after_cl):
        print(f"    branch {name}: {n:4d} pts   end={np.round(p1, 2)}")

    # ── AFTER: cap with the repaired centerline ───────────────────────────
    paths_fixed = dict(paths)
    paths_fixed["clipped_centerline"] = sandbox / repaired_name
    print()
    capped_after = _build_closed_case_mesh(paths_fixed, extrude_length=args.extrude_length)
    capped_after.export(work / "capped_after.obj")
    a = health(capped_after, "capped_after.obj")

    # repaired centerline PLUS the oblique-safe extrusion
    capped_obl = _build_closed_case_mesh(paths_fixed, extrude_length=args.extrude_length,
                                         ring_normal_extrusion=True,
                                         strip_unreferenced_input=True)
    capped_obl.export(work / "capped_ring_normal.obj")
    o = health(capped_obl, "capped_ring_normal.obj")

    shutil.copy(paths["clipped_mesh"], work / "source_mesh.ply")
    print("\n=== verdict ===")
    print(f"  {'':<12} {'stale CL':<12} {'repaired CL':<14} {'repaired+ringnormal+stripped'}")
    for k in ("components", "degenerate"):
        print(f"  {k:<12} {b[k]:<12} {a[k]:<14} {o[k]}")
    print(f"  {'watertight':<12} {str(b['watertight']):<12} {str(a['watertight']):<14} "
          f"{o['watertight']}")
    print(f"\nfiles -> {work}")


if __name__ == "__main__":
    main()
