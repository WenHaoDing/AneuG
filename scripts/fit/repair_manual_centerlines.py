#!/usr/bin/env python3
"""Re-clip the centerline onto every hand-cut ("_manual") mesh in a dataset.

A "_manual" clip is cut by hand, so its boundary openings sit somewhere else
than the automatic cut's -- 3.5-5.6 mm away on p171. The stock clipped
centerline was clipped to the AUTOMATIC openings, so pairing it with the manual
mesh leaves the extrusion starting from points that aren't on the mesh's
openings. On p171 that produced 19 components, 2 degenerate faces, 55 zero-area
faces and Euler -132; re-clipping fixed it to 1 component, Euler 2, watertight.

Writes clipped_centerline{_fallback}_manual.npy NEXT TO the existing files --
nothing is overwritten. resolve_case_paths() prefers it automatically whenever
the manual mesh is in use.

  python scripts/fit/repair_manual_centerlines.py --dataset AneuX
  python scripts/fit/repair_manual_centerlines.py --dataset AneuX --dry-run
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "AneuSeg"))

from ghd.fitting.alignment import resolve_case_paths  # noqa: E402
from IAgents.tools.vessel_clipping import clip_centerline_to_mesh  # noqa: E402

GEOMETRY_ROOT = "/media/yaplab2/wd8tb/wenhao/angioflow/data/geometry"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="AneuX")
    ap.add_argument("--geometry-root", default=GEOMETRY_ROOT)
    ap.add_argument("--cases", nargs="+", default=None, help="Restrict to these case names.")
    ap.add_argument("--opening-min-size", type=int, default=3)
    ap.add_argument("--force", action="store_true", help="Redo cases already repaired.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.geometry_root) / args.dataset
    cases = sorted(c for c in root.iterdir()
                   if c.is_dir() and (c / "endpoints_manual.npy").exists()
                   and (args.cases is None or c.name in args.cases))

    todo = []
    for c in cases:
        paths = resolve_case_paths(c)
        if not paths["used_manual_mesh"]:
            continue
        suffix = "_fallback" if paths["use_fallback"] else ""
        out = c / f"clipped_centerline{suffix}_manual.npy"
        merged = c / f"merged_centerline{suffix}.npy"
        if not merged.exists():
            merged = c / "merged_centerline.npy"
        todo.append((c, paths["clipped_mesh"].name, merged.name, out))

    print(f"{len(todo)} manual-clip case(s) in {args.dataset}")
    done = skipped = failed = 0
    for c, mesh_name, merged_name, out in todo:
        if out.exists() and not args.force:
            skipped += 1
            continue
        if args.dry_run:
            print(f"  would repair {c.name}: {mesh_name} + {merged_name} -> {out.name}")
            continue
        try:
            clip_centerline_to_mesh(
                str(c),
                r_clipped_mesh_filename=mesh_name,
                r_centerline_filename=merged_name,
                r_endpoints_filename="endpoints_manual.npy",
                w_clipped_cl_filename=out.name,
                opening_min_size=args.opening_min_size,
            )
            done += 1
            print(f"  OK   {c.name} -> {out.name}", flush=True)
        except Exception as e:  # noqa: BLE001 -- one bad case must not stop the rest
            failed += 1
            print(f"  FAIL {c.name}: {type(e).__name__}: {e}", flush=True)

    if not args.dry_run:
        print(f"\nrepaired={done}  already-present={skipped}  failed={failed}")


if __name__ == "__main__":
    main()
