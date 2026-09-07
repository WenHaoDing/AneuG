#!/usr/bin/env python3
"""Strip unreferenced vertices from a case's source clip, then re-run the
extrude -> planarize -> fan-cap closure and compare against the original.

WHY: a hand-cut ("_manual") clip typically deletes FACES and leaves their
vertices behind -- p171's manual clip carries 9538 vertices no face
references (F/V = 1.64, where a closed triangulated surface sits near 2.0).
Those orphans are invisible in a component count but are real points in the
file, and the downstream fusion/planarize step appears to choke on them,
emitting isolated 2-face islands and degenerate caps.

Writes both capped results into the case's output dir for visual comparison.

  python scripts/fit/recap_clean_mesh.py --case p171_..._cut2 --dataset AneuX
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ghd.fitting.alignment import resolve_case_paths, _build_closed_case_mesh  # noqa: E402

GEOMETRY_ROOT = "/media/yaplab2/wd8tb/wenhao/angioflow/data/geometry"


def mesh_health(m, tag):
    V, F = np.asarray(m.vertices), np.asarray(m.faces)
    used = np.unique(F) if len(F) else np.array([], dtype=int)
    comps = m.split(only_watertight=False)
    sizes = sorted((len(c.faces) for c in comps), reverse=True)
    degen = int((np.array([len(set(f)) for f in F]) < 3).sum()) if len(F) else 0
    print(f"  {tag}")
    print(f"      V={len(V):7d}  F={len(F):7d}  F/V={len(F)/max(len(V),1):.2f}")
    print(f"      unreferenced verts : {len(V) - len(used)}")
    print(f"      components         : {len(comps)}   face-sizes={sizes[:8]}")
    print(f"      degenerate faces   : {degen}")
    print(f"      watertight         : {m.is_watertight}")
    return dict(V=len(V), F=len(F), orphans=len(V) - len(used),
                components=len(comps), degenerate=degen, watertight=m.is_watertight)


def strip_unreferenced(mesh):
    """Drop vertices no face references, remapping face indices."""
    V, F = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    used = np.unique(F)
    remap = np.full(len(V), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return trimesh.Trimesh(vertices=V[used], faces=remap[F], process=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True)
    ap.add_argument("--dataset", default="AneuX")
    ap.add_argument("--config", default="default",
                    help="Which runtime/<config>/<dataset>/<case>/ dir to write into.")
    ap.add_argument("--extrude-length", type=float, default=0.25)
    args = ap.parse_args()

    geo = Path(GEOMETRY_ROOT) / args.dataset / args.case
    out = ROOT / "runtime" / args.config / args.dataset / args.case
    out.mkdir(parents=True, exist_ok=True)
    paths = resolve_case_paths(geo)
    src_path = paths["clipped_mesh"]
    print(f"source: {src_path.name}  (manual={paths['used_manual_mesh']})\n")

    src = trimesh.load(src_path, process=False)
    before = mesh_health(src, "SOURCE (as-is)")

    cleaned = strip_unreferenced(src)
    print()
    after = mesh_health(cleaned, "SOURCE (unreferenced stripped)")
    cleaned_path = out / "source_cleaned.ply"
    cleaned.export(cleaned_path)

    # cap BOTH, through the identical code path, changing only the input mesh
    print("\n=== re-capping ===")
    results = {}
    for tag, mesh_file in (("orig", src_path), ("clean", cleaned_path)):
        p = dict(paths)
        p["clipped_mesh"] = Path(mesh_file)
        try:
            capped = _build_closed_case_mesh(p, extrude_length=args.extrude_length)
        except Exception as e:
            print(f"  {tag}: FAILED -- {type(e).__name__}: {e}")
            continue
        dst = out / f"recap_{tag}.obj"
        capped.export(dst)
        print()
        results[tag] = mesh_health(capped, f"CAPPED from {tag} -> {dst.name}")

    if len(results) == 2:
        print("\n=== verdict ===")
        for k in ("components", "degenerate", "orphans"):
            o, c = results["orig"][k], results["clean"][k]
            arrow = "FIXED" if c < o else ("same" if c == o else "WORSE")
            print(f"  {k:<12} orig={o:<6} clean={c:<6} {arrow}")
        print(f"  watertight   orig={results['orig']['watertight']}  "
              f"clean={results['clean']['watertight']}")


if __name__ == "__main__":
    main()
