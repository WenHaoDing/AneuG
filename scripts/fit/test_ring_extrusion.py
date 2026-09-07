#!/usr/bin/env python3
"""Compare three ways of closing a case's clip, for eyeballing.

  A  extrude along the CENTERLINE TANGENT   (the original path)
  B  extrude along the RING's own normal, oriented from the MESH  (no centerline)
  C  no extrusion at all -- cap the clip where it is (current manual-case path)
  D  extrude the ring's OWN vertices along the ring normal, shape preserved

B is the thing under test: the ring alone determines the cut plane, and the
mesh's own topology says which side is outward (the ring's non-ring neighbours
lie in the body), so the extrusion never needs the centerline tangent. That
matters for hand-cut clips, whose cross-section is oblique to the centerline --
tangent-built rings are tilted relative to the opening they stitch to.
"""
import argparse, sys
from pathlib import Path
import numpy as np, trimesh

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ghd.fitting.alignment import resolve_case_paths, _build_closed_case_mesh  # noqa: E402

GEO = "/media/yaplab2/wd8tb/wenhao/angioflow/data/geometry"


def health(m):
    V, F = np.asarray(m.vertices), np.asarray(m.faces)
    used = np.unique(F)
    tri = V[F]
    a = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2
    return dict(V=len(V), F=len(F), comps=len(m.split(only_watertight=False)),
                degen=int((np.array([len(set(f)) for f in F]) < 3).sum()),
                orphans=len(V) - len(used), zero_area=int((a < 1e-9).sum()),
                euler=m.euler_number, watertight=bool(m.is_watertight), vol=float(m.volume))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--dataset", default="AneuX")
    ap.add_argument("--out", default=str(ROOT / "runtime" / "debug_extrusion"))
    args = ap.parse_args()

    variants = [
        ("A_tangent",     dict(no_extrusion=False, ring_normal_extrusion=False)),
        ("B_ringnormal",  dict(no_extrusion=False, ring_normal_extrusion=True)),
        ("C_noextrusion", dict(no_extrusion=True,  ring_normal_extrusion=False)),
        ("D_ringprism",    dict(ring_prism_extrusion=True)),
    ]
    for case in args.cases:
        d = Path(args.out) / case
        d.mkdir(parents=True, exist_ok=True)
        paths = resolve_case_paths(Path(GEO) / args.dataset / case)
        man = paths["used_manual_mesh"]
        print(f"\n=== {case}  (manual_mesh={man}) ===")
        print(f"  {'variant':<16} {'comps':>5} {'degen':>6} {'zeroA':>6} {'euler':>7} "
              f"{'watertight':>11} {'volume':>10}")
        trimesh.load(paths["clipped_mesh"], process=False).export(d / "source.ply")
        for tag, kw in variants:
            try:
                m = _build_closed_case_mesh(paths, extrude_length=0.25,
                                            strip_unreferenced_input=True, **kw)
            except Exception as e:
                print(f"  {tag:<16} FAILED: {type(e).__name__}: {str(e)[:60]}")
                continue
            m.export(d / f"{tag}.obj")
            h = health(m)
            print(f"  {tag:<16} {h['comps']:5d} {h['degen']:6d} {h['zero_area']:6d} "
                  f"{h['euler']:7d} {str(h['watertight']):>11} {h['vol']:10.1f}")
        print(f"  -> {d}")


if __name__ == "__main__":
    main()
