"""Measure the mesh resolution the downstream models were trained on.

The regularizer remeshes to the edge length of the AneuX reference set (~0.132 mm), which
is what makes roughness comparable against that reference. Downstream consumers (the wall
GNNs) were trained on a different tessellation, so the final output needs one more remesh
to their resolution. This reads that resolution off the training data itself rather than
guessing it.

Each case in the dataset root carries wall_data.pt holding, under "wall", a `pos` (N,3)
vertex tensor and a `faces` (F,3) index tensor. Edge lengths come straight from those.

Note on units: these tensors are in METRES (a complex spans ~0.03 = 30 mm), whereas the
whole regularizer pipeline works in millimetres. The reported target is therefore given in
both, and the millimetre figure is the one to feed back into remeshing.

    python -m mesh_regularizer.measure_downstream_edge_length --n_cases 100
"""

import os
import csv
import json
import argparse
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch

from .config import DOWNSTREAM_ROOT as DATASET_ROOT, DOWNSTREAM_WALL_FILE as WALL_FILE
CURRENT_TARGET_MM = 0.1319          # what the regularizer currently remeshes to


def edge_lengths(pos, faces):
    """All triangle edge lengths, one entry per (face, side)."""
    v, f = np.asarray(pos, float), np.asarray(faces, np.int64)
    return np.concatenate([np.linalg.norm(v[f[:, 0]] - v[f[:, 1]], axis=1),
                           np.linalg.norm(v[f[:, 1]] - v[f[:, 2]], axis=1),
                           np.linalg.norm(v[f[:, 2]] - v[f[:, 0]], axis=1)])


def measure_case(path, key="pos"):
    # mmap avoids pulling the wall-shear time series (tens of MB per case) into memory
    d = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    wall = d["wall"]
    v = wall[key].numpy()
    f = wall["faces"].numpy()
    e = edge_lengths(v, f)
    ext = v.max(0) - v.min(0)
    return {
        "n_vertices": int(len(v)), "n_faces": int(len(f)),
        "edge_mean": float(e.mean()), "edge_p05": float(np.percentile(e, 5)),
        "edge_p50": float(np.percentile(e, 50)), "edge_p95": float(np.percentile(e, 95)),
        "edge_cv": float(e.std() / max(e.mean(), 1e-30)),
        "bbox_diag": float(np.linalg.norm(ext)),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=DATASET_ROOT)
    p.add_argument("--n_cases", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--key", default="pos", choices=["pos", "pos_frame"])
    p.add_argument("--out", default=None, help="optional csv of per-case rows")
    args = p.parse_args()

    cases = sorted(c for c in os.listdir(args.root)
                   if os.path.exists(os.path.join(args.root, c, WALL_FILE)))
    rng = np.random.default_rng(args.seed)
    if args.n_cases and args.n_cases < len(cases):
        cases = [cases[i] for i in sorted(rng.choice(len(cases), args.n_cases, replace=False))]
    print("measuring %d of the available cases under %s" % (len(cases), args.root))

    rows, failures = [], []
    for i, c in enumerate(cases):
        try:
            r = measure_case(os.path.join(args.root, c, WALL_FILE), args.key)
            r["case"] = c
            rows.append(r)
        except Exception as exc:
            failures.append({"case": c, "error": repr(exc)})
        if (i + 1) % 25 == 0:
            print("  %d/%d" % (i + 1, len(cases)), flush=True)
    if not rows:
        raise SystemExit("no cases could be read")

    def col(k):
        return np.array([r[k] for r in rows], float)

    # Each case contributes equally: the target is the typical mesh's resolution, not a
    # pooled average that large meshes would dominate.
    per_case_mean = col("edge_mean")
    target_m = float(per_case_mean.mean())
    target_mm = target_m * 1000.0

    print("\n%d cases measured, %d failed" % (len(rows), len(failures)))
    print("\n%-16s %12s %12s %12s" % ("quantity", "p05", "p50", "p95"))
    for k, scale, unit in [("edge_mean", 1e3, "mm"), ("edge_p50", 1e3, "mm"),
                           ("edge_cv", 1.0, "-"), ("bbox_diag", 1e3, "mm"),
                           ("n_vertices", 1.0, "-")]:
        v = col(k) * scale
        print("%-16s %12.4f %12.4f %12.4f   %s"
              % (k, np.percentile(v, 5), np.percentile(v, 50), np.percentile(v, 95), unit))

    print("\nFINAL REMESH TARGET")
    print("  mean of per-case mean edge : %.6f m  =  %.4f mm" % (target_m, target_mm))
    print("  spread across cases        : %.4f mm (1 SD)" % (per_case_mean.std() * 1000))
    print("  current regularizer target : %.4f mm" % CURRENT_TARGET_MM)
    print("  ratio downstream/current   : %.3f x" % (target_mm / CURRENT_TARGET_MM))

    summary = {"root": args.root, "key": args.key, "n_cases": len(rows),
               "n_failed": len(failures),
               "target_edge_m": target_m, "target_edge_mm": target_mm,
               "sd_mm": float(per_case_mean.std() * 1000),
               "current_regularizer_target_mm": CURRENT_TARGET_MM,
               "failures": failures}
    out_json = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "downstream_edge_length.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print("\nsummary -> %s" % out_json)

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["case"] + [k for k in rows[0] if k != "case"])
            w.writeheader(); w.writerows(rows)
        print("per-case  -> %s" % args.out)


if __name__ == "__main__":
    main()
