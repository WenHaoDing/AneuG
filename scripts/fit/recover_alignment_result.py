"""Write the missing alignment_result.npy for assembled cases fitted with arap_init.

WHY THEY ARE MISSING. ghd/registration/fit_with_arap_init.py runs its own Stage-1
alignment but discards the result -- `transform, _ = fit_alignment(...)` -- and
save_stage1_checkpoints() never writes alignment_result.npy (alignment.py and run_case.py
save it themselves, before calling that function). So every arap_init case lacks it:
106 of the 523 assembled cases, and no other config.

WHAT IS RECOVERABLE. R and t survive in landmarks.npz (R_frame, t_vec are the Stage-1
transform itself). The scale s does not. It is recovered EXACTLY from the Stage-1 output:

    final_aligned.obj = s * (closed_mesh @ R.T) + t        (vertex order preserved)

closed_mesh is rebuilt with the same load_case_data() the fitter used, and s is solved by
least squares. Each case is verified before anything is written:

  * the rebuilt mesh has the same vertex count as final_aligned.obj,
  * with the recovered s, the transform reproduces final_aligned.obj to --max_residual_mm
    (float32 rounding is ~1e-5 mm; tested worst 2.7e-5),
  * s agrees with dataset/preprocess_assembled.py's own estimate from the aneurysm-
    centroid correspondence. This matters: that function PREFERS alignment_result.npy
    once one exists, so a disagreeing value would silently change AneuG_processed the
    next time it is rebuilt.

CASES WHOSE GEOMETRY CHANGED AFTER FITTING (3 of 106: a later _fallback_manual clip) no
longer rebuild to the fitted vertices, so exact matching is impossible. For those, s is the
centroid estimate, verified on the vertices that did not change: 88-89% of them land on
final_aligned.obj to ~1e-6 mm, and moving s by 1% drops that to 0%. The method used is
recorded in each file (`recovery_method`).

Validated on cases that DO have the file (recovered s == stored s to 6 decimals) and end to
end on three CFD cases, which then land 0.06 mm from their CFD wall.

WHAT IS WRITTEN. Only R, s, t -- the keys downstream code reads -- plus provenance
(`recovered=True`, the method, residual and cross-check). Optimizer extras the original
files carry (R_init, s_init, t_init, history, final_loss, use_fallback) are not recoverable
and are not faked. An existing alignment_result.npy is never overwritten.

    python scripts/fit/recover_alignment_result.py --dry_run    # verify all, write nothing
    python scripts/fit/recover_alignment_result.py              # write
"""
import argparse
import contextlib
import csv
import io
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ASSEMBLED = ROOT / "runtime_dataset" / "assembled"
LOG_DIR = ROOT / "runtime_debug" / "recover_alignment"
DOME_SOURCE = {"AneuX": "mesh", "ImperialNHS": "nrrd"}   # closed_mesh does not depend on it


def recover_case(job):
    ds, case, max_res, max_diff, min_overlap, write = job
    import trimesh
    from ghd.fitting.alignment import load_case_data, resolve_case_paths
    from ghd.registration.fit_with_arap_init import DOME_LOADERS
    from dataset.preprocess_assembled import stage1_transform

    P = ASSEMBLED / ds / case
    row = {"dataset": ds, "case": case}
    t0 = time.time()
    try:
        out = P / "alignment_result.npy"
        if out.exists():
            return {**row, "status": "exists"}
        gdir = json.load(open(P / "metrics.json"))["case_dir"]
        lm = np.load(P / "landmarks.npz", allow_pickle=True)
        R = np.asarray(lm["R_frame"], np.float64)
        t = np.asarray(lm["t_vec"], np.float64)
        Fa = np.asarray(trimesh.load(P / "final_aligned.obj", process=False).vertices, np.float64)
        with contextlib.redirect_stdout(io.StringIO()):
            c = load_case_data(gdir, dome_points_loader=DOME_LOADERS[DOME_SOURCE[ds]])
        X = np.asarray(c["closed_mesh"].vertices, np.float64)
        Y = X @ R.T

        # the estimate downstream will otherwise compute on the fly -- must agree
        endpoints = Path(resolve_case_paths(Path(gdir))["endpoints"])
        _, s_centroid, _ = stage1_transform(P, Path(gdir), endpoints)

        if len(X) == len(Fa):
            # EXACT: same vertices in the same order, so s is a one-line least squares
            method = "mesh_least_squares"
            s = float(((Fa - t) * Y).sum() / (Y * Y).sum())
            residual = float(np.linalg.norm(s * Y + t - Fa, axis=1).max())
            overlap = 1.0
        else:
            # The case's geometry was edited after fitting (e.g. a later _fallback_manual
            # clip), so the rebuilt mesh no longer has the fitted mesh's vertices. Use the
            # centroid estimate and VERIFY it on the part of the mesh that did not change:
            # those vertices must land exactly on final_aligned.obj. A 1% error in s moves
            # them by ~0.1 mm, so the check is sharp.
            from scipy.spatial import cKDTree
            method = "centroid_verified_on_unchanged_vertices"
            s = float(s_centroid)
            d, _ = cKDTree(Fa).query(s * Y + t)
            overlap = float((d < 1e-3).mean())
            residual = float(np.median(d))
            row.update(note="geometry changed since fit: rebuilt %d verts vs fitted %d"
                            % (len(X), len(Fa)))
            if overlap < min_overlap:
                return {**row, "status": "unverifiable", "s": s, "s_centroid": s_centroid,
                        "abs_diff": 0.0, "residual_mm": residual, "overlap": overlap}
        diff = abs(s - s_centroid)

        row.update(s=s, s_centroid=s_centroid, abs_diff=diff, residual_mm=residual,
                   overlap=overlap, method=method)
        if residual > max_res:
            return {**row, "status": "residual_too_large"}
        if diff > max_diff:
            return {**row, "status": "disagrees_with_preprocess"}

        if write:
            rec = {"R": R.astype(np.float32), "s": s, "t": t.astype(np.float32),
                   "case_dir": gdir,
                   "recovered": True,
                   "recovery_method": method,
                   "recovery_residual_mm": residual,
                   "recovery_overlap": overlap,
                   "recovery_s_centroid_check": s_centroid,
                   "recovered_by": "scripts/fit/recover_alignment_result.py",
                   "recovered_at": time.strftime("%Y-%m-%d %H:%M:%S")}
            tmp = P / "alignment_result.recovering.npy"
            np.save(tmp, rec, allow_pickle=True)
            os.replace(tmp, out)                  # atomic: never a half-written file
        return {**row, "status": "written" if write else "verified",
                "seconds": round(time.time() - t0, 1)}
    except Exception as exc:
        return {**row, "status": "failed", "error": repr(exc)}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry_run", action="store_true", help="verify every case, write nothing")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max_residual_mm", type=float, default=1e-3)
    p.add_argument("--max_scale_diff", type=float, default=1e-3,
                   help="allowed |s - preprocess_assembled's centroid estimate|")
    p.add_argument("--min_overlap", type=float, default=0.8,
                   help="for cases whose geometry changed after fitting: share of rebuilt "
                        "vertices that must land on final_aligned.obj within 1e-3 mm")
    p.add_argument("--cases", nargs="*", default=None)
    a = p.parse_args()

    jobs = []
    for ds in ("AneuX", "ImperialNHS"):
        for case in sorted(os.listdir(ASSEMBLED / ds)):
            if a.cases and case not in a.cases:
                continue
            if not (ASSEMBLED / ds / case / "alignment_result.npy").exists():
                jobs.append((ds, case, a.max_residual_mm, a.max_scale_diff, a.min_overlap,
                             not a.dry_run))
    print("%d case(s) missing alignment_result.npy%s"
          % (len(jobs), " -- DRY RUN, nothing will be written" if a.dry_run else ""), flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i, f in enumerate(as_completed([pool.submit(recover_case, j) for j in jobs]), 1):
            r = f.result()
            rows.append(r)
            if "s" in r:
                print("[%d/%d] %-28s %-26s s %.6f  centroid %.6f  |diff| %.1e  residual %.1e"
                      % (i, len(jobs), r["case"], r["status"], r["s"], r["s_centroid"],
                         r["abs_diff"], r["residual_mm"]), flush=True)
            else:
                print("[%d/%d] %-28s %s %s" % (i, len(jobs), r["case"], r["status"],
                                               r.get("error", "")[:80]), flush=True)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = LOG_DIR / ("recovery_dry_run.csv" if a.dry_run else "recovery_log.csv")
    with open(log, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "case", "status", "method", "s",
                                           "s_centroid", "abs_diff", "residual_mm", "overlap",
                                           "seconds", "note", "error"],
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["status"], r["dataset"], r["case"])))
    import collections
    print("\n%s" % dict(collections.Counter(r["status"] for r in rows)))
    ok = [r for r in rows if "s" in r]
    if ok:
        print("worst residual %.2e mm | worst |s - centroid estimate| %.2e"
              % (max(r["residual_mm"] for r in ok), max(r["abs_diff"] for r in ok)))
    print("log: %s" % log)


if __name__ == "__main__":
    main()
