"""
Mesh-regularize synthetic aneurysm bundles exported by generate_synthetic.py.

Real uploaded cases all pass through mesh_regularizer.MeshRegularizer before CFD, which
normalizes surface roughness to match a physiological reference. Synthetic training
shapes should go through the same normalization so a downstream model trains on the same
roughness distribution it will see at inference, rather than the generator's own
tessellation artifacts (see mesh_regularizer/README.md for what the regularizer does and
why it's calibrated the way it is).

Targets each {out}/synth_{NNNNN}_aneurysm1/merge_mesh_world.obj written by
generate_synthetic.py and writes, into that same folder:

    merge_mesh_world_regularized.obj   regularized mesh, remeshed to the downstream
                                        training resolution (MeshRegularizer.export_edge)
    regularization_report.json         full per-case audit trail (status, rounds,
                                        z-scores before/after, deviation, etc.)

plus a batch summary at {out}/regularization_summary.json.

mesh_regularizer.mesh_regularizer already ships a CLI that walks an
`input_root/<case>/<filename>` layout one case at a time (`python -m
mesh_regularizer.mesh_regularizer run`) -- generate_synthetic.py's output already has
exactly that shape, so this script adds the one thing that CLI doesn't have:
multiprocessing. Each case is CPU/BLAS-bound (quadric fits + sparse solves) and can take
tens of seconds, so a spawn-context Pool with each worker pinned to one BLAS thread turns
an hour of serial cases into a few minutes on a multi-core machine -- the same pattern
mesh_regularizer.ReferenceModel.scan uses for scanning reference cases.

Usage:
    conda activate new
    python scripts/generate/regularize_synthetic.py --out v2/synthetic_v1 --n-jobs 8
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path = [p for p in sys.path if Path(p or ".").resolve() != ROOT]
sys.path.insert(0, str(ROOT))

from mesh_regularizer import MeshRegularizer, model_path

INPUT_FILENAME = "merge_mesh_world.obj"
OUTPUT_FILENAME = "merge_mesh_world_regularized.obj"
REPORT_FILENAME = "regularization_report.json"

# Reference model to use by default (targets 0.3/0.5/0.8/1.0/1.2 mm, guard 1.6 mm),
# per mesh_regularizer/reference_models/reference_roughness__t0.3-0.5-0.8-1-1.2__g1.6.npz
DEFAULT_TARGET_SCALES = (0.3, 0.5, 0.8, 1.0, 1.2)
DEFAULT_GUARD_SCALE = 1.6
DEFAULT_CACHE_PATH = model_path(DEFAULT_TARGET_SCALES, DEFAULT_GUARD_SCALE)


def _regularize_one(job):
    """Runs in a worker process. Module-level so it can be sent to a Pool."""
    sample_dir, kwargs = job
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    sample_dir = Path(sample_dir)
    src = sample_dir / INPUT_FILENAME
    reg = MeshRegularizer(verbose=False, **kwargs)
    try:
        mesh, report = reg.forward(str(src), title=sample_dir.name)
    except Exception as exc:
        return {"case": sample_dir.name, "status": "failed", "error": repr(exc)}
    mesh.export(sample_dir / OUTPUT_FILENAME)
    (sample_dir / REPORT_FILENAME).write_text(json.dumps(report, indent=2))
    return {
        "case": sample_dir.name, "status": report["status"], "rounds": report["rounds"],
        "surface_deviation_max_mm": report["surface_deviation_vs_original_max_mm"],
    }


def regularize(out_dir, n_jobs=None, overwrite=False, limit=None, **reg_kwargs):
    """Regularize every synth_*_aneurysm1/merge_mesh_world.obj under out_dir."""
    import multiprocessing as mp

    out_dir = Path(out_dir)
    cases = sorted(d for d in out_dir.iterdir() if d.is_dir() and (d / INPUT_FILENAME).exists())
    if not overwrite:
        cases = [d for d in cases if not (d / OUTPUT_FILENAME).exists()]
    if limit:
        cases = cases[:limit]
    if not cases:
        print(f"nothing to regularize under {out_dir} "
              f"(pass --overwrite to reprocess existing output)")
        return []

    n_jobs = n_jobs or os.cpu_count()
    print(f"regularizing {len(cases)} case(s) with {n_jobs} worker(s): {reg_kwargs}")

    jobs = [(d, reg_kwargs) for d in cases]
    outcomes = []
    with mp.get_context("spawn").Pool(n_jobs) as pool:
        for i, result in enumerate(pool.imap_unordered(_regularize_one, jobs), 1):
            print(f"[{i}/{len(cases)}] {result['case']}: {result['status']}")
            outcomes.append(result)

    summary_path = out_dir / "regularization_summary.json"
    summary_path.write_text(json.dumps(outcomes, indent=2))
    tally = {}
    for o in outcomes:
        tally[o["status"]] = tally.get(o["status"], 0) + 1
    print(f"\nregularized {len(outcomes)} case(s) -> {tally}")
    print(f"batch summary -> {summary_path}")
    return outcomes


def _parse_args():
    p = argparse.ArgumentParser(
        description="Mesh-regularize synthetic bundles from generate_synthetic.py.")
    p.add_argument("--out", type=str, required=True,
                   help="generate_synthetic.py --out directory (contains synth_*_aneurysm1/)")
    p.add_argument("--n-jobs", type=int, default=None,
                   help="worker processes (default: os.cpu_count())")
    p.add_argument("--overwrite", action="store_true",
                   help="reprocess cases that already have a regularized output")
    p.add_argument("--limit", type=int, default=None, help="process at most this many cases")
    p.add_argument("--smoother", default="mcf", choices=["mcf", "taubin", "humphrey"])
    p.add_argument("--remesher", default="auto", choices=["auto", "pymeshlab", "vmtk", "none"])
    p.add_argument("--tolerance", type=float, default=1.0,
                   help="stop at this signed z above the reference; lower = more "
                        "aggressive (0 = reference mean, negative = smoother than typical)")
    p.add_argument("--n-seeds", default="auto",
                   help="'auto' to fit --time-budget, 'all' for every vertex, or an int")
    p.add_argument("--time-budget", type=float, default=60.0,
                   help="wall-clock seconds per case; also sizes --n-seeds auto. "
                        "0 (or negative) disables the limit, running each case to full fidelity")
    p.add_argument("--export-edge", type=float, default=None,
                   help="edge length (mm) for the final remesh of the regularized mesh; "
                        "default is MeshRegularizer's own downstream-training resolution")
    p.add_argument("--cache-path", type=str, default=DEFAULT_CACHE_PATH,
                   help="reference model .npz to regularize against (default: targets "
                        f"{DEFAULT_TARGET_SCALES} mm, guard {DEFAULT_GUARD_SCALE} mm)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    n_seeds = (None if args.n_seeds == "all"
               else args.n_seeds if args.n_seeds == "auto"
               else int(args.n_seeds))
    kwargs = dict(cache_path=args.cache_path, smoother=args.smoother, remesher=args.remesher,
                  tolerance=args.tolerance, n_seeds=n_seeds,
                  time_budget_s=(None if args.time_budget <= 0 else args.time_budget))
    if args.export_edge is not None:
        kwargs["export_edge"] = args.export_edge
    regularize(args.out, n_jobs=args.n_jobs, overwrite=args.overwrite, limit=args.limit,
               **kwargs)
