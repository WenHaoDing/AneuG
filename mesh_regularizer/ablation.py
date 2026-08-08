"""Ablation harness for the mesh regularizer.

Sweeps cached reference models (scale configurations) against smoothing tolerances over a
fixed set of target shapes, and writes one self-describing folder per (config, tolerance)
run so results from different settings never pile up on top of each other.

Layout produced under --out_root:

    00_inputs/                      the source meshes, once, prefixed smooth__ / noisy__
    cfg-<config>__tol<T>/           one folder per experimental condition
        <case>.obj                  regularized mesh
        <case>_convergence.png      convergence + wall-clock trace
        manifest.csv                one row per case
        run_config.txt              every setting used, for reproducibility
    all_runs.csv                    every run in one table, written incrementally

Cases are chosen by ranking every candidate shape on fine-scale roughness and taking the
smoothest and noisiest halves, so the sweep covers both ends of the input quality range
rather than an arbitrary slice.

    python -m mesh_regularizer.ablation rank        # rank candidates, pick the case set
    python -m mesh_regularizer.ablation run         # the sweep itself
    python -m mesh_regularizer.ablation rebuild     # regenerate manifests from all_runs.csv
    python -m mesh_regularizer.ablation summarize   # cross-tabulated results
"""

import os
import sys
import csv
import json
import glob
import shutil
import hashlib
import argparse
import warnings
import multiprocessing as mp

warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

from .config import INPUT_ROOT, INPUT_FILENAME, MODEL_DIR as DEFAULT_MODEL_DIR, OUTPUT_ROOT
from .mesh_regularizer import (
    MeshRegularizer, ReferenceModel, load_mesh, remesh, ring_sets,
    multiscale_roughness, weighted_quantile, vertex_areas)

OUT_ROOT = OUTPUT_ROOT
RANK_SCALES = (0.3, 0.5)          # ranking only needs the noise-sensitive fine scales

FIELDS = ["case", "group", "config", "tolerance", "smoother", "sigma_step",
          "status", "rounds", "dev_max_mm",
          "dev_mean_mm", "guard", "z_before_0_3", "z_0_3", "z_0_5", "z_0_8",
          "n_seeds", "n_vertices", "elapsed_s", "md5", "error"]

_model_cache = {}


def run_dir_for(out_root, cfg, tol, smoother="mcf", sigma_step=None):
    """mcf at the default step keeps the plain name; anything else is stamped, so a
    smoother or step-size sweep cannot silently overwrite a baseline run."""
    name = "cfg-%s__tol%+.1f" % (cfg, tol)
    if smoother != "mcf":
        name += "__sm-%s" % smoother
    if sigma_step is not None:
        name += "__sig%.2f" % sigma_step
    return os.path.join(out_root, name)


def config_of(model_file):
    return os.path.basename(model_file).replace("reference_roughness__", "").replace(".npz", "")


# ======================================================================================
# case selection
# ======================================================================================

def _rank_one(args):
    case, target_edge = args
    try:
        m = remesh(load_mesh(os.path.join(INPUT_ROOT, case, INPUT_FILENAME)),
                   target_edge, "auto")
        n = len(m.vertices)
        idx = (np.arange(n) if n <= 3000
               else np.random.default_rng(0).choice(n, 3000, replace=False))
        vals = multiscale_roughness(m, RANK_SCALES, idx, ring_sets(m, RANK_SCALES))
        w = vertex_areas(m)[idx]
        return case, {float(s): weighted_quantile(vals[float(s)], w) for s in RANK_SCALES}, None
    except Exception as exc:
        return case, None, repr(exc)


def cmd_rank(args):
    """Rank every candidate shape by fine-scale roughness and store the case selection."""
    model = ReferenceModel.load(args.rank_model)
    cases = [c for c in sorted(os.listdir(INPUT_ROOT))
             if os.path.exists(os.path.join(INPUT_ROOT, c, INPUT_FILENAME))]
    print("ranking %d cases with %d workers" % (len(cases), args.workers), flush=True)
    jobs = [(c, model.target_edge) for c in cases]
    rows, failures = [], []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for i, (case, curves, err) in enumerate(pool.imap_unordered(_rank_one, jobs, 1)):
            if err:
                failures.append({"case": case, "error": err})
                continue
            rows.append({"case": case,
                         "z_0_3": model.signed_excess(0.3, curves[0.3]),
                         "z_0_5": model.signed_excess(0.5, curves[0.5])})
            if (i + 1) % 25 == 0:
                print("  %d/%d" % (i + 1, len(cases)), flush=True)
    rows.sort(key=lambda r: r["z_0_3"])
    half = args.n_cases // 2
    out = {"smoothest": [r["case"] for r in rows[:half]],
           "noisiest": [r["case"] for r in rows[-half:]],
           "all_ranked": rows, "failures": failures}
    with open(args.ranking, "w") as f:
        json.dump(out, f, indent=2)
    print("\nz(0.3) spans %.2f .. %.2f over %d cases"
          % (rows[0]["z_0_3"], rows[-1]["z_0_3"], len(rows)))
    print("saved -> %s" % args.ranking)


def load_cases(args):
    sel = json.load(open(args.ranking))
    smooth = sel.get("smoothest", sel.get("smoothest_15", []))
    noisy = sel.get("noisiest", sel.get("noisiest_15", []))
    half = args.n_cases // 2
    return ([(c, "smooth") for c in smooth[:half]] +
            [(c, "noisy") for c in noisy[-half:]])


# ======================================================================================
# the sweep
# ======================================================================================

def _job(spec):
    (model_file, tol, case, group, out_root, seeds, budget,
     smoother, sigma_step) = spec
    cfg = config_of(model_file)
    if model_file not in _model_cache:
        _model_cache[model_file] = ReferenceModel.load(model_file)
    model = _model_cache[model_file]
    rd = run_dir_for(out_root, cfg, tol, smoother, sigma_step)
    try:
        kw = {} if sigma_step is None else {"sigma_step": sigma_step}
        reg = MeshRegularizer(model=model, tolerance=tol,
                              overshoot_limit=min(-1.0, tol - 1.0),
                              smoother=smoother,
                              n_seeds=seeds, time_budget_s=budget, verbose=False, **kw)
        src = os.path.join(INPUT_ROOT, case, INPUT_FILENAME)
        out, r = reg.forward(src, plot_path=os.path.join(rd, case + "_convergence.png"),
                             title="%s [%s] | cfg %s | tol %+.1f | %s | sig %s"
                                   % (case, group, cfg, tol, smoother,
                                      sigma_step or "default"))
        mesh_path = os.path.join(rd, case + ".obj")
        out.export(mesh_path)
        z = r["signed_excess_after"]
        return dict(
            case=case, group=group, config=cfg, tolerance=tol,
            smoother=smoother, sigma_step=reg.sigma_step, status=r["status"],
            rounds=r["rounds"],
            dev_max_mm=round(r["surface_deviation_vs_original_max_mm"], 4),
            dev_mean_mm=round(r["surface_deviation_vs_original_mean_mm"], 5),
            guard=round(r["below_band_after"][str(reg.guard_scale)], 3),
            z_before_0_3=round(r["signed_excess_before"]["0.3"], 2),
            z_0_3=round(z["0.3"], 2),
            z_0_5=round(z.get("0.5", float("nan")), 2),
            z_0_8=round(z.get("0.8", float("nan")), 2),
            n_seeds=r["n_seeds_used"], n_vertices=r["n_vertices_out"],
            elapsed_s=r["elapsed_s"],
            md5=hashlib.md5(open(mesh_path, "rb").read()).hexdigest()[:12])
    except Exception as exc:
        return dict(case=case, group=group, config=cfg, tolerance=tol,
                    smoother=smoother, sigma_step=sigma_step,
                    status="FAILED", error=repr(exc))


def write_run_config(path, model_file, model, tol, seeds, budget, n_cases,
                     smoother="mcf", sigma_step=None):
    kw = {} if sigma_step is None else {"sigma_step": sigma_step}
    reg = MeshRegularizer(model=model, tolerance=tol, smoother=smoother,
                          verbose=False, **kw)
    with open(path, "w") as f:
        f.write(
            "model          : %s\n"
            "config         : %s\n"
            "target scales  : %s mm\n"
            "guard scale    : %s mm\n"
            "tolerance      : %+.1f SD\n"
            "overshoot limit: %+.1f SD\n"
            "smoother       : %s\nsigma step     : %.3f mm\n"
            "compensate     : sigma %s mm, %d passes\n"
            "seeds          : %s\n"
            "time budget    : %s\n"
            "reference cases: %d\n"
            "target cases   : %d (half smoothest, half noisiest)\n"
            % (os.path.basename(model_file), config_of(model_file),
               reg.target_scales, reg.guard_scale, tol, min(-1.0, tol - 1.0),
               reg.smoother, reg.sigma_step, reg.compensate_sigma, reg.compensate_passes,
               "all vertices" if seeds is None else seeds,
               "none (runs to completion)" if not budget else "%.0f s/case" % budget,
               model.n_cases, n_cases))


def cmd_run(args):
    models = sorted(glob.glob(os.path.join(args.model_dir, "*.npz")))
    if args.configs:
        models = [m for m in models if any(c in os.path.basename(m) for c in args.configs)]
    if not models:
        raise SystemExit("no reference models matched under %s" % args.model_dir)
    cases = load_cases(args)
    seeds = None if args.seeds <= 0 else args.seeds
    budget = 0 if args.budget <= 0 else args.budget

    n_sig = len(args.sigma_steps or [None])
    print("%d configs x %d tolerances x %d smoothers x %d step sizes x %d cases = %d runs"
          % (len(models), len(args.tolerances), len(args.smoothers), n_sig, len(cases),
             len(models) * len(args.tolerances) * len(args.smoothers) * n_sig * len(cases)),
          flush=True)

    inputs = os.path.join(args.out_root, "00_inputs")
    os.makedirs(inputs, exist_ok=True)
    for c, g in cases:
        dst = os.path.join(inputs, "%s__%s.obj" % (g, c))
        if not os.path.exists(dst):
            shutil.copy2(os.path.join(INPUT_ROOT, c, INPUT_FILENAME), dst)

    sigmas = args.sigma_steps or [None]
    specs = []
    for mf in models:
        for tol in args.tolerances:
            for sm in args.smoothers:
                for sig in sigmas:
                    os.makedirs(run_dir_for(args.out_root, config_of(mf), tol, sm, sig),
                                exist_ok=True)
                    for c, g in cases:
                        specs.append((mf, tol, c, g, args.out_root, seeds, budget, sm, sig))

    live_path = os.path.join(args.out_root, "all_runs.csv")
    with open(live_path, "w", newline="") as live:
        writer = csv.DictWriter(live, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader(); live.flush()
        done = 0
        with mp.get_context("spawn").Pool(args.workers) as pool:
            for r in pool.imap_unordered(_job, specs, chunksize=1):
                writer.writerow(r); live.flush()
                done += 1
                if done % 10 == 0 or r["status"] == "FAILED":
                    print("  %d/%d  %s %s tol%+.1f -> %s (%ss)"
                          % (done, len(specs), r["case"][:18], r["config"][:24],
                             r["tolerance"], r["status"], r.get("elapsed_s")), flush=True)

    for mf in models:
        model = ReferenceModel.load(mf)
        for tol in args.tolerances:
            for sm in args.smoothers:
                for sig in sigmas:
                    write_run_config(
                        os.path.join(run_dir_for(args.out_root, config_of(mf), tol, sm, sig),
                                     "run_config.txt"),
                        mf, model, tol, seeds, budget, len(cases), sm, sig)
    rebuild_manifests(args.out_root)
    print("RUN DONE", flush=True)


# ======================================================================================
# recovery / reporting
# ======================================================================================

def rebuild_manifests(out_root):
    """Regenerate every per-run manifest.csv from all_runs.csv.

    Kept separate from the sweep so a failure while writing summaries never costs the
    runs themselves -- the meshes and all_runs.csv are the expensive artefacts.
    """
    rows = list(csv.DictReader(open(os.path.join(out_root, "all_runs.csv"))))
    groups = {}
    for r in rows:
        key = (r["config"], float(r["tolerance"]), r.get("smoother") or "mcf")
        groups.setdefault(key, []).append(r)
    for (cfg, tol, sm), sub in sorted(groups.items()):
        rd = run_dir_for(out_root, cfg, tol, sm)
        os.makedirs(rd, exist_ok=True)
        with open(os.path.join(rd, "manifest.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(sorted(sub, key=lambda r: (r["group"], r["case"])))
    print("rebuilt %d manifests from all_runs.csv" % len(groups))
    return len(groups)


def cmd_rebuild(args):
    rebuild_manifests(args.out_root)


def cmd_summarize(args):
    rows = list(csv.DictReader(open(os.path.join(args.out_root, "all_runs.csv"))))
    num = ("rounds", "dev_max_mm", "dev_mean_mm", "guard", "z_0_3", "z_before_0_3",
           "elapsed_s")
    for r in rows:
        for k in num:
            r[k] = float(r[k]) if r.get(k) else float("nan")
    print("runs=%d  failures=%d" % (len(rows), sum(r["status"] == "FAILED" for r in rows)))
    cfgs = sorted({r["config"] for r in rows})
    tols = sorted({float(r["tolerance"]) for r in rows})
    print("\n%-28s %5s %5s %6s %7s %6s %7s %6s %8s %6s"
          % ("config", "tol", "conv", "stall", "budget", "rnd", "devmax", "guard",
             "z(0.3)", "secs"))
    for c in cfgs:
        for t in tols:
            g = [r for r in rows if r["config"] == c and float(r["tolerance"]) == t]
            if not g:
                continue
            n = lambda s: sum(r["status"] == s for r in g)  # noqa: E731
            print("%-28s %+5.1f %5d %5d %6d %7.1f %6.3f %7.2f %+8.2f %6.0f"
                  % (c, t, n("converged"), n("stalled_at_overshoot_limit"),
                     n("time_budget_exceeded"), np.nanmean([r["rounds"] for r in g]),
                     np.nanmean([r["dev_max_mm"] for r in g]),
                     np.nanmean([r["guard"] for r in g]),
                     np.nanmean([r["z_0_3"] for r in g]),
                     np.nanmean([r["elapsed_s"] for r in g])))
        print()
    print("=== smooth vs noisy ===")
    for t in tols:
        for grp in ("smooth", "noisy"):
            g = [r for r in rows if float(r["tolerance"]) == t and r["group"] == grp]
            if not g:
                continue
            print("  tol%+.1f %-7s z(0.3) %+.2f -> %+.2f   devmax %.3f mm   rounds %4.1f"
                  "   guard %.2f"
                  % (t, grp, np.nanmean([r["z_before_0_3"] for r in g]),
                     np.nanmean([r["z_0_3"] for r in g]),
                     np.nanmean([r["dev_max_mm"] for r in g]),
                     np.nanmean([r["rounds"] for r in g]),
                     np.nanmean([r["guard"] for r in g])))
    el = [r["elapsed_s"] for r in rows if not np.isnan(r["elapsed_s"])]
    if el:
        print("\nelapsed p50=%.0fs p90=%.0fs max=%.0fs | unique meshes %d/%d"
              % (np.percentile(el, 50), np.percentile(el, 90), max(el),
                 len({r["md5"] for r in rows if r.get("md5")}), len(rows)))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_root", default=OUT_ROOT)
    p.add_argument("--model_dir", default=DEFAULT_MODEL_DIR)
    p.add_argument("--ranking", default=os.path.join(DEFAULT_MODEL_DIR, "case_ranking.json"))
    p.add_argument("--n_cases", type=int, default=30)
    p.add_argument("--workers", type=int, default=14)
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("rank", help="rank candidate shapes and pick the case set")
    r.add_argument("--rank_model",
                   default=os.path.join(DEFAULT_MODEL_DIR,
                                        "reference_roughness__t0.3-0.5__g0.8.npz"))
    r.set_defaults(func=cmd_rank)

    s = sub.add_parser("run", help="sweep configs x tolerances x cases")
    s.add_argument("--configs", nargs="*", default=None,
                   help="substrings selecting which cached models to sweep (default all)")
    s.add_argument("--tolerances", type=float, nargs="+", default=[0.0, -1.0, -2.0])
    s.add_argument("--smoothers", nargs="+", default=["mcf"],
                   choices=["mcf", "taubin", "humphrey"],
                   help="smoothing backends to compare")
    s.add_argument("--sigma_steps", type=float, nargs="*", default=None,
                   help="per-round smoothing lengths (mm) to compare; default uses one")
    s.add_argument("--seeds", type=int, default=4000,
                   help="measurement seeds per case; <=0 uses every vertex")
    s.add_argument("--budget", type=float, default=0,
                   help="wall-clock seconds per case; <=0 removes the limit")
    s.set_defaults(func=cmd_run)

    b = sub.add_parser("rebuild", help="regenerate manifests from all_runs.csv")
    b.set_defaults(func=cmd_rebuild)

    z = sub.add_parser("summarize", help="cross-tabulate all_runs.csv")
    z.set_defaults(func=cmd_summarize)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
