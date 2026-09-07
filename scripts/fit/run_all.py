#!/usr/bin/env python3
"""Launch full-corpus fitting across GPUs, CLONE FIRST.

Every case that has an old fitted mesh is run with the `clone` config; the rest
fall back to --fallback-config. Clone jobs are queued ahead of fallback jobs on
every GPU, so the clone results land first.

Results go to runtime/<config>/<dataset>/<case>/ -- named after the CONFIG, so
each config's results for the whole corpus sit together.

Commands come from fit_configs.build_fit_command(), the same builder
manage.py's requeue uses, so the two cannot drift.

  bash scripts/fit/run_all.sh                     # normal launch
  python scripts/fit/run_all.py --dry-run         # print the plan only
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from fit_configs import CONFIGS, build_fit_command, has_clone_source, output_dir  # noqa: E402

GEOMETRY_ROOT = "/media/yaplab2/wd8tb/wenhao/angioflow/data/geometry"
DEFAULT_PYTHON = "/home/yaplab2/miniconda3/envs/new/bin/python"
CONDA_INIT = ('source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || '
              'source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null; conda activate new')


def discover(dataset, geometry_root):
    """Fittable cases: a geometry dir with endpoints_manual.npy."""
    d = Path(geometry_root) / dataset
    if not d.is_dir():
        return []
    return sorted(c.name for c in d.iterdir()
                  if c.is_dir() and (c / "endpoints_manual.npy").exists())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["ImperialNHS", "AneuX"])
    ap.add_argument("--gpus", nargs="+", default=["cuda:0", "cuda:1", "cuda:2"],
                    help="Device, optionally with a worker count: 'cuda:0:4' runs 4 concurrent "
                         "jobs on cuda:0. Jobs are small (~0.4-2.2 GB), so utilisation rather "
                         "than memory is the limit -- a GPU sitting at 20%% can take 3-4.")
    ap.add_argument("--workers-per-gpu", type=int, default=1,
                    help="Default worker count for GPUs given without an explicit ':N'.")
    ap.add_argument("--priority-config", default="clone",
                    help="Used for every case that has an old fitted mesh. Queued first.")
    ap.add_argument("--fallback-config", default="default",
                    help="Used for cases with no old fitted mesh.")
    ap.add_argument("--geometry-root", default=GEOMETRY_ROOT)
    ap.add_argument("--runtime-root", default=str(ROOT / "runtime_fitting"))
    ap.add_argument("--python", default=DEFAULT_PYTHON)
    ap.add_argument("--n-iter", type=int, default=10000)
    ap.add_argument("--stage-a-ratio", type=float, default=None,
                    help="Override the config's own default Stage-A ratio.")
    ap.add_argument("--skip-existing", action="store_true", default=True,
                    help="Skip cases whose output already has metrics.json (default on).")
    ap.add_argument("--redo", dest="skip_existing", action="store_false",
                    help="Re-run even cases that already completed.")
    ap.add_argument("--log-dir", default=str(ROOT / "runtime_fitting" / "_logs"))
    ap.add_argument("--only-cases", nargs="+", default=None,
                    help="Restrict the plan to these case names.")
    ap.add_argument("--tier", choices=["all", "priority", "fallback"], default="all",
                    help="Which tier this launch handles. Lets a clone-only set of workers and a "
                         "fallback-only set run CONCURRENTLY instead of every worker doing clone "
                         "first and only then reaching the fallback tier.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # "cuda:0" -> 1 worker; "cuda:0:4" -> 4. Each worker is its own tmux
    # session and its own serial queue, so N workers on a GPU means N jobs
    # running concurrently there.
    slots = []                      # [(device, session_suffix), ...]
    for spec in args.gpus:
        bits = spec.split(":")
        if len(bits) == 3:
            device, n = f"{bits[0]}:{bits[1]}", int(bits[2])
        else:
            device, n = spec, args.workers_per_gpu
        for k in range(n):
            slots.append((device, f"{device.replace(':', '_')}_w{k}" if n > 1
                          else device.replace(":", "_")))
    if not slots:
        sys.exit("no GPU slots")

    for c in (args.priority_config, args.fallback_config):
        if c not in CONFIGS:
            sys.exit(f"unknown config {c!r}; known: {sorted(CONFIGS)}")

    # ── plan ──────────────────────────────────────────────────────────────
    prio, fall, skipped = [], [], 0
    for ds in args.datasets:
        cases = discover(ds, args.geometry_root)
        if not cases:
            print(f"WARNING: no fittable cases under {args.geometry_root}/{ds}")
        if args.only_cases:
            cases = [c for c in cases if c in set(args.only_cases)]
        for case in cases:
            cfg = args.priority_config if has_clone_source(ds, case) else args.fallback_config
            if CONFIGS[cfg]["needs_old_case"] and not has_clone_source(ds, case):
                cfg = args.fallback_config
            out = output_dir(cfg, ds, case, args.runtime_root)
            if args.skip_existing and (out / "metrics.json").exists():
                skipped += 1
                continue
            (prio if cfg == args.priority_config else fall).append((ds, case, cfg, out))

    if args.tier == "priority":
        fall = []
    elif args.tier == "fallback":
        prio = []

    print(f"Plan: {len(prio)} x {args.priority_config} (priority) + {len(fall)} x "
          f"{args.fallback_config} = {len(prio) + len(fall)} job(s); {skipped} already done")
    for ds in args.datasets:
        p = sum(1 for r in prio if r[0] == ds)
        f = sum(1 for r in fall if r[0] == ds)
        print(f"  {ds:14s} {p:4d} {args.priority_config} + {f:4d} {args.fallback_config}")
    if not prio and not fall:
        print("Nothing to do.")
        return

    # clone first, then fallback -- round-robin within each tier so every GPU
    # works through the priority tier before touching the fallback tier
    queues = [[] for _ in slots]
    for tier in (prio, fall):
        for i, job in enumerate(tier):
            queues[i % len(slots)].append(job)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    for g, (gpu, suffix) in enumerate(slots):
        if not queues[g]:
            continue
        tier_tag = "" if args.tier == "all" else f"_{args.tier[:4]}"
        session = f"ghd_fit{tier_tag}_{suffix}"
        lines = [CONDA_INIT]
        for ds, case, cfg, out in queues[g]:
            cmd, _ = build_fit_command(
                cfg, ds, case, f"{args.geometry_root}/{ds}/{case}", args.runtime_root,
                args.python, gpu, args.n_iter, stage_a_ratio=args.stage_a_ratio)
            # Re-check at RUN time, not just at plan time: workers run for
            # hours, and another worker may have finished this case in the
            # meantime. Without this, overlapping queues redo each other's work.
            body = [
                f"  echo '[{gpu}] === {ds}/{case} ({cfg}) -> {out} ==='",
                f"  if {cmd}; then echo '[{gpu}] OK {cfg} {ds}/{case}';"
                f" else echo '[{gpu}] FAILED {cfg} {ds}/{case}'; fi",
            ]
            if args.skip_existing:
                lines += [f"if [ -f '{out}/metrics.json' ]; then",
                          f"  echo '[{gpu}] SKIP {cfg} {ds}/{case} (already done)'",
                          "else", *body, "fi"]
            else:
                # --redo: the run-time guard would otherwise veto every case the
                # plan deliberately included for re-running.
                lines += [b[2:] if b.startswith("  ") else b for b in body]
        lines.append(f"echo ALL_DONE_{session}")
        # Name by SESSION, not slot index: two concurrent launches (e.g. a
        # priority set and a fallback set) both start their slot numbering at
        # 0, so an index-only name lets the second launch overwrite scripts the
        # first is still executing. bash reads a script by byte offset while
        # running it, so that corrupts the running worker.
        worker = log_dir / f"{session}.sh"
        worker.write_text("\n".join(lines) + "\n")
        n_prio = sum(1 for j in queues[g] if j[2] == args.priority_config)
        print(f"  {session} on {gpu}: {len(queues[g])} job(s) "
              f"({n_prio} {args.priority_config} first)")
        if args.dry_run:
            continue
        subprocess.run(["tmux", "kill-session", "-t", session], stderr=subprocess.DEVNULL)
        subprocess.run(["tmux", "new-session", "-d", "-s", session, f"bash '{worker}'"])
        subprocess.run(["tmux", "pipe-pane", "-t", session, "-o",
                        f"cat >> '{log_dir / (session + '.log')}'"])

    if args.dry_run:
        print("\n(dry run -- worker scripts written, nothing launched)")
    else:
        n_dev = len({d for d, _ in slots})
        print(f"\nLaunched {len(slots)} worker(s) across {n_dev} GPU(s). "
              f"Logs: {log_dir}/ghd_fit_*.log")
        _tt = "" if args.tier == "all" else f"_{args.tier[:4]}"
        print(f"Watch:   tmux attach -t ghd_fit{_tt}_{slots[0][1]}")
        print("Status:  python scripts/fit/manage.py scan --dataset <name> && "
              "python scripts/fit/manage.py review-list")


if __name__ == "__main__":
    main()
