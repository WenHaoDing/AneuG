#!/usr/bin/env python3
"""Run EVERY config on ONE problem case, side by side, into runtime/problem_debug/.

For the configs with a Stage A (arap_init, clone) two ratios are run: the
config's own default, and 20% of it -- rigid = 1/ratio, so a 20% ratio is 5x
the mesh-health weight, i.e. Stage A warps the shape much less. That is the
knob to reach for when a case's Stage A over-warps.

Output: runtime/problem_debug/<config>[_r<ratio>]/<dataset>/<case>/
The ratio goes in the folder name so the two ratios of one config do not
overwrite each other (the normal layout has one folder per config).

  python scripts/fit/problem_debug.py --case cqQUbCpNAd_aneurysm1
  python scripts/fit/problem_debug.py --case X --dataset AneuX --dry-run
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from fit_configs import CONFIGS, old_case_dir, resolve_flags  # noqa: E402

GEOMETRY_ROOT = "/media/yaplab2/wd8tb/wenhao/angioflow/data/geometry"
PY = "/home/yaplab2/miniconda3/envs/new/bin/python"
CONDA_INIT = ('source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || '
              'source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null; conda activate new')
RATIO_SHRINK = 0.2          # "20%" -- 5x the mesh-health weight


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True)
    ap.add_argument("--dataset", default="ImperialNHS")
    ap.add_argument("--gpus", nargs="+", default=["cuda:1", "cuda:2"])
    ap.add_argument("--workers-per-gpu", type=int, default=2)
    ap.add_argument("--n-iter", type=int, default=10000)
    ap.add_argument("--out-root", default=str(ROOT / "runtime" / "problem_debug"))
    ap.add_argument("--configs", nargs="+", default=None,
                    help="Only these config names (default: all). arap_init/clone still expand "
                         "to two ratios each.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    geo = f"{GEOMETRY_ROOT}/{args.dataset}/{args.case}"
    if not Path(geo).is_dir():
        sys.exit(f"no geometry dir: {geo}")

    # ── build the job list ────────────────────────────────────────────────
    jobs = []                              # (tag, config, ratio_or_None)
    for name, cfg in CONFIGS.items():
        if args.configs and name not in args.configs:
            continue
        if cfg["needs_old_case"] and old_case_dir(args.dataset, args.case) is None:
            print(f"  skip {name}: no old fitted mesh for {args.dataset}/{args.case}")
            continue
        if name in ("arap_init", "clone"):
            dflt = cfg["default_stage_a_ratio"]
            for r in (dflt, round(dflt * RATIO_SHRINK, 4)):
                jobs.append((f"{name}_r{r:g}", name, r))
        else:
            jobs.append((name, name, cfg.get("default_stage_a_ratio")))

    print(f"{len(jobs)} run(s) for {args.dataset}/{args.case}:")
    for tag, name, r in jobs:
        print(f"  {tag:<20} {CONFIGS[name]['script'].split('/')[-1]:<24}"
              f"{'r=' + format(r, 'g') if r else ''}")

    # ── one command per job ───────────────────────────────────────────────
    cmds = []
    for tag, name, r in jobs:
        cfg = CONFIGS[name]
        out = Path(args.out_root) / tag / args.dataset / args.case
        out_arg = (f'--save-root "{out.parent}"' if cfg["out_flag"] == "--save-root"
                   else f'--out-dir "{out}"')
        parts = [PY, cfg["script"], f'--case-dir "{geo}"', out_arg]
        if cfg["needs_old_case"]:
            parts.append(f'--old-case-dir "{old_case_dir(args.dataset, args.case)}"')
        parts.append(f"--n-iter {args.n_iter}")
        if cfg["supports_eta_min"]:
            parts.append("--eta-min 1e-4")
        parts.append(resolve_flags(name, r))
        cmds.append((tag, out, parts))

    slots = [(g, k) for g in args.gpus for k in range(args.workers_per_gpu)]
    queues = [[] for _ in slots]
    for i, c in enumerate(cmds):
        queues[i % len(slots)].append(c)

    log_dir = Path(args.out_root) / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    for si, ((gpu, k), queue) in enumerate(zip(slots, queues)):
        if not queue:
            continue
        session = f"dbg_{args.case[:12]}_{gpu.replace(':', '_')}_w{k}"
        lines = [CONDA_INIT]
        for tag, out, parts in queue:
            cmd = " ".join(x for x in parts if x) + f" --device {gpu}"
            lines += [
                f"echo '=== {tag} -> {out} ==='",
                f"if {cmd}; then echo 'OK {tag}'; else echo 'FAILED {tag}'; fi",
            ]
        lines.append(f"echo ALL_DONE_{session}")
        worker = log_dir / f"{session}.sh"
        worker.write_text("\n".join(lines) + "\n")
        print(f"  {session} on {gpu}: {len(queue)} run(s)")
        if args.dry_run:
            continue
        subprocess.run(["tmux", "kill-session", "-t", session], stderr=subprocess.DEVNULL)
        subprocess.run(["tmux", "new-session", "-d", "-s", session, f"bash '{worker}'"])
        subprocess.run(["tmux", "pipe-pane", "-t", session, "-o",
                        f"cat >> '{log_dir / (session + '.log')}'"])
    print("\n(dry run)" if args.dry_run else f"\nLaunched. Logs: {log_dir}/")


if __name__ == "__main__":
    main()
