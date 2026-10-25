"""Learning-rate range test and batch-size timing for the ostium model (never GPU 2).

For each mode and batch size: a fresh model and AdamW, with the learning rate rising exponentially
from --lr_min to --lr_max over --steps steps. Same data (training split, augmentation), losses,
weights and gradient clipping as training (train_ostium_model.train_losses). Records the total and
every loss term per step, the wall time per step (data wait included) and the peak GPU memory.
Stops a test early once the smoothed loss is 3x its minimum (diverged).

  CUDA_VISIBLE_DEVICES=1 python scripts/ostium/lr_batch_test.py --gpu 1 [--modes mlp flow --batches 16 32 64]

Writes runtime_debug/ostium_model/lr_batch_test/: runs.jsonl (per step), summary.md, lr_<mode>.png
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "train"))
OUT = REPO / "runtime_debug" / "ostium_model" / "lr_batch_test"


def smooth(v, w=20):
    v = np.asarray(v, float)
    if len(v) < w:
        return v
    c = np.convolve(v, np.ones(w) / w, mode="valid")
    return np.r_[np.full(w - 1, np.nan), c]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--modes", nargs="+", default=["mlp", "flow"])
    p.add_argument("--batches", nargs="+", type=int, default=[16, 32, 64])
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--lr_min", type=float, default=1e-5)
    p.add_argument("--lr_max", type=float, default=1e-2)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--mem_frac", type=float, default=0.45,
                   help="cap on this process's share of GPU memory, so an out-of-memory hits the test and "
                        "not the training runs sharing the GPU")
    t_args = p.parse_args()
    if t_args.gpu == 2:
        sys.exit("GPU 2 must not be used (it overheats and crashes the workstation).")
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(t_args.gpu)
    sys.argv = [sys.argv[0], "--gpu", str(t_args.gpu)]       # training defaults for everything else
    import torch
    import train_ostium_model as T
    from dataset.ostium_dataset import OstiumGraphDataset, make_loader
    from models.ostium_model import OstiumModel
    a = T.get_args()
    dev = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(t_args.mem_frac, 0)
    OUT.mkdir(parents=True, exist_ok=True)
    vnames = [r["case"] for r in json.load(open(a.val_cases))["cases"]]
    tr = OstiumGraphDataset("train", complex_targets=True, seed=a.seed, val_cases=vnames)
    ys, xs = T.target_stats(tr, passes=1)
    ys.to(dev), xs.to(dev)
    dec = T.Decoder(tr, dev, with_sensor=True)
    runs = open(OUT / "runs.jsonl", "w")
    summary = []
    for mode in t_args.modes:
        for B in t_args.batches:
            torch.manual_seed(0)
            a.mode = mode
            model = OstiumModel(d=a.d, mode=mode, with_complex=True, norm_tokens=bool(a.norm_tokens)).to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=t_args.lr_min, weight_decay=1e-4)
            gamma = (t_args.lr_max / t_args.lr_min) ** (1 / (t_args.steps - 1))
            torch.cuda.reset_peak_memory_stats()
            hist, times = [], []
            step, best, diverged = 0, float("inf"), False
            t_prev = time.time()
            try:
                while step < t_args.steps and not diverged:
                    for b in make_loader(tr, B, True, t_args.workers):
                        lr = t_args.lr_min * gamma ** step
                        for g in opt.param_groups:
                            g["lr"] = lr
                        b = b.to(dev)
                        loss, L = T.train_losses(model, b, ys, xs, dec, a, True, step)
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        opt.step()
                        torch.cuda.synchronize()
                        now = time.time()
                        times.append(now - t_prev)
                        t_prev = now
                        rec = dict(mode=mode, batch=B, step=step, lr=lr, loss=float(loss), **{k: float(v) for k, v in L.items()})
                        hist.append(rec)
                        runs.write(json.dumps(rec) + "\n")
                        step += 1
                        sm = smooth([h["loss"] for h in hist])[-1]
                        if not math.isfinite(float(loss)):
                            diverged = True
                        elif np.isfinite(sm):
                            best = min(best, sm)
                            diverged = bool(step > 40 and sm > 3 * best)
                        if diverged or step >= t_args.steps:
                            break
            except torch.OutOfMemoryError:
                print(json.dumps(dict(mode=mode, batch=B, oom_at_step=step)), flush=True)
                summary.append(dict(mode=mode, batch=B, oom_at_step=step))
                del model, opt
                torch.cuda.empty_cache()
                continue
            runs.flush()
            lrs = np.array([h["lr"] for h in hist])
            sm = smooth([h["loss"] for h in hist])
            ok = np.isfinite(sm)
            i_min = int(np.nanargmin(sm))
            slope = np.gradient(np.where(ok, sm, np.nan), np.log10(lrs))
            i_steep = int(np.nanargmin(slope[: i_min + 1])) if i_min > 0 else 0
            st = float(np.median(times[10:])) if len(times) > 20 else float(np.median(times))
            summary.append(dict(mode=mode, batch=B, steps_run=len(hist), diverged=bool(diverged),
                                lr_steepest=float(lrs[i_steep]), lr_min_loss=float(lrs[i_min]),
                                lr_end=float(lrs[-1]), sec_per_step=st, samples_per_sec=B / st,
                                sec_per_epoch=math.ceil(len(tr) / B) * st,
                                peak_mem_gb=torch.cuda.max_memory_allocated() / 1e9))
            print(json.dumps(summary[-1]), flush=True)
            del model, opt
            torch.cuda.empty_cache()
    json.dump(summary, open(OUT / "summary.json", "w"), indent=1)
    plot(OUT)
    write_md(summary, len(tr))


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [json.loads(l) for l in open(out / "runs.jsonl")]
    for mode in sorted({r["mode"] for r in rows}):
        rm = [r for r in rows if r["mode"] == mode]
        keys = [k for k in ("loss", "site", "ost", "ring", "cx", "type", "geo", "rim") if k in rm[0]]
        fig, axes = plt.subplots(2, 4, figsize=(17, 7.5), squeeze=False)
        for ax, k in zip(axes.ravel(), keys):
            for B in sorted({r["batch"] for r in rm}):
                rb = [r for r in rm if r["batch"] == B]
                lr = [r["lr"] for r in rb]
                v = [r[k] for r in rb]
                ax.plot(lr, v, lw=0.5, alpha=0.3)
                ax.plot(lr, smooth(v), lw=2, label=f"batch {B}", color=ax.lines[-1].get_color())
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.axvline(3e-4, color="k", ls=":", lw=1)
            ax.set_title(f"{k}  (dotted: current peak 3e-4)", fontsize=9)
            ax.set_xlabel("learning rate", fontsize=8)
            ax.grid(alpha=0.3, which="both")
            ax.legend(fontsize=7)
        for ax in axes.ravel()[len(keys):]:
            ax.axis("off")
        fig.suptitle(f"learning-rate range test, mode {mode} (thin: per step; thick: 20-step mean)")
        fig.tight_layout()
        fig.savefig(out / f"lr_{mode}.png", dpi=100)
        plt.close(fig)


def write_md(summary, n_train):
    L = ["# Learning-rate range test and batch timing", "",
         f"Training split ({n_train} cases), training losses and weights, AdamW, clip 1.0; learning rate "
         "rising exponentially per step. GPU 1 is shared with the two baseline runs, so the times "
         "are relative.", "",
         "| mode | batch | steps | diverged | lr, steepest descent | lr, lowest loss | s / step | samples / s | s / epoch | peak mem GB |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for s in summary:
        if "oom_at_step" in s:
            L.append(f"| {s['mode']} | {s['batch']} | out of GPU memory at step {s['oom_at_step']} | | | | | | | |")
            continue
        L.append(f"| {s['mode']} | {s['batch']} | {s['steps_run']} | {s['diverged']} | {s['lr_steepest']:.1e} | "
                 f"{s['lr_min_loss']:.1e} | {s['sec_per_step']:.2f} | {s['samples_per_sec']:.1f} | "
                 f"{s['sec_per_epoch']:.1f} | {s['peak_mem_gb']:.2f} |")
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
