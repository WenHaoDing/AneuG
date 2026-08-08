"""
Quantify how varied a synthetic phi batch is, relative to the REAL phi
distribution, per aneurysm type. Pure numpy — runs without the torch env.

Method (matches the PCA-spread plot):
  1. Fit PCA on REAL phi (flattened 432-vec), per type group (bifurcation / sidewall).
  2. Project synth phi onto the real PCs.
  3. spread_ratio[pc] = std(synth_proj[pc]) / std(real_proj[pc])
     (>1 = synth more varied than real along that axis; ~1 = matches real).
  4. inside_frac = fraction of synth points within real's per-PC [min,max] box.

Usage:
  python scripts/evaluate/measure_spread.py --synth v2/synthetic_v1
  python scripts/evaluate/measure_spread.py --synth v2/synthetic_v1 --index-min 0   --index-max 999    # OLD block
  python scripts/evaluate/measure_spread.py --synth v2/synthetic_v1 --index-min 2000 --index-max 2999  # NEW block
  python scripts/evaluate/measure_spread.py --synth /tmp/pilot_var                                      # an amp pilot
"""

import argparse
import glob
import os
import re

import numpy as np

REAL_ROOT = "dataset/processed"
# map both real ("bifurcated") and synth ("bifurcation") labels to one group key
GROUP = {"bifurcated": "bifurcation", "bifurcation": "bifurcation",
         "sidewall": "sidewall"}


def load_real_phi(real_root=REAL_ROOT):
    """Returns {group: (N,432) phi} from the real processed dataset."""
    out = {}
    for f in glob.glob(os.path.join(real_root, "*.npy")):
        d = np.load(f, allow_pickle=True).item()
        g = GROUP.get(str(d.get("canonical_type", "")).lower())
        if g is None:
            continue
        out.setdefault(g, []).append(np.asarray(d["ghd"]["phi"], np.float64).ravel())
    return {g: np.stack(v) for g, v in out.items()}


def load_synth_phi(synth_root, index_min=None, index_max=None):
    """Returns {group: (N,432) phi} from a synth output dir, optional index filter."""
    out = {}
    for d in sorted(glob.glob(os.path.join(synth_root, "synth_*"))):
        m = re.search(r"synth_(\d+)_aneurysm", os.path.basename(d))
        if not m:
            continue
        idx = int(m.group(1))
        if index_min is not None and idx < index_min:
            continue
        if index_max is not None and idx > index_max:
            continue
        try:
            phi = np.load(os.path.join(d, "ghd_coefficients.npz"))["phi"]
            t = str(np.load(os.path.join(d, "landmarks.npz"), allow_pickle=True)["aneu_type"])
        except Exception:
            continue
        g = GROUP.get(t.lower())
        if g is None:
            continue
        out.setdefault(g, []).append(np.asarray(phi, np.float64).ravel())
    return {g: np.stack(v) for g, v in out.items()}


def pca_fit(X, k=10):
    """Fit PCA on X (N,D). Returns (mean, components[k,D], real per-PC std)."""
    mean = X.mean(0)
    U, S, Vt = np.linalg.svd(X - mean, full_matrices=False)
    comp = Vt[:k]
    proj = (X - mean) @ comp.T
    return mean, comp, proj.std(0), proj


def spread_report(real_phi, synth_phi, k=10):
    rows = []
    for g in sorted(set(real_phi) & set(synth_phi)):
        R, S = real_phi[g], synth_phi[g]
        mean, comp, rstd, rproj = pca_fit(R, k)
        sproj = (S - mean) @ comp.T
        ratio = sproj.std(0) / np.where(rstd > 1e-9, rstd, 1)
        lo, hi = rproj.min(0), rproj.max(0)
        inside = float(np.mean(np.all((sproj >= lo) & (sproj <= hi), axis=1)))
        rows.append((g, len(R), len(S), ratio, inside))
    return rows


def _fmt(a):
    return "[" + " ".join(f"{x:.2f}" for x in a) + "]"


def main():
    ap = argparse.ArgumentParser(description="Synth-vs-real phi spread (per type).")
    ap.add_argument("--synth", required=True, help="synth output dir (with synth_* folders)")
    ap.add_argument("--real", default=REAL_ROOT)
    ap.add_argument("--index-min", type=int, default=None)
    ap.add_argument("--index-max", type=int, default=None)
    ap.add_argument("--k", type=int, default=10, help="number of PCs to report mean over")
    args = ap.parse_args()

    real = load_real_phi(args.real)
    synth = load_synth_phi(args.synth, args.index_min, args.index_max)
    if not synth:
        print(f"no synth phi found under {args.synth} (index filter?)")
        return

    print(f"synth: {args.synth}"
          + (f"  idx[{args.index_min}..{args.index_max}]"
             if args.index_min is not None or args.index_max is not None else ""))
    for g, nr, ns, ratio, inside in spread_report(real, synth, args.k):
        print(f"  {g:12s} real={nr:4d} synth={ns:4d}  "
              f"PC1={ratio[0]:.2f} PC2={ratio[1]:.2f}  "
              f"mean(top{args.k})={ratio.mean():.2f}  inside-real-range={inside*100:.0f}%")
    print("  (spread ratio: 1.0 = matches real;  <1 = under-varied;  >1 = over-varied)")


if __name__ == "__main__":
    main()
