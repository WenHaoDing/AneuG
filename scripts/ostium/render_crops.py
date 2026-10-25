"""Render training crops next to the original graph, for checking the trimming by eye.

Per case: the whole graph, then K crops drawn by the dataset's own trim() (crops only: the 20% of
samples that keep the whole graph look like the original). No rotation, so all panels share axes.
  grey      the whole graph (context)
  blue      points the crop keeps
  green     the protected zone: within keep_mm (6 mm) of the rim and < 15 mm along the graph from
            the site; every crop must keep all of it
  red x     crop centre Q;  cyan star: site P;  black: the true ostium ring
  red dots  cut ends: crop ends that were not ends of the original graph
Title per crop: R (crop radius along the graph), kept vessel length, nearest cut end to the site
(along the graph). Two projections per case (x-y, x-z). Also writes stats.json over many crops.

  python scripts/ostium/render_crops.py [--n_cases 8 --k 5 --out runtime_debug/ostium_model/crop_samples]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def crop_info(ds, d, sel):
    from dataset.ostium_dataset import adjacency, geodesic
    pos, ei, P = d["pos"], d["ei"], d["site"]
    n = len(pos)
    deg0 = np.bincount(ei[0], minlength=n)
    keep = np.zeros(n, bool)
    keep[sel] = True
    m = keep[ei[0]] & keep[ei[1]]
    deg1 = np.bincount(ei[0][m], minlength=n)
    cut = np.flatnonzero(keep & (deg1 < deg0))                       # lost at least one neighbour
    A = adjacency(pos, ei)
    gP = geodesic(A, P)
    # protected zone, as in trim()
    from scipy.spatial import cKDTree
    prot = (cKDTree(d["rim"]).query(pos)[0] < ds.keep_mm) & (gP < 15.0)
    seg = np.linalg.norm(pos[ei[0]] - pos[ei[1]], axis=1)
    length = lambda mask: float(seg[mask[ei[0]] & mask[ei[1]]].sum() / 2)   # edges are stored both ways
    return dict(cut=cut, prot=prot, gP=gP, kept_mm=length(keep), total_mm=length(np.ones(n, bool)),
                cut_from_site_mm=float(gP[cut].min()) if len(cut) else float("inf"),
                prot_kept=bool(keep[prot].all()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_cases", type=int, default=8)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--n_stats", type=int, default=600, help="crops for stats.json")
    p.add_argument("--out", default=str(REPO / "runtime_debug" / "ostium_model" / "crop_samples"))
    a = p.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from dataset.ostium_dataset import OstiumGraphDataset
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    vn = [r["case"] for r in json.load(open(REPO / "dataset" / "ostium_val_cases.json"))["cases"]]
    ds = OstiumGraphDataset("train", augment=True, complex_targets=False, val_cases=vn, seed=0)
    ds.p_whole = 0.0                                                   # show crops only
    rng = np.random.default_rng(0)
    pick = []
    for name in ("ImperialNHS", "AneuX"):
        idx = [i for i, (d_, _, _) in enumerate(ds.items) if d_ == name]
        pick += [idx[int(j)] for j in np.linspace(0, len(idx) - 1, a.n_cases // 2)]
    for ci, i in enumerate(pick):
        d = ds.case_data(i)
        pos, ei, P, rim = d["pos"], d["ei"], d["site"], d["rim"]
        crops = []
        for _ in range(a.k):
            r0 = rng.bit_generator.state
            sel, _ = ds.trim(d, rng)
            crops.append((sel, crop_info(ds, d, sel)))
        fig, ax = plt.subplots(2, a.k + 1, figsize=(3.6 * (a.k + 1), 7.6), squeeze=False)
        info0 = crop_info(ds, d, np.arange(len(pos)))
        lo, hi = pos.min(0), pos.max(0)
        for r, (u, v) in enumerate(((0, 1), (0, 2))):
            for c in range(a.k + 1):
                A_ = ax[r, c]
                A_.scatter(pos[:, u], pos[:, v], s=0.6, c="0.82")
                if c == 0:
                    A_.scatter(pos[info0["prot"], u], pos[info0["prot"], v], s=1.5, c="tab:green")
                    title = f"whole graph: {info0['total_mm']:.0f} mm of vessel"
                else:
                    sel, info = crops[c - 1]
                    A_.scatter(pos[sel, u], pos[sel, v], s=0.8, c="tab:blue")
                    A_.scatter(pos[info["prot"], u], pos[info["prot"], v], s=1.5, c="tab:green")
                    A_.scatter(pos[info["cut"], u], pos[info["cut"], v], s=14, c="red")
                    cut_txt = f"{info['cut_from_site_mm']:.1f}" if np.isfinite(info["cut_from_site_mm"]) else "none"
                    title = (f"crop {c}: keeps {info['kept_mm']:.0f} mm ({100 * info['kept_mm'] / info0['total_mm']:.0f}%)\n"
                             f"nearest cut end {cut_txt} mm; zone kept {info['prot_kept']}")
                A_.plot(np.r_[rim[:, u], rim[:1, u]], np.r_[rim[:, v], rim[:1, v]], "k-", lw=1.2)
                A_.plot(pos[P, u], pos[P, v], "c*", ms=11, mec="k")
                A_.set_xlim(lo[u] - 2, hi[u] + 2)
                A_.set_ylim(lo[v] - 2, hi[v] + 2)
                A_.set_aspect("equal")
                A_.tick_params(labelsize=6)
                if r == 0:
                    A_.set_title(title, fontsize=8)
        fig.suptitle(f"{d['ds']} / {d['case']}   (mm; top x-y, bottom x-z; green = protected zone, "
                     f"blue = kept, red = cut ends, star = site, black = ostium)", fontsize=9)
        fig.tight_layout()
        fig.savefig(out / f"{ci:02d}_{d['case'][:40]}.png", dpi=110)
        plt.close(fig)
        print("wrote", out / f"{ci:02d}_{d['case'][:40]}.png", flush=True)
    # statistics over many crops (crops only)
    st = dict(kept_frac=[], kept_mm=[], cut_from_site_mm=[], prot_kept=[])
    for _ in range(a.n_stats):
        i = int(rng.integers(len(ds)))
        d = ds.case_data(i)
        sel, _ = ds.trim(d, rng)
        info = crop_info(ds, d, sel)
        st["kept_frac"].append(info["kept_mm"] / info["total_mm"])
        st["kept_mm"].append(info["kept_mm"])
        st["cut_from_site_mm"].append(min(info["cut_from_site_mm"], 1e3))
        st["prot_kept"].append(info["prot_kept"])
    q = lambda v: {f"p{k}": round(float(np.percentile(v, k)), 2) for k in (5, 25, 50, 75, 95)}
    res = dict(n_crops=a.n_stats, keep_mm=ds.keep_mm, p_whole_in_training=0.2,
               kept_fraction_of_vessel=q(st["kept_frac"]), kept_vessel_mm=q(st["kept_mm"]),
               nearest_cut_end_from_site_mm=q(st["cut_from_site_mm"]),
               protected_zone_always_kept=bool(all(st["prot_kept"])))
    (out / "stats.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
