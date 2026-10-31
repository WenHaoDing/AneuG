"""Are the training graphs' frames consistent across cases and datasets?

The graphs are in each mesh's frame, origin + ijk * spacing (no direction matrix). Without
rotation augmentation the ostium model relies on that frame, so ImperialNHS and AneuX must share
one axis convention. Per case, two anatomical directions:
  inflow   unit vector from the source end (endpoint 0, the inlet; the end of branch 1) to the
           graph's centroid: where the vessel tree goes from its inlet
  axis     the graph's first principal axis (sign-free): the axis the tree mostly lies along
A consistent frame gives one cluster per dataset, at the same place for both datasets. A
swapped or flipped axis shows up as a second cluster.

  python scripts/ostium/check_frames.py      -> runtime_debug/ostium_model/frame_check/
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runtime_debug" / "ostium_model" / "frame_check"


def source(ds, case):
    """Data source: ImperialNHS, or the AneuX contributor from the case-name prefix."""
    if ds == "ImperialNHS":
        return "ImperialNHS"
    for pre, name in (("ANSYS_UNIGE", "ANSYS/USFD_UNIGE"), ("USFD_UNIGE", "ANSYS/USFD_UNIGE"), ("USFD", "USFD"),
                      ("UPF", "UPF"), ("SNF", "SNF"), ("C0", "C00xx"), ("p", "p###")):
        if case.startswith(pre):
            return name
    return "other"


def directions(f):
    z = np.load(f, allow_pickle=True)
    pos, branch, is_end = z["pos"].astype(float), z["branch"], z["is_end"]
    src = np.flatnonzero(is_end & (branch == 1))
    if len(src) == 0:
        return None
    # branch 1 has two ends when it continues into a junction-free chain; the source is the one
    # farthest from the centroid of the rest of the tree
    c = pos.mean(0)
    s = src[np.argmax(np.linalg.norm(pos[src] - c, axis=1))]
    inflow = c - pos[s]
    inflow /= np.linalg.norm(inflow)
    X = pos - c
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    axis = vt[0] * np.sign(vt[0][np.argmax(np.abs(vt[0]))])        # sign-free: largest component > 0
    return inflow, axis


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in ("ImperialNHS", "AneuX"):
        for f in sorted(glob.glob(str(REPO / f"runtime_dataset/assembled/{ds}/*/vessel_graph/vessel_graph.npz"))):
            r = directions(f)
            if r is not None:
                case = Path(f).parents[1].name
                rows.append(dict(ds=ds, src=source(ds, case), case=case, inflow=r[0], axis=r[1]))
    summ = {}
    for ds in ("ImperialNHS", "AneuX"):
        R = [r for r in rows if r["ds"] == ds]
        I = np.array([r["inflow"] for r in R])
        A = np.array([r["axis"] for r in R])
        mean_in = I.mean(0)
        dom = np.abs(I).argmax(1)
        summ[ds] = dict(
            n=len(R),
            inflow_mean=np.round(mean_in, 3).tolist(),
            inflow_mean_length=round(float(np.linalg.norm(mean_in)), 3),      # 1 = all identical, 0 = random
            inflow_dominant_axis_counts={f"{'xyz'[k]}{'+' if s > 0 else '-'}": int(((dom == k) & (np.sign(I[np.arange(len(I)), dom]) == s)).sum())
                                         for k in range(3) for s in (1, -1)},
            main_axis_dominant_counts={"xyz"[k]: int((np.abs(A).argmax(1) == k).sum()) for k in range(3)},
        )
        u = mean_in / np.linalg.norm(mean_in)
        for r in R:
            r["angle_to_mean_deg"] = float(np.degrees(np.arccos(np.clip(r["inflow"] @ u, -1, 1))))
    # the two datasets' mean inflow directions against each other
    a, b = (np.array(summ[d]["inflow_mean"]) for d in ("ImperialNHS", "AneuX"))
    summ["angle_between_dataset_means_deg"] = round(float(np.degrees(np.arccos(a @ b / np.linalg.norm(a) / np.linalg.norm(b)))), 1)
    outl = sorted([r for r in rows if r["angle_to_mean_deg"] > 90], key=lambda r: -r["angle_to_mean_deg"])
    summ["cases_inflow_over_90deg_from_their_dataset_mean"] = {
        ds: [f"{r['case']} ({r['angle_to_mean_deg']:.0f} deg)" for r in outl if r["ds"] == ds] for ds in ("ImperialNHS", "AneuX")}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    # figure: inflow directions in three projections (one colour per data source), and components
    srcs = sorted({r["src"] for r in rows}, key=lambda k: -sum(r["src"] == k for r in rows))
    col = {k: plt.get_cmap("tab10")(n) for n, k in enumerate(srcs)}
    fig, ax = plt.subplots(2, 3, figsize=(15, 9.5))
    for c_, (i, j) in enumerate(((0, 1), (0, 2), (1, 2))):
        A_ = ax[0, c_]
        A_.add_patch(plt.Circle((0, 0), 1, fill=False, color="0.7"))
        for k in srcs:
            I = np.array([r["inflow"] for r in rows if r["src"] == k])
            A_.scatter(I[:, i], I[:, j], s=9, alpha=0.6, color=col[k], label=f"{k} ({len(I)})")
            m = I.mean(0)
            A_.arrow(0, 0, m[i], m[j], color=col[k], width=0.012, length_includes_head=True)
        A_.set_xlim(-1.1, 1.1)
        A_.set_ylim(-1.1, 1.1)
        A_.set_aspect("equal")
        A_.set_xlabel("xyz"[i])
        A_.set_ylabel("xyz"[j])
        A_.set_title(f"inflow direction, {'xyz'[i]}-{'xyz'[j]} projection (arrows: source means)", fontsize=9)
        A_.legend(fontsize=7)
        B_ = ax[1, c_]
        for k in srcs:
            I = np.array([r["inflow"] for r in rows if r["src"] == k])
            B_.hist(I[:, c_], bins=24, range=(-1, 1), histtype="step", lw=1.6, color=col[k], label=k, density=True)
        B_.set_title(f"inflow, {'xyz'[c_]} component", fontsize=9)
        B_.legend(fontsize=7)
    fig.suptitle("Frame check: direction from each graph's inlet to its centroid, mesh frame (origin + ijk * spacing)")
    fig.tight_layout()
    fig.savefig(OUT / "inflow_directions.png", dpi=110)
    plt.close(fig)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
