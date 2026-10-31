"""Does choosing the site ourselves (a prompt) hurt the generated ostium and dome?

Compares run folders of run_inference.py + run_merge.py made on the same graphs, typically
  model:  sites drawn from the model's site score (run_inference.py --n_sites K)
  prompt: random prompts used as the site exactly (run_inference.py --prompts K --prompt_sigma 0)
and splits the prompted samples by how plausible the model finds the prompted site: the percentile
of its site score among the allowed points of its graph (low / mid / high thirds).

Per sample:
  accepted, rejected / not-built reasons      run_merge.py (inference/merge.py checks)
  rim_inside_lumen, rim move, dome in parent  run_merge.py
  elongation, thickness, sac volume           inference/sac_shape.py
  ring_inside_vessel, skeleton_through_ostium, dome_into_vessel   run_inference.py checks
  ring-rim gap (mm)    symmetric chamfer between the generated ring and the dome's rim (the
                       boundary of the sensor's dome patch): does the dome sit on its own ring?

  python scripts/ostium/compare_site_modes.py --graphs <graph folder> \
      --runs model=runtime_infer/ostium/<A> prompt=runtime_infer/ostium/<B> --out <file.md>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

INFER_FLAGS = ["ring_inside_vessel", "skeleton_through_ostium", "dome_into_vessel"]
REJECT = ["not_dome_like", "dome_into_other_vessel", "dome_inside_lumen"]


def ring_rim_gap(z):
    from scipy.spatial import cKDTree
    from inference.merge import boundary_loop, dome_patch
    dm = dome_patch(z["faces"], z["p_dome"])
    if not dm.any():
        return np.nan
    rim = z["verts_mm"][boundary_loop(z["faces"][dm])]
    ring = z["ring_mm"]
    return float((cKDTree(ring).query(rim)[0].mean() + cKDTree(rim).query(ring)[0].mean()) / 2)


def collect(run, graphs):
    from inference.ostium_inference import load_graph
    run = Path(run)
    infer = {(c["case"], r["sample"]): r for c in json.loads((run / "summary.json").read_text())["cases"] for r in c["samples"]}
    merge = {(r["case"], r["sample"]): r for r in json.loads((run / "merge_summary.json").read_text())}
    rows = []
    for (case, k), ri in sorted(infer.items()):
        rm = merge.get((case, k), {})
        g = load_graph(Path(graphs) / case / "graph.npz")
        score, cover = np.load(run / case / "site_scores.npy")
        allowed = g["in_core"] & (cover > 0)
        pct = float((score[allowed] < score[ri["site"]]).mean() * 100) if allowed.any() else np.nan
        z = np.load(run / case / f"sample_{k:02d}.npz", allow_pickle=True)
        sh = rm.get("shape", {})
        rows.append(dict(
            case=case, sample=k, score_pct=pct, built=rm.get("built", False), accepted=rm.get("accepted", False),
            flags=rm.get("flags", []), infer_flags=ri.get("flags", []),
            rim_move=rm.get("rim_move_median_mm", np.nan), in_parent=rm.get("dome_in_parent_frac", np.nan),
            in_other=rm.get("dome_in_other_vessel_frac", np.nan), volume=rm.get("sac_volume_mm3", np.nan),
            elongation=sh.get("elongation", np.nan), thickness=sh.get("thickness", np.nan), gap=ring_rim_gap(z)))
    return rows


def table(groups):
    """groups: [(name, rows)] -> markdown lines, one column per group."""
    def frac(rows, f):
        return f"{np.mean([f(r) for r in rows]):.0%}" if rows else "-"

    def med(rows, k):
        v = np.array([r[k] for r in rows], float)
        v = v[np.isfinite(v)]
        return f"{np.median(v):.2f}" if len(v) else "-"
    L = ["| | " + " | ".join(f"{n} (n={len(r)})" for n, r in groups) + " |", "|---" * (len(groups) + 1) + "|"]
    spec = [("**accepted**", lambda r: frac(r, lambda x: x["accepted"])),
            ("not built", lambda r: frac(r, lambda x: not x["built"]))]
    spec += [(f"rejected: {f}", (lambda f: lambda r: frac(r, lambda x: f in x["flags"]))(f)) for f in REJECT]
    spec += [("rim inside lumen (projected out)", lambda r: frac(r, lambda x: "rim_inside_lumen" in x["flags"]))]
    spec += [(f"inference check: {f}", (lambda f: lambda r: frac(r, lambda x: f in x["infer_flags"]))(f)) for f in INFER_FLAGS]
    spec += [(f"median {k}", (lambda k: lambda r: med(r, k))(k)) for k in
             ("gap", "rim_move", "in_parent", "volume", "elongation", "thickness", "score_pct")]
    for name, f in spec:
        L.append(f"| {name} | " + " | ".join(f(r) for _, r in groups) + " |")
    return L


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--graphs", required=True)
    p.add_argument("--runs", nargs="+", required=True, help="name=folder")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    runs = [(s.split("=", 1)[0], collect(s.split("=", 1)[1], a.graphs)) for s in a.runs]
    L = ["# Site chosen by the model vs by a prompt", "",
         "gap = ring-rim gap (mm), rim_move = rim projection (mm), in_parent = share of the dome inside the parent "
         "vessel, volume in mm3, score_pct = percentile of the site's score among the graph's allowed points.", "",
         "## By mode", ""] + table(runs)
    for name, rows in runs:
        if name == "model":
            continue
        thirds = [(f"{name}: score pct {lo}-{hi}", [r for r in rows if lo <= r["score_pct"] < hi])
                  for lo, hi in ((0, 33.3), (33.3, 66.7), (66.7, 100.1))]
        L += ["", f"## {name}, by how plausible the model finds the site", ""] + table(thirds)
    text = "\n".join(L) + "\n"
    if a.out:
        Path(a.out).write_text(text)
        json.dump({n: r for n, r in runs}, open(Path(a.out).with_suffix(".json"), "w"), indent=1, default=float)
    print(text)


if __name__ == "__main__":
    main()
