"""How well generated complexes sit on their 1D vessel graph, for trained checkpoints.

For each checkpoint, on the held-out cases and on a fixed set of training cases (no augmentation),
with the training script's own validation (_validate):
  gt_cap_graph_mm / gt_tube_mm        the true complex (decoded, placed in the true ostium frame):
                                      the floor these measures can reach
  cap_graph_mm / tube_mm              complexes sampled at the true site and true ostium
  chain_cap_graph_mm / chain_tube_mm  the full chain at sites sampled from the site head
  uni_*                               the same at uniform sites (>= 5 mm from a graph end), as for prompts
cap = distance (mm) from each cap centre (the sensor's cap distribution weighting the placed
vertices) to the nearest graph point; tube = mean |min_i(|v - c_i| - r_i)| (mm) over parent-vessel
vertices (sensor: not dome, not cap). Points whose nearest graph point is a graph end are left out.

  python scripts/ostium/measure_graph_fit.py runtime_train/ostium_model/<run>/epoch_1000.pth [...] \
      [--train_cases 30] [--device cpu] [--out file.md]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "train"))

KEYS = ["gt_cap_graph_mm", "cap_graph_mm", "chain_cap_graph_mm", "uni_cap_graph_mm", "gt_tube_mm", "tube_mm", "chain_tube_mm",
        "uni_tube_mm", "chain_selfrim_mm", "uni_selfrim_mm"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpts", nargs="+")
    p.add_argument("--train_cases", type=int, default=30)
    p.add_argument("--device", default="cpu")
    p.add_argument("--k_samples", type=int, default=4)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    import os
    if a.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    import train_ostium_model as T
    from dataset.ostium_dataset import OstiumGraphDataset, make_loader
    from models.ostium_model import OstiumModel
    torch.set_num_threads(8)
    dev = torch.device(a.device)
    vnames = [r["case"] for r in json.load(open(REPO / "dataset" / "ostium_val_cases.json"))["cases"]]
    va = OstiumGraphDataset("val", augment=False, complex_targets=True, val_cases=vnames, point_frames=True)
    base = OstiumGraphDataset("train", augment=False, complex_targets=True, val_cases=vnames)
    names = [c for _, c, _ in base.items]
    pick = [names[int(j)] for j in np.linspace(0, len(names) - 1, a.train_cases)]
    tr = OstiumGraphDataset("train", augment=False, complex_targets=True, val_cases=vnames, cases=pick, point_frames=True)
    dec = None
    L = ["| checkpoint | cases | " + " | ".join(KEYS) + " |", "|---" * (len(KEYS) + 2) + "|"]
    for ck_path in a.ckpts:
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        args = ck["args"]
        A = argparse.Namespace(**{**T.DEFAULTS, **args})
        model = OstiumModel(d=args["d"], mode=args["mode"], with_complex=True, norm_tokens=bool(args.get("norm_tokens", 0)),
                            flow_h=args.get("flow_h", 256), flow_blocks=args.get("flow_blocks", 4)).to(dev)
        model.load_state_dict(ck["model"])
        ys, xs = T.Stats(torch.zeros(2, 33)), T.Stats(torch.zeros(2, 26))
        ys.mean, ys.std = ck["y_stats"]
        xs.mean, xs.std = ck["x_stats"]
        ys.to(dev)
        xs.to(dev)
        if dec is None:
            dec = T.Decoder(tr, dev, with_sensor=True)
        name = f"{Path(ck_path).parent.name} ep{ck['epoch']}"
        for split, ds in (("held-out", va), ("training", tr)):
            torch.manual_seed(0)
            with torch.no_grad():
                m, _ = T._validate(model, make_loader(ds, 4, False, 0), ys, xs, dec, A, dev, k_samples=a.k_samples)
            L.append(f"| {name} | {split} ({len(ds)}) | " + " | ".join(f"{m.get(k, float('nan')):.2f}" for k in KEYS) + " |")
            print(L[-1], flush=True)
    text = "# Generated complexes on the 1D graph (mm)\n\n" + "\n".join(L) + "\n"
    if a.out:
        Path(a.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
