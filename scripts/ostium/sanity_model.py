"""Model-side sanity checks (CPU or one GPU, never GPU 2). Writes
runtime_debug/ostium_model/sanity_model/report.md.

  M1  the model's torch decoding (ring_from_code + frame_from_code) of the TRUE code reproduces
      the dataset's target ring (numpy decode_ostium): checks the two implementations agree
  M2  auxiliary losses evaluated at the TRUE complex target (x1_hat = x_true): geo must be ~0
      (exact rebuild through the same functions only if the VAE is bypassed -> here it measures
      the VAE reconstruction floor), rim = the sensor floor
  M3  gradients: the auxiliary losses reach the complex head's parameters (non-zero), and the
      main losses reach the encoder

  CUDA_VISIBLE_DEVICES=1 python scripts/ostium/sanity_model.py
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
OUT = REPO / "runtime_debug" / "ostium_model" / "sanity_model"


def main():
    import os
    assert os.environ.get("CUDA_VISIBLE_DEVICES", "") != "2", "never GPU 2"
    import torch
    sys.path.insert(0, str(REPO / "scripts" / "train"))
    import train_ostium_model as T
    from dataset.ostium_dataset import OstiumGraphDataset, make_loader
    from models.ostium_model import OstiumModel, frame_from_code, ring_from_code
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = OstiumGraphDataset("train", seed=0)
    ys, xs = T.target_stats(ds, passes=1)
    ys.to(dev), xs.to(dev)
    b = next(iter(make_loader(ds, 8, True, 0))).to(dev)
    res = {}
    # M1
    y = b.y_ost
    Rf = frame_from_code(y[:, :3], b.tangent)
    C = b.pos[b.site] + y[:, :3] * b.scale
    ring = C[:, None] + b.scale[:, None] * torch.einsum("bnc,bcd->bnd", ring_from_code(y, 5, 64), Rf)
    res["M1_ring_max_mm"] = float((ring - b.ring_img).norm(dim=2).max() * 10)
    res["M1_frame_max_abs_diff"] = float((Rf - b.frame).abs().max())
    # M2
    dec = T.Decoder(ds, dev, with_sensor=True)

    class A:
        w_geo, w_rim = 1.0, 1.0
    t1 = torch.ones(b.num_graphs, device=dev)
    geo, rim = T.aux_losses(b, b.x_cx, t1, dec, A)
    res["M2_geo_at_truth_units_of_s"] = float(geo)
    res["M2_rim_at_truth_units_of_s2"] = float(rim)
    # same, with a perturbed target: the losses must grow
    xp = b.x_cx.clone()
    xp[:, 16 + 7:] += 0.3
    geo_p, rim_p = T.aux_losses(b, xp, t1, dec, A)
    res["M2_geo_perturbed"], res["M2_rim_perturbed"] = float(geo_p), float(rim_p)
    # M3
    model = OstiumModel(d=64).to(dev)
    enc = model.enc(b)
    y_std = ys.fwd(b.y_ost)
    c2 = model.complex_condition(enc, b, y_std, b.y_ost, b.site)
    _, x1h, tt = model.cx.fm_loss(model.with_type(c2, b.atype), xs.fwd(b.x_cx))
    g, r = T.aux_losses(b, xs.inv(x1h), tt, dec, A)
    (g + r).backward()
    gn = {k: float(sum(p.grad.norm() ** 2 for p in m.parameters() if p.grad is not None) ** 0.5)
          for k, m in (("complex_head", model.cx), ("complex_tokens", model.cx_tf), ("encoder", model.enc))}
    res["M3_aux_grad_norm"] = gn
    model.zero_grad()
    enc = model.enc(b)
    ls, _ = T.site_loss(model, enc, b)
    lo, _, _ = model.ost.fm_loss(model.ostium_condition(enc, b, b.site), y_std)
    (ls + lo).backward()
    res["M3_main_grad_norm_encoder"] = float(sum(p.grad.norm() ** 2 for p in model.enc.parameters() if p.grad is not None) ** 0.5)
    (OUT / "results.json").write_text(json.dumps(res, indent=1))
    L = ["# Model sanity checks", "", "| check | result |", "|---|---|"] + [f"| {k} | {v} |" for k, v in res.items()]
    (OUT / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
