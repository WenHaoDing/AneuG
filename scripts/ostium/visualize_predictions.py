"""Render a trained ostium/complex model's predictions for checking by eye.

Per case (two projections each):
  site      the graph coloured by the predicted site probability; true site = star; true ring
            black; full chain (as at inference): k sampled sites (x) with the ostium sampled at
            each (colours), if the dataset has point_frames
  ostium    the true ring (black) and K sampled rings (colours), sampled at the true site
  complex   the true placed complex (grey) and one sampled complex (orange), at the true site
            and the true ostium; the type is sampled from the type head and the complex is decoded
            with the sampled type (as at inference)
  sensor    the frozen morphology sensor on that sampled complex: per-vertex dome probability,
            its p = 0.5 rim (green) against the true ring (black); the red part is the sac that
            would be cut and merged at inference

  CUDA_VISIBLE_DEVICES=1 python scripts/ostium/visualize_predictions.py --ckpt <run>/epoch_0300.pth \
      [--split val|train --cases ... --n 8 --out <dir>]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "train"))


def render_sample(model, b, ys, xs, dec, path, k=4):
    """One case (a batch of 1, on the model's device): site probability, k sampled rings vs the
    true ring at the true site, one sampled complex vs the true one. Saves `path`."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from torch_geometric.utils import softmax
    import train_ostium_model as T
    from models.ostium_model import frame_from_code, ring_from_code
    cx = getattr(model, "with_complex", False)
    cols = plt.get_cmap("tab10")
    was_training = model.training
    model.eval()
    with torch.no_grad():
        enc = model.enc(b)
        pr_t = softmax(model.site_logits(enc, b), b.batch)
        pr = pr_t.cpu().numpy()
        chain = []                                  # full chain: sampled site -> ostium sampled there
        if hasattr(b, "tangent_pts"):
            for _ in range(k):
                idx = T.sample_sites(pr_t, b.batch, 1)
                tg, sc = b.tangent_pts[idx], b.scale_pts[idx]
                y = ys.inv(model.ost.sample(model.ostium_condition(enc, b, idx, tg, sc)))
                chain.append((b.pos[idx][0].cpu().numpy(), T.decode_ring(y, b.pos[idx], tg, sc)[0].cpu().numpy()))
        c1 = model.ostium_condition(enc, b, b.site)
        rings = []
        for _ in range(k):
            y = ys.inv(model.ost.sample(c1))
            Rf = frame_from_code(y[:, :3], b.tangent)
            C = b.pos[b.site] + y[:, :3] * b.scale
            rings.append((C[:, None] + b.scale[:, None] * torch.einsum("bnc,bcd->bnd", ring_from_code(y, 5, 64), Rf))[0].cpu().numpy())
        V_true = V_pred = p_dome = rim_pts = None
        type_txt = ""
        if cx and xs is not None:
            c2 = model.complex_condition(enc, b, ys.fwd(b.y_ost), b.y_ost, b.site)
            p_type = torch.softmax(model.type_head(c2), 1)
            t_s = torch.multinomial(p_type, 1)[:, 0]                      # sampled type, as at inference
            x = xs.inv(model.cx.sample(model.with_type(c2, t_s)))
            phi_h, l_h = dec.phi_from_z(x[:, :16], t_s)
            to_img = lambda Vl: (b.C[:, None] + b.scale[:, None] * torch.einsum("bnc,bcd->bnd", Vl, b.frame))[0].cpu().numpy()
            V_pred = to_img(T.local_vertices(dec.vcan(phi_h, t_s[0]), l_h, x[:, 16:]))
            V_true = to_img(T.local_vertices(dec.vcan(b.phi, b.atype[0]), b.log_scale, b.x_cx[:, 16:]))
            if dec.sensor is not None:
                V_loc = T.local_vertices(dec.vcan(phi_h, t_s[0]), l_h, x[:, 16:])
                p_, X_, _, hard = T.sensor_rim(dec, phi_h, V_loc, t_s)
                p_dome = p_[0].cpu().numpy()
                rim_pts = to_img(X_[:, hard[0]])
            type_txt = (f"   sampled type {int(t_s[0])}   p(type) = "
                        + " / ".join(f"{q:.2f}" for q in p_type[0].tolist()))
    pos = b.pos.cpu().numpy()
    gt = b.ring_img[0].cpu().numpy()
    st = pos[int(b.site_true[0])]
    fig, ax = plt.subplots(2, 4, figsize=(20, 9.5))
    for r, (i, j) in enumerate(((0, 1), (0, 2))):
        a0 = ax[r, 0]
        sc = a0.scatter(pos[:, i], pos[:, j], c=pr, s=2, cmap="magma")
        a0.plot(st[i], st[j], "c*", ms=14, mec="k")
        for k_, (ps, rg) in enumerate(chain):
            a0.plot(np.r_[rg[:, i], rg[:1, i]], np.r_[rg[:, j], rg[:1, j]], color=cols(k_), lw=1.5)
            a0.plot(ps[i], ps[j], "x", color=cols(k_), ms=7, mew=2)
        a0.plot(np.r_[gt[:, i], gt[:1, i]], np.r_[gt[:, j], gt[:1, j]], "k-", lw=1.5)
        a0.set_title(f"site probability (max {pr.max():.3f}); true site *, true ring black;\n"
                     "full chain: sampled sites x + the ostia sampled there (colours)", fontsize=8)
        plt.colorbar(sc, ax=a0, fraction=0.04)
        a1 = ax[r, 1]
        a1.scatter(pos[:, i], pos[:, j], c="0.85", s=1)
        for k_, rg in enumerate(rings):
            a1.plot(np.r_[rg[:, i], rg[:1, i]], np.r_[rg[:, j], rg[:1, j]], color=cols(k_), lw=1.2)
        a1.plot(np.r_[gt[:, i], gt[:1, i]], np.r_[gt[:, j], gt[:1, j]], "k-", lw=2.2)
        c = gt.mean(0)
        w = max(np.ptp(gt[:, i]), np.ptp(gt[:, j])) * 1.8 + 0.2
        a1.set_xlim(c[i] - w, c[i] + w)
        a1.set_ylim(c[j] - w, c[j] + w)
        a1.set_title("ostium: true (black) vs samples, at the true site", fontsize=9)
        a2 = ax[r, 2]
        a2.scatter(pos[:, i], pos[:, j], c="0.85", s=1)
        if V_true is not None:
            a2.scatter(V_true[::2, i], V_true[::2, j], s=0.5, c="0.4", alpha=0.5)
            a2.scatter(V_pred[::2, i], V_pred[::2, j], s=0.5, c="darkorange", alpha=0.5)
        a2.plot(np.r_[gt[:, i], gt[:1, i]], np.r_[gt[:, j], gt[:1, j]], "k-", lw=1.5)
        a2.set_title("complex: true (grey) vs sampled (orange), sampled type", fontsize=9)
        a3 = ax[r, 3]
        a3.scatter(pos[:, i], pos[:, j], c="0.85", s=1)
        if p_dome is not None:
            sc3 = a3.scatter(V_pred[:, i], V_pred[:, j], s=0.8, c=p_dome, cmap="coolwarm", vmin=0, vmax=1)
            plt.colorbar(sc3, ax=a3, fraction=0.04)
            if len(rim_pts):
                a3.scatter(rim_pts[:, i], rim_pts[:, j], s=3, c="lime", edgecolors="none")
            c = gt.mean(0)
            w = max(np.ptp(V_pred[:, i]), np.ptp(V_pred[:, j])) * 0.6 + 0.2
            a3.set_xlim(c[i] - w, c[i] + w)
            a3.set_ylim(c[j] - w, c[j] + w)
        a3.plot(np.r_[gt[:, i], gt[:1, i]], np.r_[gt[:, j], gt[:1, j]], "k-", lw=2)
        a3.set_title("sensor on the sample: p(dome) (red = sac), its 0.5 rim (green) vs true ring (black)", fontsize=8)
        for a_ in ax[r]:
            a_.set_aspect("equal")
    fig.suptitle(f"{b.case[0]}   true type {int(b.atype[0])}{type_txt}   units: 10 mm")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    model.train(was_training)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--overfit_cases", type=int, default=0, help="use the run's first N training cases")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    import os
    assert os.environ.get("CUDA_VISIBLE_DEVICES", "") != "2", "never GPU 2"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from torch_geometric.utils import softmax
    import train_ostium_model as T
    from dataset.ostium_dataset import OstiumGraphDataset, make_loader
    from models.ostium_model import OstiumModel, frame_from_code, ring_from_code
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    args = ck["args"]
    cx = not args.get("no_complex", False)
    model = OstiumModel(d=args["d"], mode=args["mode"], with_complex=cx,
                        norm_tokens=bool(args.get("norm_tokens", 0))).to(dev).eval()
    model.load_state_dict(ck["model"])
    ys = T.Stats(torch.zeros(2, 33)); ys.mean, ys.std = ck["y_stats"]; ys.to(dev)
    xs = None
    if ck.get("x_stats") is not None:
        xs = T.Stats(torch.zeros(2, 26)); xs.mean, xs.std = ck["x_stats"]; xs.to(dev)
    n_over = a.overfit_cases or args.get("overfit", 0)
    if n_over:
        base = OstiumGraphDataset("train", augment=False, complex_targets=cx)
        cases = [c for _, c, _ in base.items[:n_over]]
        ds = OstiumGraphDataset("train", augment=False, complex_targets=cx, cases=cases, point_frames=True)
    else:
        vfile = args.get("val_cases", "none")
        vnames = ([r["case"] for r in __import__("json").load(open(vfile))["cases"]]
                  if vfile and vfile.lower() != "none" else None)        # the run's own held-out list
        ds = OstiumGraphDataset(a.split, augment=False, complex_targets=cx, val_cases=vnames, point_frames=True)
    dec = T.Decoder(ds, dev, with_sensor=True) if cx else None
    out = Path(a.out) if a.out else Path(a.ckpt).parent / f"predictions_{Path(a.ckpt).stem}_{'overfit' if n_over else a.split}"
    out.mkdir(parents=True, exist_ok=True)
    loader = make_loader(ds, 1, False, 0)
    for n_done, b in enumerate(loader):
        if n_done >= a.n:
            break
        render_sample(model, b.to(dev), ys, xs, dec, out / f"{n_done:02d}_{b.case[0][:40]}.png", k=a.k)
    print("wrote", out)


if __name__ == "__main__":
    main()
