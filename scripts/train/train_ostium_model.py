"""Train the ostium model (site + ostium code) and the complex model (GHD latent + affine).

Design: TASK_vessel_conditioned_aneurysm.md section 5.6. Data: dataset/ostium_dataset.py.

Losses: site soft cross-entropy; flow matching (or MSE in --mode mlp) on the standardised
ostium code y; flow matching on the standardised complex target x = [z, affine]; aneurysm-type
cross-entropy; and two LOW-WEIGHT auxiliary losses on the one-step estimate
x1_hat = x_t + (1 - t) v (direct prediction in mlp mode), weighted by t:
  geo  vertex-wise distance between the decoded complex (frozen VAE + GHD) and the true one,
       in the ostium frame
  rim  weighted chamfer between the frozen morphology sensor's soft rim on the decoded complex
       and the target ring

Never uses GPU 2 (it overheats and crashes the workstation).

  python scripts/train/train_ostium_model.py --gpu 1 --name full_flow --epochs 600     # all cases
  python scripts/train/train_ostium_model.py --gpu 1 --name overfit --overfit 8 --epochs 300
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
SENSOR = REPO / "runtime_train/morphoformer/morphology_sensor/h128_gps4_tw1_pw2_dome1_phi0.5_rot0.5/epoch_02000.pth"


def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--name", default="run")
    p.add_argument("--out_root", default=str(REPO / "runtime_train" / "ostium_model"))
    p.add_argument("--mode", choices=["flow", "mlp"], default="flow")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no_complex", action="store_true", help="train the ostium model only")
    p.add_argument("--w_site", type=float, default=1.0)
    p.add_argument("--w_ost", type=float, default=1.0)
    p.add_argument("--w_cx", type=float, default=1.0)
    p.add_argument("--w_type", type=float, default=0.1)
    p.add_argument("--w_geo", type=float, default=0.01)
    p.add_argument("--w_rim", type=float, default=0.01)
    p.add_argument("--w_ring", type=float, default=0.1,
                   help="ostium ring point loss (true vs predicted ring, point to point, units of s); "
                        "on the one-step estimate weighted by t in flow mode, on the prediction in mlp mode")
    p.add_argument("--norm_tokens", type=int, default=1,
                   help="1: graph-token positions as [direction, log(1+distance)] and log(s) centred; "
                        "0: the raw relative positions (as in the first runs)")
    p.add_argument("--aux_every", type=int, default=1, help="compute the auxiliary losses every N steps")
    p.add_argument("--val_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=50)
    p.add_argument("--sanity_every", type=int, default=50,
                   help="render the fixed sanity cases (and their metrics) every N epochs")
    p.add_argument("--sanity_cases", type=int, default=4)
    p.add_argument("--overfit", type=int, default=0, help="train and validate on N training cases, no augmentation")
    p.add_argument("--val_cases", default=str(REPO / "dataset" / "ostium_val_cases.json"),
                   help="json list of held-out cases (split by patient); 'none' trains on all cases")
    p.add_argument("--holdout", action="store_true",
                   help="debug only: hold out ~15%% of patients and validate on them. By default the model "
                        "trains on ALL cases with no validation: the task is generative, so it is evaluated "
                        "with generative metrics, not a held-out reconstruction error")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rotate", type=int, default=1,
                   help="1: random SO(3) rotation per training sample; 0: keep the scan orientation "
                        "(trimming, sigma flips and site jitter unchanged)")
    p.add_argument("--stop_at", type=int, default=0,
                   help="stop after this epoch (0: run all --epochs); the schedule still follows --epochs")
    a = p.parse_args()
    if a.gpu == 2:
        sys.exit("GPU 2 must not be used (it overheats and crashes the workstation).")
    return a


class Stats:
    """Per-dimension standardisation of a target."""

    def __init__(self, x):
        self.mean = x.mean(0)
        self.std = x.std(0) + 1e-3

    def to(self, dev):
        self.mean, self.std = self.mean.to(dev), self.std.to(dev)
        return self

    def fwd(self, x):
        return (x - self.mean) / self.std

    def inv(self, x):
        return x * self.std + self.mean


def target_stats(ds, passes=2):
    import torch
    ys, xs = [], []
    for _ in range(passes):
        for i in range(len(ds)):
            s = ds[i]
            ys.append(s.y_ost[0])
            if hasattr(s, "x_cx"):
                xs.append(s.x_cx[0])
    return Stats(torch.stack(ys)), (Stats(torch.stack(xs)) if xs else None)


class Decoder:
    """Frozen VAE decoder + GHD reconstruction + morphology sensor, differentiable in the input."""

    def __init__(self, ds, dev, with_sensor):
        import copy
        import torch
        from models.multi_canonical_ghd_reconstruct import MultiCanonicalGHDReconstruct
        self.vae = copy.deepcopy(ds._vae).to(dev).eval()     # a copy: the dataset keeps encoding on the CPU
        for prm in self.vae.parameters():
            prm.requires_grad_(False)
        self.mean, self.std = ds._vae_mean.to(dev), ds._vae_std.to(dev)
        self.mc = MultiCanonicalGHDReconstruct(str(REPO / "dataset" / "canonical"))
        self.dev = dev
        self.sensor = None
        if with_sensor:
            # without gradients (validation, renders) nn.MultiheadAttention takes its inference fast
            # path, which builds the full vertex x vertex attention: ~1.1 GB per complex, 10x the
            # training path. Off, the sensor's output is the same (|logit difference| ~1e-4).
            torch.backends.mha.set_fastpath_enabled(False)
            from models.morphoformer import SensorUncapper
            self.sensor = SensorUncapper.from_checkpoint(str(SENSOR), self.mc, device=dev).model
        self.edges = {}

    def phi_from_z(self, z, atype):
        import torch
        out, sc = self.vae.decode(z, atype)
        xd = torch.cat([out, sc], 1) * self.std + self.mean
        return xd[:, :-1].reshape(len(z), -1, 3), torch.log(xd[:, -1].clamp(min=1e-4))

    def vcan(self, phi, t):
        rec = self.mc.get(int(t))
        return rec.ghd_forward_as_Meshes(phi.reshape(len(phi), -1).to(rec.GHD_eigvec.device),
                                         denormalize_shape=False).verts_padded().to(self.dev)

    def faces_edges(self, t):
        import torch
        if t not in self.edges:
            rec = self.mc.get(int(t))
            m = rec.ghd_forward_as_Meshes(torch.zeros(1, 432, device=rec.GHD_eigvec.device), denormalize_shape=False)
            F_ = m.faces_packed()
            e = torch.cat([F_[:, [0, 1]], F_[:, [1, 2]], F_[:, [2, 0]]]).sort(1).values.unique(dim=0)
            self.edges[t] = e.to(self.dev)
        return self.edges[t]


def rot_from_6d(v):
    import torch.nn.functional as F
    import torch
    a1, a2 = v[:, :3], v[:, 3:6]
    m1 = F.normalize(a1, dim=1)
    m2 = F.normalize(a2 - (a2 * m1).sum(1, keepdim=True) * m1, dim=1)
    return torch.stack([m1, m2, torch.cross(m1, m2, dim=1)], 2)


def local_vertices(vc, log_scale, aff):
    """Canonical vertices + affine (10) -> vertices in the ostium frame (units of s)."""
    import torch
    M = rot_from_6d(aff[:, :6])
    return (torch.exp(aff[:, 6:7]) * torch.exp(log_scale)[:, None])[..., None] * torch.bmm(vc, M) + aff[:, None, 7:10]


def sensor_rim(dec, phi_h, V_h, atype):
    """Frozen sensor on decoded complexes of one type: per-vertex dome probability p (b, V), the
    p = 0.5 crossing point X on every mesh edge (b, E, 3, same frame as V_h), the soft crossing
    weight we (b, E) and the hard crossing mask (b, E)."""
    import torch
    p = torch.sigmoid(dec.sensor(phi_h, atype)[4]["dome_logits"])[:, :V_h.size(1)]
    e = dec.faces_edges(int(atype[0]))
    pi, pj = p[:, e[:, 0]], p[:, e[:, 1]]
    we = pi * (1 - pj) + pj * (1 - pi)
    den = torch.where((pj - pi).abs() < 1e-4, torch.full_like(pi, 1e-4), pj - pi)
    lam = ((0.5 - pi) / den).clamp(0, 1)
    X = V_h[:, e[:, 0]] + lam[..., None] * (V_h[:, e[:, 1]] - V_h[:, e[:, 0]])
    return p, X, we, (pi - 0.5) * (pj - 0.5) < 0


def ring_loss(y_hat, y_true, tangent, t_w, n=64):
    """Ostium ring point loss: the predicted and true codes decoded to n ring points relative to the
    site, in units of s (centre offset + ring in the frame from n and the tangent). Both rings start
    on +e1 and run counter-clockwise, so point i matches point i. Mean squared distance, weighted by t_w."""
    import torch
    from models.ostium_model import frame_from_code, ring_from_code

    def pts(y):
        return y[:, None, :3] + torch.einsum("bnc,bcd->bnd", ring_from_code(y, 5, n), frame_from_code(y[:, :3], tangent))
    return (((pts(y_hat) - pts(y_true)) ** 2).sum(2).mean(1) * t_w).mean()


def aux_losses(b, x_hat, t_w, dec, a):
    """geo + rim on the destandardised complex estimate x_hat (B, 26), weighted by t_w."""
    import torch
    geo = x_hat.new_zeros(())
    rim = x_hat.new_zeros(())
    z, aff = x_hat[:, :16], x_hat[:, 16:]
    for t in b.atype.unique():
        m = b.atype == t
        phi_h, l_h = dec.phi_from_z(z[m], b.atype[m])
        V_h = local_vertices(dec.vcan(phi_h, t), l_h, aff[m])
        w = t_w[m]
        if a.w_geo > 0:
            V_t = local_vertices(dec.vcan(b.phi[m], t), b.log_scale[m], b.x_cx[m][:, 16:])
            geo = geo + ((V_h - V_t).norm(dim=2).mean(1) * w).sum() / len(b.atype)
        if a.w_rim > 0 and dec.sensor is not None:
            _, X, we, _ = sensor_rim(dec, phi_h, V_h, b.atype[m])
            ring = b.ring_loc[m]
            d = torch.cdist(X, ring)                                  # (b, E, 64)
            fwd = (we * d.min(2).values ** 2).sum(1) / we.sum(1).clamp(min=1e-6)
            bwd = (d ** 2 + 0.25 * (1 - we)[..., None]).min(1).values.mean(1)
            rim = rim + ((fwd + bwd) * w).sum() / len(b.atype)
    return geo, rim


def site_loss(model, enc, b):
    import torch
    from torch_geometric.utils import scatter
    lg = model.site_logits(enc, b)
    mx = scatter(lg, b.batch, reduce="max")[b.batch]
    lse = torch.log(scatter((lg - mx).exp(), b.batch, reduce="sum"))[b.batch] + mx
    return -scatter(b.site_q * (lg - lse), b.batch, reduce="sum").mean(), lg


@__import__("torch").no_grad()
def decode_ring(y, p_site, tangent, scale, n=64):
    """Destandardised code at a site (position, signed tangent, scale; batch units) -> ring points."""
    import torch
    from models.ostium_model import frame_from_code, ring_from_code
    Rf = frame_from_code(y[:, :3], tangent)
    C = p_site + y[:, :3] * scale
    return C[:, None] + scale[:, None] * torch.einsum("bnc,bcd->bnd", ring_from_code(y, 5, n), Rf)


def sample_sites(pr, batch, num_graphs):
    """One site per graph, drawn from the site distribution."""
    import torch
    return torch.stack([m[torch.multinomial(pr[m], 1)[0]]
                        for m in (torch.nonzero(batch == i).squeeze(1) for i in range(num_graphs))])


def train_losses(model, b, ys, xs, dec, a, cx, step):
    """One training step's losses on a batch (already on the device): (weighted total, dict of terms)."""
    import torch
    enc = model.enc(b)
    L = {}
    L["site"], _ = site_loss(model, enc, b)
    c1 = model.ostium_condition(enc, b, b.site)
    L["ost"], y1h, ty = model.ost.fm_loss(c1, ys.fwd(b.y_ost))
    if a.w_ring > 0:
        L["ring"] = ring_loss(ys.inv(y1h), b.y_ost, b.tangent, ty)
    if cx:
        y_std = ys.fwd(b.y_ost)
        c2 = model.complex_condition(enc, b, y_std, b.y_ost, b.site)
        L["cx"], x1h, tt = model.cx.fm_loss(model.with_type(c2, b.atype), xs.fwd(b.x_cx))
        L["type"] = torch.nn.functional.cross_entropy(model.type_head(c2), b.atype)
        if (a.w_geo > 0 or a.w_rim > 0) and step % a.aux_every == 0:
            L["geo"], L["rim"] = aux_losses(b, xs.inv(x1h), tt, dec, a)
    loss = (a.w_site * L["site"] + a.w_ost * L["ost"] + a.w_cx * L.get("cx", 0) + a.w_type * L.get("type", 0)
            + a.w_geo * L.get("geo", 0) + a.w_rim * L.get("rim", 0) + a.w_ring * L.get("ring", 0))
    return loss, L


def validate(model, loader, ys, xs, dec, a, dev, k_samples=4):
    """Metrics without gradients; the cached GPU memory is released afterwards, so two runs (or a
    run and a test) can share one GPU."""
    import torch
    with torch.no_grad():
        out = _validate(model, loader, ys, xs, dec, a, dev, k_samples)
    torch.cuda.empty_cache()
    return out


def _validate(model, loader, ys, xs, dec, a, dev, k_samples=4):
    import torch
    from geometry.complex_target import rot_from_6d as _r6  # noqa: F401
    from models.ostium_model import frame_from_code, ring_from_code
    model.eval()
    rows = []
    for b in loader:
        b = b.to(dev)
        enc = model.enc(b)
        lg = model.site_logits(enc, b)
        from torch_geometric.utils import softmax
        pr = softmax(lg, b.batch)
        for i in range(b.num_graphs):
            m = torch.nonzero(b.batch == i).squeeze(1)
            top = m[pr[m].argmax()]
            d_top = float((b.pos[top] - b.pos[b.site_true[i]]).norm() * 10)
            near = (b.pos[m] - b.pos[b.site_true[i]]).norm(dim=1) * 10 < 2.0
            rows.append(dict(case=b.case[i], site_top1_mm=d_top, site_mass_2mm=float(pr[m][near].sum())))
        c1 = model.ostium_condition(enc, b, b.site)
        errs, offs = [], []
        for _ in range(k_samples):
            y = ys.inv(model.ost.sample(c1))
            Rf = frame_from_code(y[:, :3], b.tangent)
            C = b.pos[b.site] + y[:, :3] * b.scale
            ring = C[:, None] + b.scale[:, None] * torch.einsum("bnc,bcd->bnd", ring_from_code(y, 5, 64), Rf)
            d = torch.cdist(ring, b.ring_img)
            errs.append(10 * 0.5 * (d.min(2).values.mean(1) + d.min(1).values.mean(1)))
            offs.append(10 * ((y[:, :3] - b.y_ost[:, :3]) * b.scale).norm(dim=1))
        errs, offs = torch.stack(errs), torch.stack(offs)
        for i in range(b.num_graphs):
            rows[-b.num_graphs + i].update(ring_mean_mm=float(errs[:, i].mean()), ring_best_mm=float(errs[:, i].min()),
                                           centre_err_mm=float(offs[:, i].mean()))
        if hasattr(b, "tangent_pts"):
            # full chain, as at inference: sampled site -> ostium sampled there -> distance to the true ring
            ch_r, ch_c = [], []
            gt_c = b.ring_img.mean(1)
            for _ in range(k_samples):
                idx = sample_sites(pr, b.batch, b.num_graphs)
                tg, sc = b.tangent_pts[idx], b.scale_pts[idx]
                y = ys.inv(model.ost.sample(model.ostium_condition(enc, b, idx, tg, sc)))
                ring = decode_ring(y, b.pos[idx], tg, sc)
                d = torch.cdist(ring, b.ring_img)
                ch_r.append(10 * 0.5 * (d.min(2).values.mean(1) + d.min(1).values.mean(1)))
                ch_c.append(10 * (ring.mean(1) - gt_c).norm(dim=1))
            ch_r, ch_c = torch.stack(ch_r), torch.stack(ch_c)
            for i in range(b.num_graphs):
                rows[-b.num_graphs + i].update(chain_centre_mm=float(ch_c[:, i].mean()), chain_ring_mm=float(ch_r[:, i].mean()),
                                               chain_ring_best_mm=float(ch_r[:, i].min()))
        if xs is not None and hasattr(model, "cx"):
            y_std = ys.fwd(b.y_ost)
            c2 = model.complex_condition(enc, b, y_std, b.y_ost, b.site)
            tacc = (model.type_head(c2).argmax(1) == b.atype).float()
            verr, rerr = [], []
            for _ in range(min(k_samples, 2)):
                x = xs.inv(model.cx.sample(model.with_type(c2, b.atype)))   # teacher-forced type
                ve = torch.zeros(b.num_graphs, device=dev)
                re = torch.full((b.num_graphs,), float("nan"), device=dev)
                for t in b.atype.unique():
                    mm = b.atype == t
                    idx = torch.nonzero(mm).squeeze(1)
                    phi_h, l_h = dec.phi_from_z(x[mm][:, :16], b.atype[mm])
                    V_h = local_vertices(dec.vcan(phi_h, t), l_h, x[mm][:, 16:])
                    V_t = local_vertices(dec.vcan(b.phi[mm], t), b.log_scale[mm], b.x_cx[mm][:, 16:])
                    ve[idx] = (V_h - V_t).norm(dim=2).mean(1) * b.scale[mm, 0] * 10
                    if dec.sensor is not None:
                        # the sac the sensor cuts from the sampled complex: its p = 0.5 rim vs the true ring
                        _, X, _, hard = sensor_rim(dec, phi_h, V_h, b.atype[mm])
                        for j, i in enumerate(idx.tolist()):
                            if hard[j].any():
                                d = torch.cdist(X[j][hard[j]], b.ring_loc[i])
                                re[i] = 0.5 * (d.min(1).values.mean() + d.min(0).values.mean()) * b.scale[i, 0] * 10
                verr.append(ve)
                rerr.append(re)
            verr, rerr = torch.stack(verr), torch.stack(rerr)
            for i in range(b.num_graphs):
                rows[-b.num_graphs + i].update(complex_vertex_mm=float(verr[:, i].mean()), type_ok=float(tacc[i]))
                if dec.sensor is not None:
                    ok = ~torch.isnan(rerr[:, i])
                    rows[-b.num_graphs + i].update(sensor_rim_found=float(ok.float().mean()))
                    if ok.any():
                        rows[-b.num_graphs + i].update(sensor_rim_mm=float(rerr[ok, i].mean()))
    model.train()
    keys = [k for k in rows[0] if k != "case"]
    return {k: float(np.mean([r[k] for r in rows if k in r])) for k in keys}, rows


def main():
    a = get_args()
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)          # never 2 (checked in get_args)
    import torch
    from dataset.ostium_dataset import OstiumGraphDataset, make_loader
    from models.ostium_model import OstiumModel
    torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out_root) / a.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=1))
    cx = not a.no_complex
    if a.overfit:
        base = OstiumGraphDataset("train", augment=False, complex_targets=cx, seed=a.seed)
        cases = [c for _, c, _ in base.items[:a.overfit]]
        tr = OstiumGraphDataset("train", augment=False, complex_targets=cx, cases=cases, seed=a.seed)
        va = OstiumGraphDataset("train", augment=False, complex_targets=cx, cases=cases, seed=a.seed)
    elif a.holdout:
        tr = OstiumGraphDataset("train", augment=True, rotate=bool(a.rotate), complex_targets=cx, seed=a.seed)
        va = OstiumGraphDataset("val", augment=False, complex_targets=cx, seed=a.seed)
    elif a.val_cases.lower() != "none":
        vnames = [r["case"] for r in json.load(open(a.val_cases))["cases"]]
        tr = OstiumGraphDataset("train", augment=True, rotate=bool(a.rotate), complex_targets=cx, seed=a.seed, val_cases=vnames)
        va = OstiumGraphDataset("val", augment=False, complex_targets=cx, seed=a.seed, val_cases=vnames,
                                point_frames=True)
    else:
        # augment explicitly: the dataset's default is augment = (split == "train"), which for
        # split "all" silently turned trimming, rotation, sigma flips and site jitter off
        tr = OstiumGraphDataset("all", augment=True, rotate=bool(a.rotate), complex_targets=cx, seed=a.seed)
        va = None
    print(f"train {len(tr)} cases, {'val ' + str(len(va)) if va is not None else 'no validation'}, on {dev}", flush=True)
    ys, xs = target_stats(tr, passes=1 if a.overfit else 2)
    ys.to(dev)
    if xs:
        xs.to(dev)
    tl = make_loader(tr, a.batch, True, a.workers)
    # small validation batches: the frozen sensor's vertex attention grows with the batch
    vl = make_loader(va, min(a.batch, 4), False, max(1, a.workers // 2)) if va is not None else None
    # fixed sanity cases: training cases without augmentation, spread over both datasets
    pick = []
    for dsn in ("ImperialNHS", "AneuX"):
        names = [c for d_, c, _ in tr.items if d_ == dsn]
        k = a.sanity_cases // 2 if dsn == "ImperialNHS" else a.sanity_cases - a.sanity_cases // 2
        pick += [names[int(j)] for j in np.linspace(0, len(names) - 1, k)] if names and k else []
    san = OstiumGraphDataset("all", augment=False, complex_targets=cx, cases=pick, seed=a.seed, point_frames=True)
    sl = make_loader(san, 1, False, 0)
    vsl = make_loader(va, 1, False, 0) if va is not None else None     # every validation case, for sanity
    model = OstiumModel(d=a.d, mode=a.mode, with_complex=cx, norm_tokens=bool(a.norm_tokens)).to(dev)
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M", flush=True)
    dec = Decoder(tr, dev, with_sensor=cx and a.w_rim > 0) if cx else None
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * max(1, math.ceil(len(tr) / a.batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.05)
    log = open(out / "log.jsonl", "a")
    step = 0
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        agg = {}
        for b in tl:
            b = b.to(dev)
            loss, L = train_losses(model, b, ys, xs, dec, a, cx, step)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            for k, v in L.items():
                agg.setdefault(k, []).append(float(v))
            agg.setdefault("loss", []).append(float(loss))
        rec = dict(epoch=ep, step=step, sec=round(time.time() - t0, 1), lr=sched.get_last_lr()[0],
                   **{k: float(np.mean(v)) for k, v in agg.items()})
        if ep % a.val_every == 0 or ep == a.epochs:
            # the same metrics on the held-out cases (val_) and on the fixed training cases (trn_),
            # so the generalisation gap is on the curves
            (out / "metrics").mkdir(exist_ok=True)
            if vl is not None:
                vm, rows = validate(model, vl, ys, xs, dec, a, dev)
                rec.update({f"val_{k}": v for k, v in vm.items()})
                (out / "metrics" / f"val_epoch{ep:04d}.json").write_text(json.dumps(rows, indent=1))
            tm, _ = validate(model, sl, ys, xs, dec, a, dev)
            rec.update({f"trn_{k}": v for k, v in tm.items()})
        if ep % a.sanity_every == 0 or ep == a.epochs:
            sanity(model, sl, ys, xs, dec, a, dev, out / "sanity" / f"epoch_{ep:04d}" / "train")
            if vsl is not None:
                sanity(model, vsl, ys, xs, dec, a, dev, out / "sanity" / f"epoch_{ep:04d}" / "val")
        log.write(json.dumps(rec) + "\n")
        log.flush()
        plot_curves(out)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
        if ep % a.save_every == 0 or ep == a.epochs or (a.stop_at and ep >= a.stop_at):
            torch.save(dict(model=model.state_dict(), args=vars(a), epoch=ep,
                            y_stats=(ys.mean.cpu(), ys.std.cpu()),
                            x_stats=((xs.mean.cpu(), xs.std.cpu()) if xs else None)), out / f"epoch_{ep:04d}.pth")
        if a.stop_at and ep >= a.stop_at:
            break


def sanity(model, loader, ys, xs, dec, a, dev, folder):
    """Fixed training cases, no augmentation: metrics (sampled at the true site / ostium) and one
    render per case (site probability, sampled rings vs true ring, sampled complex vs true)."""
    sys.path.insert(0, str(REPO / "scripts" / "ostium"))
    from visualize_predictions import render_sample
    folder.mkdir(parents=True, exist_ok=True)
    vm, rows = validate(model, loader, ys, xs, dec, a, dev)
    (folder / "metrics.json").write_text(json.dumps(dict(mean=vm, cases=rows), indent=1))
    for i, b in enumerate(loader):
        render_sample(model, b.to(dev), ys, xs, dec, folder / f"{i:02d}_{b.case[0][:40]}.png")


def plot_curves(out):
    """One panel per loss term (log scale) and, if present, the validation metrics; redrawn every
    epoch so training can be watched live."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [json.loads(l) for l in open(out / "log.jsonl")]
    keys = [k for k in ("loss", "site", "ost", "ring", "cx", "type", "geo", "rim") if k in rows[-1]]
    mkeys = [m for m in ("chain_centre_mm", "chain_ring_mm", "site_top1_mm", "site_mass_2mm", "centre_err_mm",
                         "ring_mean_mm", "ring_best_mm",
                         "complex_vertex_mm", "sensor_rim_mm", "type_ok")
             if any(f"val_{m}" in r or f"trn_{m}" in r for r in rows)]
    n = len(keys) + len(mkeys)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), squeeze=False)
    names = dict(loss="total", site="site (soft CE)", ost="ostium (flow/MSE)", ring="ostium ring points (raw; weight w_ring)",
                 cx="complex (flow/MSE)", type="aneurysm type (CE)", geo="geo aux (raw; weight w_geo)",
                 rim="rim aux, sensor (raw; weight w_rim)")
    for ax, k in zip(axes.ravel(), keys):
        ep = [r["epoch"] for r in rows if k in r]
        v = [r[k] for r in rows if k in r]
        ax.plot(ep, v, lw=1)
        if len(v) >= 5:
            w = max(3, len(v) // 20)
            ax.plot(ep[w - 1:], np.convolve(v, np.ones(w) / w, mode="valid"), lw=2, color="k")
        if min(v) > 0:
            ax.set_yscale("log")
        ax.set_title(names.get(k, k), fontsize=9)
        ax.set_xlabel("epoch", fontsize=8)
        ax.grid(alpha=0.3)
    for ax, m in zip(axes.ravel()[len(keys):], mkeys):
        for pre, lab, col in (("val_", "held-out (15)", "C3"), ("trn_", "training cases (4)", "C0")):
            pts = [(r["epoch"], r[pre + m]) for r in rows if pre + m in r]
            if pts:
                ax.plot(*zip(*pts), marker="o", ms=2, lw=1.2, color=col, label=lab)
        ax.set_title(m.replace("_", " ") + ("  (full chain: sampled site)" if m.startswith("chain")
                                            else "" if m.startswith("site") else "  (at the true site)"), fontsize=9)
        ax.set_xlabel("epoch", fontsize=8)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(f"{out.name}: epoch {rows[-1]['epoch']}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    main()
