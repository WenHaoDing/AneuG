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

--morph_cond 1: the morphology-conditioned variant (models/ostium_model.MorphOstiumModel), conditioned
on m = [log dome volume (mm3), aspect ratio] (models/dome_markers.py, labels from
scripts/ostium/morph_reference.py). The real m conditions the true-site terms, and the morph loss
compares it with the markers of the predicted complex there; the self-supervised terms (self rim,
cap_s, tube_s) generate at random targets drawn from the training cases' m (plus jitter), and
morph_s compares the generated dome's markers with the target. Validation uses each case's real m.
Every --morph_sanity_every epochs, morph_sanity generates the held-out and sanity cases for a fixed
grid of targets and checks the generated domes' markers against them (sanity_morph/, msan_ curves).

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
    p.add_argument("--w_selfrim", type=float, default=0.0,
                   help="self-consistency at sampled sites: sites drawn from the site head, the generated "
                        "dome's sensor rim against the generated ring (ring detached); in flow mode on the "
                        "one-step estimate from a random point of the model's own sampling path, weighted by t")
    p.add_argument("--w_cap", type=float, default=0.0,
                   help="cap-centre loss: each cap centre of the complex (the sensor's loc_probs weighting the "
                        "placed vertices) to its nearest 1D-graph point, squared, units of s. Order-free")
    p.add_argument("--w_tube", type=float, default=0.0,
                   help="vessel-surface loss: parent-vessel vertices of the complex (sensor: not dome, not cap) "
                        "to the graph's tube wall, min_i(|v - c_i| - r_i), squared, units of s")
    p.add_argument("--morph_cond", type=int, default=0,
                   help="1: condition on the dome morphology [log volume, aspect ratio] (MorphOstiumModel)")
    p.add_argument("--w_morph", type=float, default=0.1,
                   help="morph loss (standardised Huber on log volume and aspect ratio of the sensor's dome) at the "
                        "true site (morph, real m) and at the self-supervised sites (morph_s, random target m)")
    p.add_argument("--morph_jitter", type=float, default=0.25,
                   help="random target m: a training case's m plus Gaussian noise of this many condition stds")
    p.add_argument("--site_sampling", default="model", choices=["model", "uniform", "mix"],
                   help="where the self-supervised terms (self rim, cap_s, tube_s) generate: model = sites drawn "
                        "from the site head (the earlier runs); uniform = any graph point at least "
                        "--site_end_margin_mm from a graph end, uniformly; mix = uniform for a --uniform_frac share "
                        "of the graphs in each batch, the site head for the rest")
    p.add_argument("--uniform_frac", type=float, default=0.5)
    p.add_argument("--site_end_margin_mm", type=float, default=5.0)
    p.add_argument("--selfrim_every", type=int, default=1, help="compute the self-consistency loss every N steps")
    p.add_argument("--norm_tokens", type=int, default=1,
                   help="1: graph-token positions as [direction, log(1+distance)] and log(s) centred; "
                        "0: the raw relative positions (as in the first runs)")
    p.add_argument("--aux_every", type=int, default=1, help="compute the auxiliary losses every N steps")
    p.add_argument("--val_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=50)
    p.add_argument("--sanity_every", type=int, default=50,
                   help="render the fixed sanity cases (and their metrics) every N epochs")
    p.add_argument("--sanity_cases", type=int, default=4)
    p.add_argument("--morph_sanity_every", type=int, default=25,
                   help="--morph_cond 1: every N epochs, generate the held-out and sanity cases for a fixed grid of "
                        "target conditions and measure the generated domes' markers (morph_sanity; sanity_morph/)")
    p.add_argument("--overfit", type=int, default=0, help="train and validate on N training cases, no augmentation")
    p.add_argument("--val_cases", default=str(REPO / "dataset" / "ostium_val_cases.json"),
                   help="json list of held-out cases (split by patient); 'none' trains on all cases")
    p.add_argument("--holdout", action="store_true",
                   help="debug only: hold out ~15%% of patients and validate on them. By default the model "
                        "trains on ALL cases with no validation: the task is generative, so it is evaluated "
                        "with generative metrics, not a held-out reconstruction error")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--augment", type=int, default=1,
                   help="0: no augmentation at all (no trimming, rotation, sigma flips or site jitter)")
    p.add_argument("--rotate", type=int, default=1,
                   help="1: random SO(3) rotation per training sample; 0: keep the scan orientation "
                        "(trimming, sigma flips and site jitter unchanged)")
    p.add_argument("--rot_max_deg", type=float, default=180.0,
                   help="with --rotate 1: maximum rotation angle (random axis); 180 = any rotation")
    p.add_argument("--crop", type=int, default=1,
                   help="1: off-centre trimming of training graphs; 0: always the whole graph "
                        "(rotation, sigma flips and site jitter unchanged)")
    p.add_argument("--resume", default="",
                   help="checkpoint of this run to continue from (same --name, --epochs, --batch, --lr, --mode); "
                        "the log and outputs after its epoch are dropped")
    p.add_argument("--flow_h", type=int, default=256, help="width of the ostium / complex heads (VelocityMLP)")
    p.add_argument("--flow_blocks", type=int, default=4, help="residual blocks of the ostium / complex heads")
    p.add_argument("--sched", default="onecycle", choices=["onecycle", "hold_cos"],
                   help="onecycle: warm-up over the first 5%% then cosine decay to ~0 (the earlier runs); "
                        "hold_cos: linear warm-up over --warmup_frac, constant --lr until --hold_frac of the "
                        "run, then cosine decay to --final_lr_frac x --lr (more time at a useful rate, a "
                        "long late decay)")
    p.add_argument("--warmup_frac", type=float, default=0.03)
    p.add_argument("--hold_frac", type=float, default=0.4)
    p.add_argument("--final_lr_frac", type=float, default=0.01)
    p.add_argument("--init_from", default="",
                   help="checkpoint (any run, same --mode / --d / --norm_tokens) to start from: its weights and "
                        "target standardisation only; optimiser, schedule, epochs and log start fresh (a warm "
                        "restart). Ignored with --resume, whose checkpoint already carries them")
    p.add_argument("--stop_at", type=int, default=0,
                   help="stop after this epoch (0: run all --epochs); the schedule still follows --epochs")
    a = p.parse_args()
    if a.gpu == 2:
        sys.exit("GPU 2 must not be used (it overheats and crashes the workstation).")
    return a


# settings added after some checkpoints were made: their value in those checkpoints
DEFAULTS = dict(flow_h=256, flow_blocks=4, sched="onecycle", warmup_frac=0.03, hold_frac=0.4, final_lr_frac=0.01,
                w_cap=0.0, w_tube=0.0, site_sampling="model", uniform_frac=0.5, site_end_margin_mm=5.0,
                morph_cond=0, w_morph=0.1, morph_jitter=0.25)


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

    def faces(self, t):
        import torch
        if ("F", t) not in self.edges:
            rec = self.mc.get(int(t))
            m = rec.ghd_forward_as_Meshes(torch.zeros(1, 432, device=rec.GHD_eigvec.device), denormalize_shape=False)
            self.edges[("F", t)] = m.faces_packed().to(self.dev)
        return self.edges[("F", t)]

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


def sensor_rim(dec, phi_h, V_h, atype, caps=False):
    """Frozen sensor on decoded complexes of one type: per-vertex dome probability p (b, V), the
    p = 0.5 crossing point X on every mesh edge (b, E, 3, same frame as V_h), the soft crossing
    weight we (b, E) and the hard crossing mask (b, E). With caps=True also the sensor's cap
    distributions (b, n_open, V): one per opening of the type (the template gives only the count;
    which vertices form a cap is the sensor's call, GHD fitting keeps no correspondence)."""
    import torch
    # The frozen sensor has crashed 4 times with an out-of-range index (MLP runs, GPU 1), never
    # with CUDA_LAUNCH_BLOCKING=1, which points to a race; finish all GPU work before and after it.
    if phi_h.is_cuda:
        torch.cuda.synchronize()
    so = dec.sensor(phi_h, atype)
    p = torch.sigmoid(so[4]["dome_logits"])[:, :V_h.size(1)]
    if phi_h.is_cuda:
        torch.cuda.synchronize()
    e = dec.faces_edges(int(atype[0]))
    pi, pj = p[:, e[:, 0]], p[:, e[:, 1]]
    we = pi * (1 - pj) + pj * (1 - pi)
    den = torch.where((pj - pi).abs() < 1e-4, torch.full_like(pi, 1e-4), pj - pi)
    lam = ((0.5 - pi) / den).clamp(0, 1)
    X = V_h[:, e[:, 0]] + lam[..., None] * (V_h[:, e[:, 1]] - V_h[:, e[:, 0]])
    if not caps:
        return p, X, we, (pi - 0.5) * (pj - 0.5) < 0
    n_open = min(len(dec.mc._load_openings(int(atype[0]))), so[2].size(1))
    loc = so[2][:, :n_open, :V_h.size(1)]
    loc = loc / loc.sum(2, keepdim=True).clamp(min=1e-12)
    return p, X, we, (pi - 0.5) * (pj - 0.5) < 0, loc


def graph_fit(V, p, loc, gpos, grad, gend, dome_thr=0.2):
    """How well one placed complex sits on its 1D graph (all in the graph's frame and units).
    V (Nv, 3) placed vertices, p (Nv,) sensor dome probability, loc (k, Nv) sensor cap
    distributions; gpos (Ng, 3), grad (Ng,) graph points and radii, gend (Ng,) graph ends.
    Returns cap distances (k,) from each cap centre (loc-weighted vertices) to its nearest graph
    point, the tube distances (Nv,) min_i(|v - c_i| - r_i) (0 on the wall, < 0 inside), and the
    masks of the caps / vertices that count: parent-vessel vertices only (dome p < dome_thr, not
    in a cap region, by the sensor's own rule in models/morphoformer.region_from_probs), and
    nothing whose nearest graph point is an end (beyond a crop cut the graph is simply missing)."""
    import torch
    cen = loc @ V                                                           # (k, 3)
    dc, ic = torch.cdist(cen, gpos).min(1)
    cap_ok = ~gend[ic]
    sdf, iv = (torch.cdist(V, gpos) - grad[None]).min(1)
    nv = V.size(0)
    thr = torch.maximum(0.2 * loc.max(1, keepdim=True).values, torch.full_like(loc[:, :1], 3.0 / nv))
    in_cap = (loc >= thr).any(0)
    tube_ok = (p < dome_thr) & ~in_cap & ~gend[iv]
    return dc, cap_ok, sdf, tube_ok


def graph_terms(b, ids, V_loc, p, loc, C, Rf, s):
    """Cap and tube losses (units of s, squared; per sample) and their mm values (per sample, NaN
    where nothing counts) for complexes placed at centre C (b, 3), frame Rf (b, 3, 3), scale s (b, 1)
    in the graph's batch frame; ids are the samples' graph indices in the batch."""
    import torch
    V = C[:, None] + s[:, :, None] * torch.einsum("bnc,bcd->bnd", V_loc, Rf)
    cap_l, tube_l, cap_mm, tube_mm = [], [], [], []
    nan = V.new_tensor(float("nan"))
    for j, i in enumerate(ids.tolist()):
        m = b.batch == i
        dc, cok, sdf, tok = graph_fit(V[j], p[j], loc[j], b.pos[m], b.radius[m], b.x[m][:, -1] > 0.5)
        sj = s[j, 0]
        cap_l.append(((dc[cok] / sj) ** 2).mean() if cok.any() else V.new_zeros(()))
        tube_l.append(((sdf[tok] / sj) ** 2).mean() if tok.any() else V.new_zeros(()))
        cap_mm.append(dc[cok].mean().detach() * 10 if cok.any() else nan)
        tube_mm.append(sdf[tok].abs().mean().detach() * 10 if tok.any() else nan)
    return torch.stack(cap_l), torch.stack(tube_l), torch.stack(cap_mm), torch.stack(tube_mm)


def ring_loss(y_hat, y_true, tangent, t_w, n=64):
    """Ostium ring point loss: the predicted and true codes decoded to n ring points relative to the
    site, in units of s (centre offset + ring in the frame from n and the tangent). Both rings start
    on +e1 and run counter-clockwise, so point i matches point i. Mean squared distance, weighted by t_w."""
    import torch
    from models.ostium_model import frame_from_code, ring_from_code

    def pts(y):
        return y[:, None, :3] + torch.einsum("bnc,bcd->bnd", ring_from_code(y, 5, n), frame_from_code(y[:, :3], tangent))
    return (((pts(y_hat) - pts(y_true)) ** 2).sum(2).mean(1) * t_w).mean()


GUARD_LOG = None          # set in main: aux_guard.jsonl in the run folder


def aux_losses(b, x_hat, t_w, dec, a, extra=None, model=None):
    """geo + rim on the destandardised complex estimate x_hat (B, 26), weighted by t_w.
    Guard: a type group whose decoded shape or vertices are not finite skips the auxiliary terms
    for this step and is logged (GUARD_LOG) with its cases, instead of reaching the sensor.
    With extra (a dict) and --w_cap / --w_tube > 0, extra["cap"], extra["tube"] get the graph terms
    of the complex placed in the true ostium frame (same weighting by t_w). With a MorphOstiumModel,
    extra["morph"] gets the morph loss of the predicted dome against the case's real m."""
    import torch
    geo = x_hat.new_zeros(())
    rim = x_hat.new_zeros(())
    want_graph = extra is not None and (getattr(a, "w_cap", 0) > 0 or getattr(a, "w_tube", 0) > 0)
    if want_graph:
        extra["cap"], extra["tube"] = x_hat.new_zeros(()), x_hat.new_zeros(())
    want_morph = extra is not None and model is not None and is_morph(model)
    if want_morph:
        extra["morph"] = x_hat.new_zeros(())
    z, aff = x_hat[:, :16], x_hat[:, 16:]
    for t in b.atype.unique():
        m = b.atype == t
        phi_h, l_h = dec.phi_from_z(z[m], b.atype[m])
        V_h = local_vertices(dec.vcan(phi_h, t), l_h, aff[m])
        ok = bool(torch.isfinite(phi_h).all() and torch.isfinite(l_h).all() and torch.isfinite(V_h).all())
        if not ok:
            if GUARD_LOG is not None:
                cases = [c for c, k in zip(b.case, m.tolist()) if k]
                with open(GUARD_LOG, "a") as f:
                    f.write(json.dumps(dict(type=int(t), cases=cases,
                                            x_hat_absmax=float(x_hat[m].abs().max()),
                                            nonfinite_phi=int((~torch.isfinite(phi_h)).sum()),
                                            nonfinite_V=int((~torch.isfinite(V_h)).sum()))) + "\n")
            continue
        w = t_w[m]
        if a.w_geo > 0:
            V_t = local_vertices(dec.vcan(b.phi[m], t), b.log_scale[m], b.x_cx[m][:, 16:])
            geo = geo + ((V_h - V_t).norm(dim=2).mean(1) * w).sum() / len(b.atype)
        if (a.w_rim > 0 or want_graph or want_morph) and dec.sensor is not None:
            p_h, X, we, _, loc = sensor_rim(dec, phi_h, V_h, b.atype[m], caps=True)
            if want_morph:
                mh, ok = morph_markers(dec, V_h, p_h, t, b.scale[m])
                extra["morph"] = extra["morph"] + (morph_loss(model, mh, ok, b.morph[m]) * w).sum() / len(b.atype)
            if want_graph:
                ids = torch.nonzero(m).squeeze(1)
                cl, tl, _, _ = graph_terms(b, ids, V_h, p_h, loc, b.C[m], b.frame[m], b.scale[m])
                extra["cap"] = extra["cap"] + (cl * w).sum() / len(b.atype)
                extra["tube"] = extra["tube"] + (tl * w).sum() / len(b.atype)
            if a.w_rim <= 0:
                continue
            ring = b.ring_loc[m]
            d = torch.cdist(X, ring)                                  # (b, E, 64)
            fwd = (we * d.min(2).values ** 2).sum(1) / we.sum(1).clamp(min=1e-6)
            bwd = (d ** 2 + 0.25 * (1 - we)[..., None]).min(1).values.mean(1)
            rim = rim + ((fwd + bwd) * w).sum() / len(b.atype)
    return geo, rim


def rim_chamfer(X, we, ring):
    """Soft rim chamfer (squared, units of s^2) between the sensor's crossing points X (b, E, 3),
    weighted by we (b, E), and rings (b, n, 3). Per sample."""
    import torch
    d = torch.cdist(X, ring)
    fwd = (we * d.min(2).values ** 2).sum(1) / we.sum(1).clamp(min=1e-6)
    bwd = (d ** 2 + 0.25 * (1 - we)[..., None]).min(1).values.mean(1)
    return fwd + bwd


MORPH_POOL = None         # set in main with --morph_cond: the training cases' m (n, 2), raw
_DM = None


def is_morph(model):
    return hasattr(model, "morph_ost")


def morph_markers(dec, V_loc, p, t, s):
    """Dome markers of complexes of type t (local vertices V_loc in units of s, sensor p):
    m_hat = [log volume (mm3), aspect ratio] (b, 2) and the mask of domes that exist (a rim and a
    positive volume)."""
    import torch
    global _DM
    if _DM is None:
        from models.dome_markers import DomeMarkers
        _DM = DomeMarkers()
    o = _DM(V_loc, p, dec.faces(int(t)))
    vol = o["volume"] * (s[:, 0] * 10) ** 3
    ok = (o["rim_points"] > 0) & (vol > 0)
    return torch.stack([torch.log(vol.clamp(min=0.1)), o["aspect_ratio"].clamp(0, 20)], 1), ok


def morph_loss(model, m_hat, ok, target):
    """Per sample: Huber on the standardised [log volume, aspect ratio] error, 0 where no dome."""
    import torch
    e = torch.nn.functional.smooth_l1_loss(m_hat / model.morph_std, target / model.morph_std, reduction="none").sum(1)
    return torch.where(ok, e, torch.zeros_like(e))


def morph_errors(m_hat, m):
    """(volume error in %, absolute aspect-ratio error) of one dome against a condition, both raw."""
    return float(100 * abs(float((m_hat[0] - m[0]).exp()) - 1)), float((m_hat[1] - m[1]).abs())


def random_morph(n, a, model, device):
    """n random target conditions: training cases' m plus jitter (a.morph_jitter condition stds)."""
    import torch
    pool = MORPH_POOL.to(device)
    m = pool[torch.randint(len(pool), (n,), device=device)]
    return m + torch.randn_like(m) * model.morph_std * getattr(a, "morph_jitter", 0.25)


def generate_at_sites(model, enc, b, ys, xs, idx, tg, sc, steps=32, morph=None):
    """At one site per graph (idx, with its signed tangent tg and scale sc): the ostium code y
    (no gradient), then the type (sampled) and the complex x (with gradient), and a per-sample
    weight w. morph: requested conditions (MorphOstiumModel; default the batch's own). MLP mode: x is the prediction, w = 1. Flow mode: the complex is sampled without
    gradient, and x is the one-step estimate x_k + (1 - t_k) v(x_k, t_k) from a random step k of
    that sampling path (gradient through this one evaluation), w = t_k."""
    import torch
    mk = {} if morph is None else dict(morph=morph)
    with torch.no_grad():
        y = ys.inv(model.ost.sample(model.ostium_condition(enc, b, idx, tg, sc, **mk)))
    c2 = model.complex_condition(enc, b, ys.fwd(y), y, idx, tangent=tg, scale=sc, **mk)
    with torch.no_grad():
        t = torch.multinomial(torch.softmax(model.type_head(c2), 1), 1)[:, 0]
    c = model.with_type(c2, t)
    if model.cx.mode == "mlp":
        return y, t, xs.inv(model.cx(c)), torch.ones(len(t), device=c.device)
    n = len(c)
    k_at = torch.randint(0, steps, (n,), device=c.device)
    with torch.no_grad():
        xk = x = torch.randn(n, model.cx.x_dim, device=c.device)
        for k in range(steps):
            xk = torch.where((k_at == k)[:, None], x, xk)
            x = x + model.cx(c.detach(), x, torch.full((n,), k / steps, device=c.device)) / steps
    tk = k_at.float() / steps
    x1_hat = xk + (1 - tk[:, None]) * model.cx(c, xk, tk)
    return y, t, xs.inv(x1_hat), tk


def self_rim_loss(model, enc, b, ys, xs, dec, a=None, extra=None):
    """Self-consistency at sites drawn from the site head (detached): the sensor rim of the dome
    generated there against the ring generated there (detached), so the complex head learns to
    build its dome on the ring it is given, also away from the true site. No ground truth needed.
    With extra and --w_cap / --w_tube > 0, extra["cap_s"], extra["tube_s"] get the graph terms of the
    complexes generated there (placed in the generated ostium frame), from the same sensor call.
    MorphOstiumModel: generated at random target conditions, and extra["morph_s"] gets the morph
    loss of the generated domes against them."""
    import torch
    from models.ostium_model import frame_from_code, ring_from_code
    from torch_geometric.utils import softmax
    idx = self_sites(model, enc, b, a)
    tgt = random_morph(b.num_graphs, a, model, idx.device) if is_morph(model) else None
    y, t, x, w = generate_at_sites(model, enc, b, ys, xs, idx, b.tangent_pts[idx], b.scale_pts[idx], morph=tgt)
    if tgt is not None and extra is not None:
        extra["morph_s"] = x.new_zeros(())
    ring = ring_from_code(y, 5, 64).detach()
    loss = x.new_zeros(())
    want_graph = extra is not None and a is not None and (getattr(a, "w_cap", 0) > 0 or getattr(a, "w_tube", 0) > 0)
    if want_graph:
        extra["cap_s"], extra["tube_s"] = x.new_zeros(()), x.new_zeros(())
        tg, sc = b.tangent_pts[idx], b.scale_pts[idx]
        Rf_all = frame_from_code(y[:, :3], tg).detach()                   # the ostium is not trained here
        C_all = (b.pos[idx] + y[:, :3] * sc).detach()
    for tt in t.unique():
        m = t == tt
        phi_h, l_h = dec.phi_from_z(x[m][:, :16], t[m])
        V_h = local_vertices(dec.vcan(phi_h, tt), l_h, x[m][:, 16:])
        if not (torch.isfinite(phi_h).all() and torch.isfinite(V_h).all()):
            continue
        p_h, X, we, _, loc = sensor_rim(dec, phi_h, V_h, t[m], caps=True)
        loss = loss + (rim_chamfer(X, we, ring[m]) * w[m]).sum() / len(t)
        if tgt is not None and extra is not None:
            mh, ok = morph_markers(dec, V_h, p_h, tt, b.scale_pts[idx][m])
            extra["morph_s"] = extra["morph_s"] + (morph_loss(model, mh, ok, tgt[m]) * w[m]).sum() / len(t)
        if want_graph:
            ids = torch.nonzero(m).squeeze(1)
            cl, tl, _, _ = graph_terms(b, ids, V_h, p_h, loc, C_all[m], Rf_all[m], sc[m])
            extra["cap_s"] = extra["cap_s"] + (cl * w[m]).sum() / len(t)
            extra["tube_s"] = extra["tube_s"] + (tl * w[m]).sum() / len(t)
    return loss


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


def uniform_sites(b, margin_mm, fallback):
    """One site per graph, uniformly over its points at least margin_mm (straight line, so at least
    as far along the graph) from every graph end; a graph with no such point keeps fallback's site."""
    import torch
    out = fallback.clone()
    end = b.x[:, -1] > 0.5
    for i in range(b.num_graphs):
        m = torch.nonzero(b.batch == i).squeeze(1)
        ends = m[end[m]]
        ok = torch.ones(len(m), dtype=torch.bool, device=m.device)
        if len(ends):
            ok = torch.cdist(b.pos[m], b.pos[ends]).min(1).values * 10 >= margin_mm
        if ok.any():
            c = m[ok]
            out[i] = c[torch.randint(len(c), (1,), device=c.device)[0]]
    return out


def self_sites(model, enc, b, a):
    """Sites for the self-supervised terms, by --site_sampling (see get_args)."""
    import torch
    from torch_geometric.utils import softmax
    with torch.no_grad():
        idx = sample_sites(softmax(model.site_logits(enc, b), b.batch), b.batch, b.num_graphs)
    mode = getattr(a, "site_sampling", "model") if a is not None else "model"
    if mode == "model":
        return idx
    u = uniform_sites(b, getattr(a, "site_end_margin_mm", 5.0), idx)
    if mode == "uniform":
        return u
    pick = torch.rand(b.num_graphs, device=idx.device) < getattr(a, "uniform_frac", 0.5)
    return torch.where(pick, u, idx)


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
        w_cap, w_tube = getattr(a, "w_cap", 0), getattr(a, "w_tube", 0)
        if (a.w_geo > 0 or a.w_rim > 0 or w_cap > 0 or w_tube > 0 or is_morph(model)) and step % a.aux_every == 0:
            L["geo"], L["rim"] = aux_losses(b, xs.inv(x1h), tt, dec, a, extra=L, model=model)
        if getattr(a, "w_selfrim", 0) > 0 and step % a.selfrim_every == 0:
            L["selfrim"] = self_rim_loss(model, enc, b, ys, xs, dec, a, extra=L)
    loss = (a.w_site * L["site"] + a.w_ost * L["ost"] + a.w_cx * L.get("cx", 0) + a.w_type * L.get("type", 0)
            + a.w_geo * L.get("geo", 0) + a.w_rim * L.get("rim", 0) + a.w_ring * L.get("ring", 0)
            + getattr(a, "w_selfrim", 0) * L.get("selfrim", 0)
            + getattr(a, "w_cap", 0) * (L.get("cap", 0) + L.get("cap_s", 0))
            + getattr(a, "w_tube", 0) * (L.get("tube", 0) + L.get("tube_s", 0))
            + (getattr(a, "w_morph", 0) * (L.get("morph", 0) + L.get("morph_s", 0)) if is_morph(model) else 0))
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
            if xs is not None and hasattr(model, "cx") and dec.sensor is not None:
                # self-consistency at the last sampled sites: the generated dome's sensor rim (hard
                # p = 0.5 crossings) against the ring generated there, in mm
                from models.ostium_model import ring_from_code
                c2s = model.complex_condition(enc, b, ys.fwd(y), y, idx, tangent=tg, scale=sc)
                ts = torch.multinomial(torch.softmax(model.type_head(c2s), 1), 1)[:, 0]
                xg = xs.inv(model.cx.sample(model.with_type(c2s, ts)))
                rl = ring_from_code(y, 5, 64)
                for tt in ts.unique():
                    mm = ts == tt
                    ii = torch.nonzero(mm).squeeze(1)
                    phi_h, l_h = dec.phi_from_z(xg[mm][:, :16], ts[mm])
                    V_h = local_vertices(dec.vcan(phi_h, tt), l_h, xg[mm][:, 16:])
                    p_h, X, _, hard, loc = sensor_rim(dec, phi_h, V_h, ts[mm], caps=True)
                    for j, i in enumerate(ii.tolist()):
                        if hard[j].any():
                            d = torch.cdist(X[j][hard[j]], rl[i])
                            rows[-b.num_graphs + i]["chain_selfrim_mm"] = float(
                                0.5 * (d.min(1).values.mean() + d.min(0).values.mean()) * sc[i, 0] * 10)
                    # how the generated complex sits on the graph (cap centres, parent-vessel wall), mm
                    Rf_s, C_s = frame_from_code(y[mm][:, :3], tg[mm]), b.pos[idx[mm]] + y[mm][:, :3] * sc[mm]
                    _, _, cmm, tmm = graph_terms(b, ii, V_h, p_h, loc, C_s, Rf_s, sc[mm])
                    for j, i in enumerate(ii.tolist()):
                        for k_, v_ in (("chain_cap_graph_mm", cmm[j]), ("chain_tube_mm", tmm[j])):
                            if torch.isfinite(v_):
                                rows[-b.num_graphs + i][k_] = float(v_)
        if hasattr(b, "tangent_pts") and xs is not None and hasattr(model, "cx") and dec.sensor is not None:
            # at uniform sites (any point >= the margin from a graph end): what prompted generation does
            _uniform_site_metrics(model, enc, b, ys, xs, dec, a, rows)
        if xs is not None and hasattr(model, "cx"):
            y_std = ys.fwd(b.y_ost)
            c2 = model.complex_condition(enc, b, y_std, b.y_ost, b.site)
            tacc = (model.type_head(c2).argmax(1) == b.atype).float()
            verr, rerr = [], []
            gcap, gtube, macc = [], [], {}
            if dec.sensor is not None:
                # the floor: the true complex, decoded through the same decoder and placed in the true
                # ostium frame, against the graph
                gc = torch.full((b.num_graphs,), float("nan"), device=dev)
                gt_ = torch.full((b.num_graphs,), float("nan"), device=dev)
                for t in b.atype.unique():
                    mm = b.atype == t
                    idx = torch.nonzero(mm).squeeze(1)
                    phi_t, l_t = dec.phi_from_z(b.x_cx[mm][:, :16], b.atype[mm])
                    V_t = local_vertices(dec.vcan(phi_t, t), l_t, b.x_cx[mm][:, 16:])
                    p_t, _, _, _, loc_t = sensor_rim(dec, phi_t, V_t, b.atype[mm], caps=True)
                    _, _, gc[idx], gt_[idx] = graph_terms(b, idx, V_t, p_t, loc_t, b.C[mm], b.frame[mm], b.scale[mm])
                for i in range(b.num_graphs):
                    for k_, v_ in (("gt_cap_graph_mm", gc[i]), ("gt_tube_mm", gt_[i])):
                        if torch.isfinite(v_):
                            rows[-b.num_graphs + i][k_] = float(v_)
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
                        p_h, X, _, hard, loc = sensor_rim(dec, phi_h, V_h, b.atype[mm], caps=True)
                        for j, i in enumerate(idx.tolist()):
                            if hard[j].any():
                                d = torch.cdist(X[j][hard[j]], b.ring_loc[i])
                                re[i] = 0.5 * (d.min(1).values.mean() + d.min(0).values.mean()) * b.scale[i, 0] * 10
                        _, _, cmm, tmm = graph_terms(b, idx, V_h, p_h, loc, b.C[mm], b.frame[mm], b.scale[mm])
                        gcap.append((idx, cmm))
                        gtube.append((idx, tmm))
                        if is_morph(model):
                            # the dome generated at the true site for the case's real m, against that m
                            mh, ok = morph_markers(dec, V_h, p_h, t, b.scale[mm])
                            for j, i in enumerate(idx.tolist()):
                                if ok[j]:
                                    macc.setdefault(i, []).append(morph_errors(mh[j], b.morph[i]))
                verr.append(ve)
                rerr.append(re)
            verr, rerr = torch.stack(verr), torch.stack(rerr)
            for name, parts in (("cap_graph_mm", gcap), ("tube_mm", gtube)):
                acc = {}
                for idx, v in parts:
                    for i, x_ in zip(idx.tolist(), v.tolist()):
                        if np.isfinite(x_):
                            acc.setdefault(i, []).append(x_)
                for i, v in acc.items():
                    rows[-b.num_graphs + i][name] = float(np.mean(v))
            for i, v in macc.items():
                rows[-b.num_graphs + i].update(morph_vol_err_pct=float(np.mean([e[0] for e in v])),
                                               morph_ar_err=float(np.mean([e[1] for e in v])))
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


def _uniform_site_metrics(model, enc, b, ys, xs, dec, a, rows):
    """uni_selfrim_mm, uni_cap_graph_mm, uni_tube_mm: generated ring -> type -> complex at one uniform
    site per graph; its sensor rim against its ring, its caps and vessel wall against the graph.
    MorphOstiumModel: generated for a random target m, and uni_morph_vol_err_pct, uni_morph_ar_err
    against it."""
    import torch
    from models.ostium_model import frame_from_code, ring_from_code
    idx = uniform_sites(b, getattr(a, "site_end_margin_mm", 5.0), b.site)
    tg, sc = b.tangent_pts[idx], b.scale_pts[idx]
    tgt = random_morph(b.num_graphs, a, model, idx.device) if is_morph(model) else None
    mk = {} if tgt is None else dict(morph=tgt)
    y = ys.inv(model.ost.sample(model.ostium_condition(enc, b, idx, tg, sc, **mk)))
    c2 = model.complex_condition(enc, b, ys.fwd(y), y, idx, tangent=tg, scale=sc, **mk)
    ts = torch.multinomial(torch.softmax(model.type_head(c2), 1), 1)[:, 0]
    xg = xs.inv(model.cx.sample(model.with_type(c2, ts)))
    rl = ring_from_code(y, 5, 64)
    for tt in ts.unique():
        mm = ts == tt
        ii = torch.nonzero(mm).squeeze(1)
        phi_h, l_h = dec.phi_from_z(xg[mm][:, :16], ts[mm])
        V_h = local_vertices(dec.vcan(phi_h, tt), l_h, xg[mm][:, 16:])
        if not (torch.isfinite(phi_h).all() and torch.isfinite(V_h).all()):
            continue
        p_h, X, _, hard, loc = sensor_rim(dec, phi_h, V_h, ts[mm], caps=True)
        Rf_u, C_u = frame_from_code(y[mm][:, :3], tg[mm]), b.pos[idx[mm]] + y[mm][:, :3] * sc[mm]
        _, _, cmm, tmm = graph_terms(b, ii, V_h, p_h, loc, C_u, Rf_u, sc[mm])
        if tgt is not None:
            mh, ok = morph_markers(dec, V_h, p_h, tt, sc[mm])
        for j, i in enumerate(ii.tolist()):
            r = rows[-b.num_graphs + i]
            if tgt is not None and ok[j]:
                r["uni_morph_vol_err_pct"], r["uni_morph_ar_err"] = morph_errors(mh[j], tgt[i])
            if hard[j].any():
                d = torch.cdist(X[j][hard[j]], rl[i])
                r["uni_selfrim_mm"] = float(0.5 * (d.min(1).values.mean() + d.min(0).values.mean()) * sc[i, 0] * 10)
            for k_, v_ in (("uni_cap_graph_mm", cmm[j]), ("uni_tube_mm", tmm[j])):
                if torch.isfinite(v_):
                    r[k_] = float(v_)


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
    ck = None
    if a.resume:
        ck = torch.load(a.resume, map_location="cpu", weights_only=False)
        diff = {k: (ck["args"].get(k), getattr(a, k)) for k in ("epochs", "batch", "lr", "mode", "d", "norm_tokens",
                                                                "rotate", "crop", "augment", "w_ring", "val_cases", "rot_max_deg", "w_selfrim",
                                                                "flow_h", "flow_blocks", "sched", "warmup_frac", "hold_frac", "final_lr_frac",
                                                                "w_cap", "w_tube", "site_sampling", "uniform_frac",
                                                                "site_end_margin_mm", "morph_cond", "w_morph", "morph_jitter")
                if ck["args"].get(k, DEFAULTS.get(k, getattr(a, k))) != getattr(a, k)}
        if diff:
            sys.exit(f"--resume: settings differ from the checkpoint's (checkpoint, now): {diff}")
        keep = ck["epoch"]
        if (out / "log.jsonl").exists():                   # drop what came after the checkpoint
            rows = [l for l in open(out / "log.jsonl") if json.loads(l)["epoch"] <= keep]
            (out / "log.jsonl").write_text("".join(rows))
        import shutil
        for f in (out / "metrics").glob("val_epoch*.json") if (out / "metrics").exists() else []:
            if int(f.stem[9:]) > keep:
                f.unlink()
        for sd in ("sanity", "sanity_morph"):
            for d_ in (out / sd).glob("epoch_*") if (out / sd).exists() else []:
                if int(d_.name[6:]) > keep:
                    shutil.rmtree(d_)
        print(f"resuming from {a.resume} (epoch {keep})", flush=True)
    init = None
    if a.init_from and not a.resume:
        init = torch.load(a.init_from, map_location="cpu", weights_only=False)
        diff = {k: (init["args"].get(k), getattr(a, k)) for k in ("mode", "d", "norm_tokens", "flow_h", "flow_blocks", "morph_cond")
                if init["args"].get(k, DEFAULTS.get(k, getattr(a, k))) != getattr(a, k)}
        if diff:
            sys.exit(f"--init_from: model settings differ from the checkpoint's (checkpoint, now): {diff}")
        print(f"weights from {a.init_from} (epoch {init['epoch']}), fresh optimiser and schedule", flush=True)
    src = ck if ck is not None else init                    # where weights and stats come from, if anywhere
    (out / "args.json").write_text(json.dumps(vars(a), indent=1))
    cx = not a.no_complex
    mo = dict(morph=bool(a.morph_cond))
    if a.morph_cond:
        assert cx, "--morph_cond needs the complex model"
    if a.overfit:
        base = OstiumGraphDataset("train", augment=False, complex_targets=cx, seed=a.seed, **mo)
        cases = [c for _, c, _ in base.items[:a.overfit]]
        tr = OstiumGraphDataset("train", augment=False, point_frames=a.w_selfrim > 0, complex_targets=cx, cases=cases, seed=a.seed, **mo)
        va = OstiumGraphDataset("train", augment=False, complex_targets=cx, cases=cases, seed=a.seed, **mo)
    elif a.holdout:
        tr = OstiumGraphDataset("train", augment=bool(a.augment), rotate=bool(a.rotate), crop=bool(a.crop), rot_max_deg=a.rot_max_deg, point_frames=a.w_selfrim > 0, complex_targets=cx, seed=a.seed, **mo)
        va = OstiumGraphDataset("val", augment=False, complex_targets=cx, seed=a.seed, **mo)
    elif a.val_cases.lower() != "none":
        vnames = [r["case"] for r in json.load(open(a.val_cases))["cases"]]
        tr = OstiumGraphDataset("train", augment=bool(a.augment), rotate=bool(a.rotate), crop=bool(a.crop), rot_max_deg=a.rot_max_deg, point_frames=a.w_selfrim > 0, complex_targets=cx, seed=a.seed, val_cases=vnames, **mo)
        va = OstiumGraphDataset("val", augment=False, complex_targets=cx, seed=a.seed, val_cases=vnames,
                                point_frames=True, **mo)
    else:
        # augment explicitly: the dataset's default is augment = (split == "train"), which for
        # split "all" silently turned trimming, rotation, sigma flips and site jitter off
        tr = OstiumGraphDataset("all", augment=bool(a.augment), rotate=bool(a.rotate), crop=bool(a.crop), rot_max_deg=a.rot_max_deg, point_frames=a.w_selfrim > 0, complex_targets=cx, seed=a.seed, **mo)
        va = None
    print(f"train {len(tr)} cases, {'val ' + str(len(va)) if va is not None else 'no validation'}, on {dev}", flush=True)
    if src is not None:                                     # the checkpoint's standardisation, not new stats
        ys = Stats(torch.zeros(2, 33))
        ys.mean, ys.std = src["y_stats"]
        xs = None
        if src.get("x_stats") is not None:
            xs = Stats(torch.zeros(2, 26))
            xs.mean, xs.std = src["x_stats"]
    else:
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
    san = OstiumGraphDataset("all", augment=False, complex_targets=cx, cases=pick, seed=a.seed, point_frames=True, **mo)
    sl = make_loader(san, 1, False, 0)
    vsl = make_loader(va, 1, False, 0) if va is not None else None     # every validation case, for sanity
    kw = dict(d=a.d, mode=a.mode, with_complex=cx, norm_tokens=bool(a.norm_tokens), flow_h=a.flow_h, flow_blocks=a.flow_blocks)
    if a.morph_cond:
        from models.ostium_model import MorphOstiumModel
        global MORPH_POOL
        ref = json.loads(Path(REPO / "dataset" / "ostium_morph_reference.json").read_text())
        names = {(d_, c) for d_, c, _ in tr.items}
        MORPH_POOL = torch.tensor([tr.morph[k] for k in sorted(names)], dtype=torch.float32)   # training cases only
        model = MorphOstiumModel(morph_mean=ref["mean"], morph_std=ref["std"], **kw).to(dev)
        print(f"morphology conditioned: {len(MORPH_POOL)} training conditions, mean {ref['mean']}, std {ref['std']}", flush=True)
    else:
        model = OstiumModel(**kw).to(dev)
    if src is not None:
        model.load_state_dict(src["model"])
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M", flush=True)
    dec = Decoder(tr, dev, with_sensor=cx and (a.w_rim > 0 or bool(a.morph_cond))) if cx else None
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * max(1, math.ceil(len(tr) / a.batch))
    if a.sched == "hold_cos":
        w_s, h_s = max(1, int(a.warmup_frac * steps)), int(a.hold_frac * steps)

        def lr_factor(i):
            if i < w_s:
                return 1 / 25 + (1 - 1 / 25) * i / w_s
            if i < h_s:
                return 1.0
            u = min(1.0, (i - h_s) / max(1, steps - h_s))
            return a.final_lr_frac + (1 - a.final_lr_frac) * 0.5 * (1 + math.cos(math.pi * u))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    else:
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.05)
    start_ep, step = 1, 0
    if ck is not None:
        start_ep = ck["epoch"] + 1
        step = ck["epoch"] * max(1, math.ceil(len(tr) / a.batch))
        if "opt" in ck:                                     # checkpoints from now on carry both
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
        else:                                               # older ones: AdamW moments restart from zero,
            import warnings                                 # the schedule is put back where it was
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for _ in range(step):
                    sched.step()
        print(f"schedule at step {step}, lr {sched.get_last_lr()[0]:.2e}", flush=True)
    global GUARD_LOG
    GUARD_LOG = out / "aux_guard.jsonl"
    log = open(out / "log.jsonl", "a")
    t0 = time.time()
    for ep in range(start_ep, a.epochs + 1):
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
        if is_morph(model) and (ep % a.morph_sanity_every == 0 or ep == a.epochs):
            for pre, ld, sub in (("val_", vsl, "val"), ("trn_", sl, "train")):
                if ld is not None:
                    ms = morph_sanity(model, ld, ys, xs, dec, a, dev, out / "sanity_morph" / f"epoch_{ep:04d}" / sub)
                    rec.update({f"{pre}msan_{k}": v for k, v in ms.items() if k != "n"})
        log.write(json.dumps(rec) + "\n")
        log.flush()
        plot_curves(out)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
        if ep % a.save_every == 0 or ep == a.epochs or (a.stop_at and ep >= a.stop_at):
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), args=vars(a), epoch=ep,
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


MSAN_VOL_Q = (0.1, 0.3, 0.5, 0.7, 0.9)        # morph sanity: target grid, training-pool quantiles of log volume
MSAN_AR_Q = (0.15, 0.5, 0.85)                 # and of aspect ratio (all 15 combinations)


def morph_sanity(model, loader, ys, xs, dec, a, dev, folder, n_render=3):
    """MorphOstiumModel: does a generated dome have the requested markers? Every case of the loader
    is generated for a fixed grid of target conditions (MSAN_VOL_Q x MSAN_AR_Q of the training
    conditions), the whole chain sampled (ostium, type, complex), at the true site and at one
    uniform site (fixed per case); the generated dome's markers (the training ones: the sensor's
    p = 0.5 cut, models/dome_markers.py) against the target. Same seed every time, so epochs differ
    only by the model; the training random state is untouched. Writes metrics.json (every sample,
    and per site: errors, the slope / R^2 of achieved vs target, 1 = obeys, 0 = ignores),
    scatter.png and sweep.png (domes of the first n_render cases at the uniform site: the volume
    targets at the middle AR, the AR targets at the middle volume). Returns the uniform-site summary."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from dataset.ostium_dataset import FOLLOW_BATCH
    from torch_geometric.data import Batch
    folder.mkdir(parents=True, exist_ok=True)
    pool = MORPH_POOL.numpy()
    lv, ar = np.quantile(pool[:, 0], MSAN_VOL_Q), np.quantile(pool[:, 1], MSAN_AR_Q)
    grid = np.array([[v, r] for v in lv for r in ar], np.float32)
    tgt = torch.tensor(grid, device=dev)
    K = len(grid)
    rows, keep = [], {}
    was_training = model.training
    model.eval()
    with torch.no_grad(), torch.random.fork_rng(devices=[dev]):
        torch.manual_seed(0)
        for ci in range(len(loader.dataset)):
            d0 = loader.dataset[ci]
            b1 = Batch.from_data_list([d0], follow_batch=FOLLOW_BATCH).to(dev)
            B = Batch.from_data_list([d0] * K, follow_batch=FOLLOW_BATCH).to(dev)
            enc = model.enc(B)
            i_uni = uniform_sites(b1, getattr(a, "site_end_margin_mm", 5.0), b1.site)
            for site in ("true", "uniform"):
                if site == "true":
                    idx, tg, sc, s_m = B.site, None, None, B.scale
                else:
                    idx = B.ptr[:-1] + (i_uni[0] - b1.ptr[0])
                    tg, sc = B.tangent_pts[idx], B.scale_pts[idx]
                    s_m = sc
                y = ys.inv(model.ost.sample(model.ostium_condition(enc, B, idx, tg, sc, morph=tgt)))
                c2 = model.complex_condition(enc, B, ys.fwd(y), y, idx, tangent=tg, scale=sc, morph=tgt)
                ts = torch.multinomial(torch.softmax(model.type_head(c2), 1), 1)[:, 0]
                xg = xs.inv(model.cx.sample(model.with_type(c2, ts)))
                for tt in ts.unique():
                    mm = ts == tt
                    ii = torch.nonzero(mm).squeeze(1).tolist()
                    phi_h, l_h = dec.phi_from_z(xg[mm][:, :16], ts[mm])
                    V_h = local_vertices(dec.vcan(phi_h, tt), l_h, xg[mm][:, 16:])
                    if not (torch.isfinite(phi_h).all() and torch.isfinite(V_h).all()):
                        continue
                    p_h, *_ = sensor_rim(dec, phi_h, V_h, ts[mm])
                    mh, ok = morph_markers(dec, V_h, p_h, tt, s_m[mm])
                    for j, k in enumerate(ii):
                        r = dict(case=d0.case, site=site, target=k, type=int(tt), dome=bool(ok[j]),
                                 target_volume_mm3=float(np.exp(grid[k, 0])), target_ar=float(grid[k, 1]))
                        if ok[j]:
                            r.update(volume_mm3=float(mh[j, 0].exp()), ar=float(mh[j, 1]))
                            r["vol_err_pct"], r["ar_err"] = morph_errors(mh[j], tgt[k])
                        rows.append(r)
                        if site == "uniform" and ci < n_render:
                            mmu = float(s_m[k, 0]) * 10
                            keep[(ci, k)] = (V_h[j].cpu().numpy() * mmu, p_h[j].cpu().numpy(), r)
    model.train(was_training)

    def summary(rs):
        ok = [r for r in rs if r["dome"]]
        out = dict(n=len(rs), dome_found=len(ok) / max(1, len(rs)))
        if len(ok) < 3:
            return out
        ve, ae = np.array([r["vol_err_pct"] for r in ok]), np.array([r["ar_err"] for r in ok])
        tv, gv = np.log([r["target_volume_mm3"] for r in ok]), np.log([r["volume_mm3"] for r in ok])
        ta, ga = np.array([r["target_ar"] for r in ok]), np.array([r["ar"] for r in ok])
        for k, (t_, g_) in (("logv", (tv, gv)), ("ar", (ta, ga))):
            out[f"{k}_slope"] = float(np.polyfit(t_, g_, 1)[0])
            out[f"{k}_r2"] = float(np.corrcoef(t_, g_)[0, 1] ** 2)
        out.update(vol_err_med=float(np.median(ve)), vol_err_p90=float(np.percentile(ve, 90)),
                   vol_within_25pct=float((ve <= 25).mean()), ar_err_med=float(np.median(ae)),
                   ar_err_p90=float(np.percentile(ae, 90)), ar_within_0p3=float((ae <= 0.3).mean()))
        return out
    summ = {s: summary([r for r in rows if r["site"] == s]) for s in ("true", "uniform")}
    (folder / "metrics.json").write_text(json.dumps(dict(
        grid=dict(log_volume_q=MSAN_VOL_Q, ar_q=MSAN_AR_Q, volume_mm3=np.exp(lv).tolist(), ar=ar.tolist()),
        summary=summ, samples=rows), indent=1))

    fig, ax = plt.subplots(2, 3, figsize=(15, 9.5))
    for r_, s in enumerate(("true", "uniform")):
        ok = [r for r in rows if r["site"] == s and r["dome"]]
        if not ok:
            continue
        jit = lambda n: np.exp(np.random.default_rng(0).uniform(-0.04, 0.04, n))
        tv, gv = np.array([r["target_volume_mm3"] for r in ok]), np.array([r["volume_mm3"] for r in ok])
        ta, ga = np.array([r["target_ar"] for r in ok]), np.array([r["ar"] for r in ok])
        lev = np.array([r["target"] % len(MSAN_AR_Q) for r in ok])                 # the grid is volume-major
        S = summ[s]
        a0 = ax[r_, 0]
        for li in range(len(MSAN_AR_Q)):
            q = lev == li
            a0.scatter(tv[q] * jit(q.sum()), gv[q], s=8, alpha=0.6, label=f"target AR {ar[li]:.2f}")
        lim = [min(tv.min(), gv.min()) * 0.7, max(tv.max(), gv.max()) * 1.4]
        a0.plot(lim, lim, "k-", lw=1)
        a0.fill_between(lim, [l / 1.25 for l in lim], [l * 1.25 for l in lim], color="k", alpha=0.08)
        a0.set_xscale("log"), a0.set_yscale("log"), a0.set_xlim(lim), a0.set_ylim(lim)
        a0.set_xlabel("target volume (mm3)"), a0.set_ylabel("generated dome volume (mm3)")
        a0.set_title(f"{s} site: volume (band +-25%)  slope {S.get('logv_slope', np.nan):.2f}  "
                     f"R2 {S.get('logv_r2', np.nan):.2f}", fontsize=9)
        a0.legend(fontsize=7)
        a1 = ax[r_, 1]
        a1.scatter(ta + np.random.default_rng(1).uniform(-0.04, 0.04, len(ta)), ga, s=8, alpha=0.6, c=np.log(tv), cmap="viridis")
        lim = [min(ta.min(), ga.min()) - 0.2, max(ta.max(), ga.max()) + 0.2]
        a1.plot(lim, lim, "k-", lw=1)
        a1.fill_between(lim, [l - 0.3 for l in lim], [l + 0.3 for l in lim], color="k", alpha=0.08)
        a1.set_xlim(lim), a1.set_ylim(lim)
        a1.set_xlabel("target aspect ratio (height / neck radius)"), a1.set_ylabel("generated")
        a1.set_title(f"{s} site: AR (band +-0.3, colour = target log volume)  slope {S.get('ar_slope', np.nan):.2f}  "
                     f"R2 {S.get('ar_r2', np.nan):.2f}", fontsize=9)
        a2 = ax[r_, 2]
        ve = np.array([r["vol_err_pct"] for r in ok])
        vl_ = sorted(set(tv.round(3)))
        a2.boxplot([ve[tv.round(3) == v] for v in vl_], positions=range(len(vl_)), widths=0.5)
        a2.set_xticks(range(len(vl_)), [f"{v:.0f}" for v in vl_])
        a2.axhline(25, color="k", lw=0.8, ls="--")
        a2.set_yscale("symlog", linthresh=10)
        a2.set_xlabel("target volume (mm3)"), a2.set_ylabel("|volume error| (%)")
        a2.set_title(f"{s} site: median {S.get('vol_err_med', np.nan):.0f}%, p90 {S.get('vol_err_p90', np.nan):.0f}%, "
                     f"<=25% {S.get('vol_within_25pct', np.nan):.0%}; |AR err| median {S.get('ar_err_med', np.nan):.2f}; "
                     f"dome {S['dome_found']:.0%}", fontsize=8)
        for a_ in ax[r_]:
            a_.grid(alpha=0.3)
    fig.suptitle(f"{folder}: requested vs generated dome markers ({len(loader.dataset)} cases x {K} targets x 2 sites; "
                 "VAE floor: volume median 5%, p90 31%)", fontsize=9)
    fig.tight_layout()
    fig.savefig(folder / "scatter.png", dpi=100)
    plt.close(fig)

    cols = [(vi * len(MSAN_AR_Q) + 1, f"V {np.exp(lv[vi]):.0f}") for vi in range(len(MSAN_VOL_Q))] + \
           [(2 * len(MSAN_AR_Q) + ri, f"AR {ar[ri]:.2f}") for ri in range(len(MSAN_AR_Q))]
    cases = sorted({c for c, _ in keep})
    if cases:
        fig, ax = plt.subplots(len(cases), len(cols), figsize=(2.6 * len(cols), 3.1 * len(cases)), squeeze=False)
        for ri, ci in enumerate(cases):
            pts = [keep[(ci, k)] for k, _ in cols if (ci, k) in keep]
            span = max(np.ptp(V, 0).max() for V, _, _ in pts) * 0.55 if pts else 1
            for cj, (k, lab) in enumerate(cols):
                a_ = ax[ri, cj]
                a_.set_xticks([]), a_.set_yticks([])
                if (ci, k) not in keep:
                    a_.set_title(f"{lab}: not generated", fontsize=7)
                    continue
                V, p, r = keep[(ci, k)]
                dome = p > 0.5
                if dome.any() and (~dome).any():
                    o = V[dome].mean(0) - V[~dome].mean(0)
                    n = o / max(np.linalg.norm(o), 1e-9)               # side view: dome direction up
                else:
                    n = np.array([0, 0, 1.0])
                u = np.cross(n, [1, 0, 0] if abs(n[0]) < 0.9 else [0, 1, 0])
                u /= np.linalg.norm(u)
                c0 = V.mean(0)
                X, Y = (V - c0) @ u, (V - c0) @ n
                a_.scatter(X[~dome], Y[~dome], s=0.3, c="0.6")
                a_.scatter(X[dome], Y[dome], s=0.5, c="crimson")
                a_.set_xlim(-span, span), a_.set_ylim(-span, span), a_.set_aspect("equal")
                got = f"{r['volume_mm3']:.0f} mm3, AR {r['ar']:.2f}" if r["dome"] else "no dome"
                a_.set_title(f"want {lab}\ngot {got}  (type {r['type']})", fontsize=7)
            ax[ri, 0].set_ylabel(str(loader.dataset[ci].case)[:28], fontsize=7)
        fig.suptitle("uniform site, same seed: volume targets at the middle AR (left), AR targets at the middle volume "
                     "(right); red = sensor dome (p > 0.5), dome direction up, mm, same scale per row", fontsize=9)
        fig.tight_layout()
        fig.savefig(folder / "sweep.png", dpi=100)
        plt.close(fig)
    torch.cuda.empty_cache()
    return summ["uniform"]


def plot_curves(out):
    """One panel per loss term (log scale) and, if present, the validation metrics; redrawn every
    epoch so training can be watched live."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [json.loads(l) for l in open(out / "log.jsonl")]
    keys = [k for k in ("loss", "site", "ost", "ring", "cx", "type", "geo", "rim", "selfrim", "cap", "tube", "cap_s", "tube_s",
                        "morph", "morph_s")
            if k in rows[-1]]
    mkeys = [m for m in ("chain_centre_mm", "chain_ring_mm", "chain_selfrim_mm", "site_top1_mm", "site_mass_2mm", "centre_err_mm",
                         "ring_mean_mm", "ring_best_mm",
                         "complex_vertex_mm", "sensor_rim_mm", "type_ok", "cap_graph_mm", "tube_mm", "chain_cap_graph_mm",
                         "chain_tube_mm", "gt_cap_graph_mm", "gt_tube_mm", "uni_selfrim_mm", "uni_cap_graph_mm",
                         "uni_tube_mm", "morph_vol_err_pct", "morph_ar_err", "uni_morph_vol_err_pct", "uni_morph_ar_err",
                         "msan_vol_err_med", "msan_vol_within_25pct", "msan_logv_slope", "msan_ar_err_med",
                         "msan_ar_within_0p3", "msan_ar_slope", "msan_dome_found")
             if any(f"val_{m}" in r or f"trn_{m}" in r for r in rows)]
    n = len(keys) + len(mkeys)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), squeeze=False)
    names = dict(loss="total", site="site (soft CE)", ost="ostium (flow/MSE)", ring="ostium ring points (raw; weight w_ring)",
                 cx="complex (flow/MSE)", type="aneurysm type (CE)", geo="geo aux (raw; weight w_geo)",
                 rim="rim aux, sensor (raw; weight w_rim)", selfrim="self rim at sampled sites (raw; weight w_selfrim)",
                 cap="cap centres to graph, true site (raw; weight w_cap)",
                 tube="vessel wall to graph tube, true site (raw; weight w_tube)",
                 cap_s="cap centres to graph, sampled sites (raw; weight w_cap)",
                 tube_s="vessel wall to graph tube, sampled sites (raw; weight w_tube)",
                 morph="dome volume + AR vs real m, true site (raw; weight w_morph)",
                 morph_s="dome volume + AR vs random target m, sampled sites (raw; weight w_morph)")
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
                                            else "  (target grid, uniform site)" if m.startswith("msan")
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
