"""
Train the conditional MLP-VAE over per-branch Fourier coefficients
(models.branch_mlp_vae.BranchFourierVAE) on VesselSkeletonDatasetFourier.

Adapted from scripts/train/train_branch_transformer.py.

conda activate new
python scripts/train/train_branch_mlp_vae.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]


def _parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--gan", action="store_true",
                   help="Add a WGAN-GP critic over generated branch CENTERLINES. The critic "
                        "never sees the merged mesh: fusion is non-differentiable, and "
                        "everything this model controls is already in the curve.")
    p.add_argument("--adv-weight", type=float, default=0.1,
                   help="Adversarial pressure as a FRACTION of reconstruction pressure. The "
                        "raw adversarial gradient is first rescaled to match the "
                        "reconstruction gradient at the decoder's output layer, so this is a "
                        "readable ratio rather than a number found by trial and error.")
    p.add_argument("--n-critic", type=int, default=5)
    p.add_argument("--gp-weight", type=float, default=10.0)
    p.add_argument("--critic-hidden", type=int, default=128)
    p.add_argument("--critic-lr", type=float, default=1e-4)
    p.add_argument("--adv-warmup", type=int, default=300,
                   help="Epochs of pure VAE before the critic is consulted. A critic scoring "
                        "an untrained decoder gives a large, uninformative gradient.")
    p.add_argument("--kl-mult", type=int, default=None,
                   help="KL weight multiplier over KL_BASE. Raise it only while reconstruction "
                        "is unaffected: at 4 the rotation runs plateaued at coeff 0.49 train / "
                        "0.66 val, and whether KL was the binding constraint has never been "
                        "tested with rotation on.")
    p.add_argument("--syn-dir-weight", type=float, default=None,
                   help="Override SYN_DIR_WEIGHT. 0 disables the tangent constraint on "
                        "synthetic shapes entirely, which is the A/B for whether that "
                        "constraint is earning its place.")
    p.add_argument("--no-rotation", dest="rotation", action="store_false", default=True,
                   help="Disable random SO(3) augmentation. The A/B against the default is "
                        "what shows whether it helps: with 471 branches it should curb "
                        "memorisation, but it also makes reconstruction genuinely harder.")
    p.add_argument("--device", default=None)
    p.add_argument("--epochs", type=int, default=None)
    return p.parse_args() if __name__ == "__main__" else p.parse_args([])


_args = _parse_args()
sys.path = [path for path in sys.path if Path(path or ".").resolve() != ROOT]
sys.path.insert(0, str(ROOT))

from dataset.skeleton_dataset_fourier import VesselSkeletonDatasetFourier, reconstruct_branch
from models.branch_transformer import MultiBranchConditions
from models.branch_mlp_vae import BranchFourierVAE, BranchFourierVAE_GCNConditioner
from dataset.synthetic_shape_dataset import SyntheticShapeDataset

# ── variant ───────────────────────────────────────────────────────────────────
USE_GCN        = True   # True → GCN mesh encoder for phi;  False → GHDTokenEncoder (simple MLP)

PROCESSED_ROOT = ROOT / "runtime_dataset" / "AneuG_processed"
CANONICAL_ROOT = ROOT / "dataset" / "canonical"   # only used when USE_GCN=True

DEVICE     = torch.device(_args.device or ("cuda:1" if torch.cuda.is_available() else "cpu"))
EPOCHS     = _args.epochs if _args.epochs is not None else 1500
BATCH_SIZE = 64
LR         = 1e-3
LR_MIN     = 1e-5

K            = 8           # Fourier harmonics per coordinate (must match the dataset)
HIDDEN_DIM   = 512
LATENT_DIM   = 8
NUM_LAYERS   = 4           # MLP depth in encoder / decoder
COND_DIM     = 64
GHD_DIM      = 16
MAX_BRANCHES = 3
NUM_TYPES    = 3

# GCN encoder (only used when USE_GCN=True)
GCN_HIDDEN     = 32
GCN_POOL_RATIO = 0.5

KL_BASE      = 1e-3
KL_MULT      = _args.kl_mult if _args.kl_mult is not None else 4
# Raised from 2. At 24 nats over an 8-d latent the posterior sat far from the
# prior, which is the exact condition that gave every stage-1 run high precision
# and poor recall. More KL was the single biggest lever there, optimum 2-4.
KL_WEIGHT    = KL_BASE * KL_MULT
KL_ANNEAL_EPOCHS = 300
VEC_WEIGHT      = 1.0      # weight of the chord (branch_vector) loss vs coeff loss
PRESENCE_WEIGHT = 1.0      # weight of the per-branch presence (BCE) loss

# Candidate openings per aneurysm type (matches the canonical openings.npz counts
# and compute_branch_conditions): bifurcated has 3 openings, sidewall has 2.
TYPE_N_OPEN = {0: 3, 1: 2, 2: 2}
CANONICAL_TYPE_NAME = {0: "bifurcated", 1: "sidewall", 2: "sidewall"}


def opening_mask_from_types(aneurysm_type, max_branches):
    """[B, max_branches] bool — which slots are candidate openings for each type."""
    n = torch.tensor([TYPE_N_OPEN.get(int(t), max_branches) for t in aneurysm_type.tolist()])
    return torch.arange(max_branches).unsqueeze(0) < n.unsqueeze(1)

_VARIANT = "branch_mlp_vae_gcn" if USE_GCN else "branch_mlp_vae"
# _args, not SYN_DIR_WEIGHT: the tag is built before that constant exists, and
# referring to it here is a NameError at import.
_SYN_W = _args.syn_dir_weight if _args.syn_dir_weight is not None else 0.5
# The weight is SPELLED OUT, not reduced to a nosyn flag. The flag only
# distinguished zero from non-zero, so a sweep over 0 / 0.5 / 1.0 would have put
# the last two runs in the same directory and had them overwrite each other.
_gan_tag = ((f"_gan_adv{_args.adv_weight:g}" if _args.gan else "")
            + ("" if _args.rotation else "_norot")
            + f"_syn{_SYN_W:g}")
SAVE_DIR = (ROOT / "runtime_train" / "branch_mlp_vae"
            / f"{_VARIANT}_h{HIDDEN_DIM}_l{NUM_LAYERS}_z{LATENT_DIM}_kl{KL_MULT}{_gan_tag}")
# Synthetic shapes, PRE-COMPUTED. This run loads neither the stage-1 GHD VAE
# nor the morphology sensor: both are frozen and neither depends on the stage-2
# weights, so their whole contribution -- phi, the cap regions, the tangent at
# each opening -- is drawn once by dataset/synthetic_shape_dataset.py and read
# back here as arrays. That removes two networks from the GPU and an encoder
# forward plus a boundary walk from every step that uses synthetic shapes.
#
# The pool's provenance (which generator, which sensor, the acceptance rate)
# lives in its manifest.json and is copied into this run's hparams.
SYNTHETIC_POOL = ROOT / "runtime_dataset" / "synthetic_shapes" / "default"

# Synthetic-direction loss: take pre-computed phi and its mesh openings (start +
# outward direction) from the pool, generate Fourier branches off that
# condition, and constrain each branch's *initial tangent* (analytic from
# branch_vector + coeffs) to match the opening's outward direction.
SYN_DIR_WEIGHT = _SYN_W
B_SYN          = 24       # synthetic batch size; must be divisible by NUM_TYPES

SAVE_EVERY   = 500
LOG_EVERY    = 10
SANITY_EVERY = 500
SANITY_SYNTHETIC = True    # also render fully-generated branches from synthetic GHD phi
VAL_FRACTION = 0.1
ROTATION_AUGMENT = _args.rotation   # random SO(3) on mesh + conditions + targets
VAL_SEED     = 42

WANDB_PROJECT = "AneuGv2_stage2"
WANDB_RUN_NAME = SAVE_DIR.name


def compute_target_stats(dataset):
    """Per-(harmonic, coord) coeff stats and per-coord chord stats from the fit cache."""
    coeffs, vecs = [], []
    for sample_fits in dataset.fits:
        for bf in sample_fits:
            if bf is not None:
                coeffs.append(bf["coeffs"])
                vecs.append(bf["end"] - bf["start"])
    coeffs = np.stack(coeffs, axis=0)                       # [B, k, 3]
    vecs   = np.stack(vecs, axis=0)                         # [B, 3]
    return (
        torch.as_tensor(coeffs.mean(0), dtype=torch.float32),
        torch.as_tensor(coeffs.std(0) + 1e-4, dtype=torch.float32),
        torch.as_tensor(vecs.mean(0), dtype=torch.float32),
        torch.as_tensor(vecs.std(0) + 1e-4, dtype=torch.float32),
    )


def make_conditions(batch):
    return MultiBranchConditions(
        aneurysm_type    = batch["aneurysm_type"].to(DEVICE),
        scale            = batch["scale"].to(DEVICE),
        start_points     = batch["start_points"].to(DEVICE),
        branch_direction = batch["branch_direction"].to(DEVICE),
        branch_mask      = batch["branch_mask"].to(DEVICE),
    )


def rotate_batch(batch, device):
    """Random SO(3) per sample, applied to the whole condition/target system.

    The map from opening geometry to branch shape is rotation-equivariant, so
    this is a true inductive bias rather than noise: turning the mesh, the
    opening's start and direction, the branch vector and the Fourier
    coefficients together leaves the relationship unchanged. With 471 training
    branches that is a lot of free data.

    The coefficients rotate as plain 3-vectors, because the curve is
        p(s) = start + vec*(s/pi) + sum_n coeffs[n]*sin(n*s)
    and every term is linear in a 3-vector, so R p(s) is obtained by rotating
    start, vec and each coeffs[n] separately.

    phi is deliberately NOT rotated -- see encode_phi; the rotation is applied
    to the reconstructed vertices instead.
    """
    from models.morphoformer import RandomSO3Rotation
    B = batch["phi"].shape[0]
    R = RandomSO3Rotation.sample(B, batch["phi"].dtype, device)          # [B,3,3]
    rv = lambda x: torch.einsum('bij,b...j->b...i', R, x.to(device))
    out = dict(batch)
    for k in ("start_points", "branch_direction", "branch_vector", "fourier_coeffs"):
        if k in batch:
            out[k] = rv(batch[k])
    return out, R


def prepare_batch(batch):
    return {
        "branch_vector":  batch["branch_vector"].to(DEVICE),    # [B, mb, 3]
        "fourier_coeffs": batch["fourier_coeffs"].to(DEVICE),   # [B, mb, k, 3]
        "phi":            batch["phi"].to(DEVICE),              # [B, 144, 3]
        "branch_mask":    batch["branch_mask"].to(DEVICE),     # [B, mb]  present branches
        "opening_mask":   opening_mask_from_types(batch["aneurysm_type"], MAX_BRANCHES).to(DEVICE),  # candidates
        "cond":           make_conditions(batch),
    }


def model_args():
    """Kwargs needed to reconstruct the branch model from a checkpoint alone.

    Excludes `multi_recon` (the GCN variant's first positional arg) since it
    isn't serializable — pass it in separately when use_gcn is True.
    """
    args = {
        "use_gcn": USE_GCN,
        "k": K,
        "hidden_dim": HIDDEN_DIM,
        "latent_dim": LATENT_DIM,
        "num_layers": NUM_LAYERS,
        "num_types": NUM_TYPES,
        "max_branches": MAX_BRANCHES,
        "ghd_dim": GHD_DIM,
        "cond_dim": COND_DIM,
    }
    if USE_GCN:
        args["gcn_hidden"] = GCN_HIDDEN
        args["gcn_pool_ratio"] = GCN_POOL_RATIO
    return args


def all_hparams():
    """Every module-level training hyperparameter, for record-keeping in the checkpoint."""
    return {
        "use_gcn": USE_GCN,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "lr_min": LR_MIN,
        "k": K,
        "hidden_dim": HIDDEN_DIM,
        "latent_dim": LATENT_DIM,
        "num_layers": NUM_LAYERS,
        "cond_dim": COND_DIM,
        "ghd_dim": GHD_DIM,
        "max_branches": MAX_BRANCHES,
        "num_types": NUM_TYPES,
        "gcn_hidden": GCN_HIDDEN,
        "gcn_pool_ratio": GCN_POOL_RATIO,
        "kl_base": KL_BASE,
        "kl_mult": KL_MULT,
        "kl_weight": KL_WEIGHT,
        "kl_anneal_epochs": KL_ANNEAL_EPOCHS,
        "vec_weight": VEC_WEIGHT,
        "presence_weight": PRESENCE_WEIGHT,
        "syn_dir_weight": SYN_DIR_WEIGHT,
        "b_syn": B_SYN,
        "synthetic_pool": str(SYNTHETIC_POOL),
        "save_every": SAVE_EVERY,
        "log_every": LOG_EVERY,
        "sanity_every": SANITY_EVERY,
        "sanity_synthetic": SANITY_SYNTHETIC,
        "val_fraction": VAL_FRACTION,
        "rotation_augment": ROTATION_AUGMENT,
        "val_seed": VAL_SEED,
    }


def save_checkpoint(model, optimizer, epoch):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "k": K, "num_types": NUM_TYPES, "max_branches": MAX_BRANCHES,
            "coeff_mean": model.coeff_mean, "coeff_std": model.coeff_std,
            "vec_mean": model.vec_mean, "vec_std": model.vec_std,
            "args": model_args(),
            "hparams": all_hparams(),
        },
        SAVE_DIR / f"epoch_{epoch:05d}.pth",
    )


@torch.no_grad()
@torch.no_grad()
def evaluate(model, subset, batch_size=64):
    """Same losses as training, on held-out cases.

    Without this the run reports only training loss, and on 471 samples with a
    hidden-512 decoder a training loss near zero is equally consistent with
    learning the distribution and with memorising the set. The val/train ratio
    is the only thing that separates them.
    """
    from torch.utils.data import DataLoader
    was_training = model.training
    model.eval()
    tot, n = {}, 0
    for batch in DataLoader(subset, batch_size=batch_size):
        d = prepare_batch(batch)
        out, pl, mu, lv = model(d["branch_vector"], d["fourier_coeffs"], d["phi"], d["cond"])
        c, v, pr, k = model.get_loss(out, d["branch_vector"], d["fourier_coeffs"],
                                     d["branch_mask"], d["opening_mask"], pl, mu, lv)
        b = batch["phi"].size(0); n += b
        for kk, vv in (("coeff", c), ("vec", v), ("pres", pr), ("kl", k)):
            tot[kk] = tot.get(kk, 0.0) + float(vv) * b
    if was_training:
        model.train()
    return {f"val_{kk}": vv / max(n, 1) for kk, vv in tot.items()}


def plot_loss_curves(hist, save_dir, title):
    """Overwrite loss_curves.png each log step, so a run can be judged without
    opening wandb."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    keys = ["loss", "coeff", "vec", "pres", "kl", "syn", "w_dist", "d_sep",
            "adv", "lam", "kl_w", "lr"]
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    # train and val drawn on the SAME axes, so a gap opening up is visible
    # rather than something you have to reconstruct from two pictures.
    overlay = {"coeff": "val_coeff", "vec": "val_vec", "pres": "val_pres", "kl": "val_kl"}
    for ax, k in zip(axes.flat, keys):
        v = hist.get(k, [])
        if len(v) != len(hist["epoch"]):
            ax.set_axis_off(); continue
        ax.plot(hist["epoch"], v, lw=1.2, label="train")
        vk = overlay.get(k)
        if vk and len(hist.get(vk, [])) == len(hist["epoch"]):
            ax.plot(hist["epoch"], hist[vk], lw=1.2, ls="--", color="crimson", label="val")
            ax.legend(fontsize=7)
        ax.set_title(k); ax.set_xlabel("epoch"); ax.grid(alpha=0.3)
        if k in ("loss", "coeff", "vec", "kl"):
            ax.set_yscale("log")
    fig.suptitle(title); fig.tight_layout()
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(save_dir) / "loss_curves.png", dpi=130)
    plt.close(fig)


def sanity_check(model, val_set, multi_recon, epoch, save_dir, z_zero=True, n=8, ncols=4):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from dataset.preprocess_ImperialNHS import _reconstruct_ghd_numpy, _set_axes_equal

    model.eval()
    n = min(n, len(val_set))
    nrows = (n + ncols - 1) // ncols
    fig = plt.figure(figsize=(4 * ncols * 2, 4 * nrows))
    bright = ["#e6194B", "#3cb44b", "#4363d8"]

    for i in range(n):
        item = val_set[i]
        cond = MultiBranchConditions(
            aneurysm_type    = item["aneurysm_type"].unsqueeze(0).to(DEVICE),
            scale            = item["scale"].unsqueeze(0).to(DEVICE),
            start_points     = item["start_points"].unsqueeze(0).to(DEVICE),
            branch_direction = item["branch_direction"].unsqueeze(0).to(DEVICE),
            branch_mask      = item["branch_mask"].unsqueeze(0).to(DEVICE),
        )
        phi = item["phi"].unsqueeze(0).to(DEVICE)
        z = torch.zeros(1, model.latent_dim, device=DEVICE) if z_zero else None
        vec_p, coeff_p, pres_p = model.sample(phi, cond, z=z)
        vec_p, coeff_p, pres_p = vec_p[0].cpu().numpy(), coeff_p[0].cpu().numpy(), pres_p[0].cpu().numpy()

        atype  = int(item["aneurysm_type"])
        n_open = TYPE_N_OPEN.get(atype, model.max_branches)
        # multi_recon, NOT _reconstruct_ghd_numpy: the latter's norm_canonical
        # carries a legacy * 1.10 * 2.50, which leaves the canonical template in
        # place but scales the DEFORMATION by 2.75 -- the panel would show a
        # caricature of the shape, with correctly-placed centerlines beside it.
        verts = multi_recon._reconstruct_verts_np(item["phi"].numpy(), atype)
        faces = multi_recon.get(atype).canonical_Meshes.faces_packed().cpu().numpy()

        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces,
                        color="lightgray", edgecolor="none", alpha=0.12)
        start = item["start_points"].numpy()
        vec_g = item["branch_vector"].numpy()
        coeff_g = item["fourier_coeffs"].numpy()
        gt_mask = item["branch_mask"].numpy().astype(bool)
        all_pts, pres_str = [verts], []
        for b in range(model.max_branches):
            c = bright[b % len(bright)]
            # ground truth: branches actually present in the data
            if gt_mask[b]:
                gt = reconstruct_branch(start[b], start[b] + vec_g[b], coeff_g[b])
                ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], color="black", lw=2.5)
                all_pts.append(gt)
            # prediction: only candidate openings the model deems present
            if b < n_open:
                pres_str.append(f"{pres_p[b]:.2f}")
                if pres_p[b] > 0.5:
                    pr = reconstruct_branch(start[b], start[b] + vec_p[b], coeff_p[b])
                    ax.plot(pr[:, 0], pr[:, 1], pr[:, 2], color=c, lw=2.0, ls=(0, (4, 2)))
                    all_pts.append(pr)
        ax.set_axis_off()
        ax.set_title(f"{item['case']}\npres=[{','.join(pres_str)}]", fontsize=7)
        _set_axes_equal(ax, *all_pts)

    fig.suptitle(f"epoch {epoch}  (black=GT, dashed=pred, z={'0' if z_zero else '~N'})", fontsize=10)
    fig.tight_layout()
    save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / f"sanity_epoch_{epoch:05d}.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    model.train()
    return out


@torch.no_grad()
def sanity_synthetic(model, pool, multi_recon, epoch, save_dir,
                     z_zero=True, n=8, ncols=4, seed=0):
    """Fully-generative sanity: synthetic GHD phi → mesh openings → generated
    branches (presence head decides which openings get a branch).

    Shapes and openings come from the pre-computed pool, so this renders without
    the generator or the sensor. The same seed picks the same shapes every time,
    which makes the panel comparable across epochs -- it was previously redrawn
    from the VAE each call, so a change between two panels mixed the model
    getting better with the shapes being different.
    """
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from dataset.preprocess_ImperialNHS import _reconstruct_ghd_numpy, _set_axes_equal

    # Rendered in the CANONICAL frame: phi, the mesh, the caps, the tangents and
    # the generated centerlines are all unrotated, so the panel is internally
    # consistent. It is not a rotation test -- the training loss is where
    # rotation is applied, and there the conditions turn with the mesh.
    model.eval()
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = pool.sample_batch(n, generator=g)["index"]
    b = pool.batch(idx, device=DEVICE)
    types, phi = b["aneurysm_type"], b["phi"]
    starts, dirs, omask = b["start_points"], b["branch_direction"], b["branch_mask"]
    cond = MultiBranchConditions(types, torch.ones(n, device=DEVICE), starts, dirs, omask)
    z = torch.zeros(n, model.latent_dim, device=DEVICE) if z_zero else None
    vec_p, coeff_p, pres_p = model.sample(phi, cond, z=z)

    starts_np = starts.cpu().numpy()
    vec_np, coeff_np, pres_np = vec_p.cpu().numpy(), coeff_p.cpu().numpy(), pres_p.cpu().numpy()
    bright = ["#e6194B", "#3cb44b", "#4363d8"]
    nrows = (n + ncols - 1) // ncols
    fig = plt.figure(figsize=(4 * ncols * 2, 4 * nrows))
    for i in range(n):
        atype  = int(types[i])
        n_open = int(omask[i].sum())
        # multi_recon, NOT _reconstruct_ghd_numpy: the latter carries a legacy
        # norm_canonical * 1.10 * 2.50, which leaves the canonical template in
        # place but multiplies the DEFORMATION by 2.75 -- so the panel would
        # show a caricature of the shape the model actually saw.
        verts = multi_recon._reconstruct_verts_np(phi[i].cpu().numpy(), atype)
        faces = multi_recon.get(atype).canonical_Meshes.faces_packed().cpu().numpy()

        # generated centerlines, kept for the fusion below
        branch_pts, pres_str = [None] * n_open, []
        for b in range(n_open):
            pres_str.append(f"{pres_np[i, b]:.2f}")
            if pres_np[i, b] > 0.5:
                branch_pts[b] = reconstruct_branch(
                    starts_np[i, b], starts_np[i, b] + vec_np[i, b], coeff_np[i, b])

        # LEFT: stage-1 mesh + the generated centerlines
        ax = fig.add_subplot(nrows, ncols * 2, 2 * i + 1, projection="3d")
        ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces,
                        color="lightgray", edgecolor="none", alpha=0.18)
        all_pts = [verts]
        for b, pr in enumerate(branch_pts):
            if pr is not None:
                ax.plot(pr[:, 0], pr[:, 1], pr[:, 2], color=bright[b % len(bright)], lw=2.2)
                all_pts.append(pr)
        ax.set_axis_off()
        ax.set_title(f"type {atype} stage-1 + centerlines\npres=[{','.join(pres_str)}]", fontsize=7)
        _set_axes_equal(ax, *all_pts)

        # RIGHT: the merged mesh -- planarized caps with the cross-section swept
        # along those same centerlines. None when a rim will not close.
        ax2 = fig.add_subplot(nrows, ncols * 2, 2 * i + 2, projection="3d")
        fused = None
        try:
            fused = pool.fused_mesh(
                int(idx[i]), multi_recon, branch_pts,
                np.array([p is not None for p in branch_pts]))
        except Exception as exc:
            tag = f"fusion error\n{type(exc).__name__}"
        else:
            tag = "merged mesh" if fused is not None else "uncapping failed\n(skipped)"
        if fused is not None:
            fv = np.asarray(fused.points)
            ff = fused.faces.reshape(-1, 4)[:, 1:]
            ax2.plot_trisurf(fv[:, 0], fv[:, 1], fv[:, 2], triangles=ff,
                             color="#8fa7d4", edgecolor="none", alpha=0.9)
            _set_axes_equal(ax2, fv)
        ax2.set_axis_off(); ax2.set_title(tag, fontsize=7)

    fig.suptitle(f"epoch {epoch}  synthetic generation (z={'0' if z_zero else '~N'})", fontsize=10)
    fig.tight_layout()
    save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / f"sanity_synth_epoch_{epoch:05d}.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    model.train()
    return out


class SyntheticDirectionLoss:
    """Constrain generated branches' initial tangent to the GHD mesh opening direction.

    For a batch of pre-computed synthetic shapes:
      1. (no file read is more than an index) take phi and the mesh openings --
         per-branch start points, outward directions, valid mask -- from the pool.
      2. (grad) inject those as the condition, decode Fourier branches with z~N(0,I),
         compute each branch's analytic initial tangent, and penalise its angular
         deviation from the conditioned outward direction.
    Gradients flow through the branch model's ghd encoder / cond embed / decoder only.

    Step 1 used to decode a frozen GHD VAE and run the sensor on every call. Both
    are frozen, so it produced a fresh draw from a fixed distribution at the cost
    of two resident networks and an encoder forward per step. Drawing from a
    pre-computed pool instead gives the same distribution; what it gives up is
    novelty, since the loss now revisits 1000 shapes rather than never repeating.
    """

    def __init__(self, pool, num_types=NUM_TYPES, max_branches=MAX_BRANCHES,
                 b_syn=B_SYN, device=DEVICE):
        self.pool = pool
        self.num_types = num_types
        self.max_branches = max_branches
        self.b_syn = b_syn
        self.device = device

    def __call__(self, model):
        # Every pool record passed the cap check when it was written, so there is
        # no skip_failed equivalent here: a branch that could not be swept was
        # never stored, and b_mask carries only openings that close.
        b = self.pool.sample_batch(self.b_syn, device=self.device)
        phi, types = b["phi"], b["aneurysm_type"]
        starts, dirs, b_mask = b["start_points"], b["branch_direction"], b["branch_mask"]
        if not b_mask.any():
            return None

        # Same rotation augmentation as the reconstruction step. Without it the
        # conditioner would be trained on rotated meshes by one loss term and
        # unrotated ones by this one, i.e. two different input distributions.
        #
        # ROTATE THE CONDITIONS WITH THE MESH. The pool stores start points and
        # tangents in the canonical frame, the frame phi is expressed in. `rot`
        # goes to encode_phi, which turns the reconstructed vertices, so the
        # start point and the tangent must turn by the same R or the model would
        # be told to aim a branch in a direction that no longer points out of the
        # opening it belongs to. Both are plain 3-vectors, so one einsum each.
        rot = None
        if ROTATION_AUGMENT and USE_GCN:
            from models.morphoformer import RandomSO3Rotation
            rot = RandomSO3Rotation.sample(self.b_syn, phi.dtype, self.device)
            starts = torch.einsum('bij,bkj->bki', rot, starts)
            dirs = torch.einsum('bij,bkj->bki', rot, dirs)

        # grad path: condition on the (frozen) opening geometry, generate branches
        ghd_embed = model.encode_phi(phi, types, rot)
        cond = MultiBranchConditions(
            aneurysm_type    = types,
            scale            = torch.ones(self.b_syn, device=self.device),
            start_points     = starts,
            branch_direction = dirs,
            branch_mask      = b_mask,
            ghd_embed        = ghd_embed,
        )
        cond_emb = model.cond_embed(cond)
        z = torch.randn(self.b_syn, model.latent_dim, device=self.device)
        vec, coeffs = model._denorm_out(model._decode(z, cond_emb))
        pred_dir = model.initial_tangent(vec, coeffs)                 # [B, mb, 3]
        cos = (pred_dir * dirs).sum(dim=-1)                           # [B, mb]
        return (1.0 - cos)[b_mask].mean()


def build_model(multi_recon=None):
    args = model_args()
    use_gcn = args.pop("use_gcn")
    if use_gcn:
        return BranchFourierVAE_GCNConditioner(multi_recon, **args).to(DEVICE)
    return BranchFourierVAE(**args).to(DEVICE)


def main():
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config=all_hparams(),
    )

    dataset = VesselSkeletonDatasetFourier(PROCESSED_ROOT, max_branches=MAX_BRANCHES,
                                           k=K, normalize=False)   # model owns normalization
    n_val   = max(1, int(len(dataset) * VAL_FRACTION))
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(VAL_SEED))
    # drop_last=False: with 471 cases the remainder is a real part of the corpus,
    # and discarding it meant a different slice went unseen every epoch -- 23 cases
    # at batch 64, and 215 of 471 back when the batch was 256. Nothing here needs
    # equal-sized batches: normalization is per layer, not per batch.
    # num_workers is overridable because the workers share /dev/shm: with 16
    # concurrent runs at 4 workers each, one run died mid-training with
    # "DataLoader worker killed by signal: Bus error", which is shared-memory
    # exhaustion rather than anything wrong with the run itself.
    loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=int(os.environ.get("NUM_WORKERS", 4)),
                        drop_last=False)

    # multi_recon reconstructs meshes from phi. Still needed: the GCN
    # conditioner encodes the stage-1 mesh, and the sanity panel draws it.
    # The stage-1 VAE and the sensor are NOT loaded -- everything this run used
    # them for is already in the pool.
    multi_recon = None
    if USE_GCN or SANITY_SYNTHETIC:
        from models.multi_canonical_ghd_reconstruct import MultiCanonicalGHDReconstruct
        multi_recon = MultiCanonicalGHDReconstruct(CANONICAL_ROOT, device=DEVICE)

    critic = opt_d = None
    if _args.gan:
        from models.branch_critic import CurveCritic, curve_from_coeffs, gradient_penalty
        critic = CurveCritic(hidden=_args.critic_hidden, n_types=NUM_TYPES,
                             max_branches=MAX_BRANCHES).to(DEVICE)
        # betas (0, 0.9): the WGAN-GP default. Low momentum keeps the critic
        # responsive to a generator that is still moving.
        opt_d = torch.optim.Adam(critic.parameters(), lr=_args.critic_lr, betas=(0.0, 0.9))
        print(f"Curve critic enabled: adv_weight={_args.adv_weight}, "
              f"n_critic={_args.n_critic}, warmup={_args.adv_warmup} epochs")

    model = build_model(multi_recon)
    model.set_normalization_stats(*compute_target_stats(dataset))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # Cosine rather than linear decay. Linear spends the whole run stepping the
    # rate down at a constant pace, so the middle of training is already at a
    # fraction of LR while the model is still making coarse progress. Cosine
    # holds near LR for longer and then drops quickly, which is where the long
    # runs need the fine steps: the same shape the transformer already uses.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(EPOCHS, 1), eta_min=LR_MIN
    )

    pool = None
    if SYN_DIR_WEIGHT > 0 or SANITY_SYNTHETIC:
        pool = SyntheticShapeDataset(SYNTHETIC_POOL, max_branches=MAX_BRANCHES)
        m = pool.manifest
        print(f"Synthetic pool: {pool} "
              f"[generator {Path(m.get('ghd_vae', '?')).parent.name}, "
              f"sensor {Path(m.get('sensor', '?')).parent.name}, "
              f"accepted {100 * m.get('acceptance_rate', float('nan')):.1f}%]")

    syn_dir_loss = None
    if SYN_DIR_WEIGHT > 0:
        syn_dir_loss = SyntheticDirectionLoss(pool)
        print(f"Synthetic-direction loss enabled (B_SYN={B_SYN})")
    print(f"{_VARIANT}: {len(train_set)} train / {len(val_set)} val  | coeff_dim={dataset.get_coeff_dim()}")

    history = {"epoch": [], "kl_w": [], "lr": []}
    for epoch in range(EPOCHS + 1):
        model.train()
        kl_weight = KL_WEIGHT * min(1.0, epoch / KL_ANNEAL_EPOCHS)
        metrics = {}
        for batch in loader:
            data = prepare_batch(batch)
            rot = None
            if ROTATION_AUGMENT and USE_GCN:
                # GCN variant only: the non-GCN conditioner reads phi directly,
                # which lives in the fixed canonical basis and cannot be rotated.
                batch_r, rot = rotate_batch(batch, DEVICE)
                data = prepare_batch(batch_r)
            out, presence_logit, mu, logvar = model(
                data["branch_vector"], data["fourier_coeffs"], data["phi"], data["cond"], rot=rot)
            coeff_loss, vec_loss, presence_loss, kl_loss = model.get_loss(
                out, data["branch_vector"], data["fourier_coeffs"],
                data["branch_mask"], data["opening_mask"], presence_logit, mu, logvar)
            loss = (coeff_loss + VEC_WEIGHT * vec_loss
                    + PRESENCE_WEIGHT * presence_loss + kl_weight * kl_loss)
            metrics = {
                "loss": float(loss), "coeff": float(coeff_loss), "vec": float(vec_loss),
                "pres": float(presence_loss), "kl": float(kl_loss),
            }

            if syn_dir_loss is not None:
                syn = syn_dir_loss(model)
                if syn is not None:
                    loss = loss + SYN_DIR_WEIGHT * syn
                    metrics["syn"] = float(syn)

            if critic is not None and epoch >= _args.adv_warmup:
                cond = data["cond"]
                bm = data["branch_mask"]
                if bm.any():
                    idx = bm.nonzero(as_tuple=False)                       # [M, 2] (sample, slot)
                    bi, sl = idx[:, 0], idx[:, 1]
                    types_f = cond.aneurysm_type[bi]
                    starts = cond.start_points[bi, sl]
                    real_curve = curve_from_coeffs(
                        starts, data["branch_vector"][bi, sl], data["fourier_coeffs"][bi, sl])

                    def _fake(detach):
                        c2 = cond._replace(ghd_embed=model.encode_phi(data["phi"], cond.aneurysm_type, rot))
                        ce = model.cond_embed(c2)
                        z = torch.randn(data["phi"].size(0), model.latent_dim, device=DEVICE)
                        o = model._decode(z, ce)
                        v, cf = model._denorm_out(o)
                        if detach:
                            v, cf = v.detach(), cf.detach()
                        return curve_from_coeffs(starts, v[bi, sl], cf[bi, sl])

                    # critic steps: maximise E[D(real)] - E[D(fake)]
                    for _ in range(_args.n_critic):
                        with torch.no_grad():
                            fc = _fake(detach=True)
                        sr, sf = critic(real_curve, types_f, sl), critic(fc, types_f, sl)
                        gp = gradient_penalty(critic, real_curve, fc, types_f, sl)
                        d_loss = (sf.mean() - sr.mean()) + _args.gp_weight * gp
                        opt_d.zero_grad(); d_loss.backward(); opt_d.step()
                        metrics["w_dist"] = float((sr - sf).mean().detach())
                        metrics["d_sep"] = float((sr.detach()[:, None] > sf.detach()[None, :]).float().mean())

                    # generator: -E[D(fake)], rescaled so its gradient matches the
                    # reconstruction gradient at the decoder's output layer
                    adv = -critic(_fake(detach=False), types_f, sl).mean()
                    last = model.dec[-1] if hasattr(model.dec, "__getitem__") else None
                    w = last.weight if last is not None and hasattr(last, "weight") else None
                    if w is not None:
                        gr = torch.autograd.grad(coeff_loss, w, retain_graph=True)[0]
                        ga = torch.autograd.grad(adv, w, retain_graph=True)[0]
                        lam = (gr.norm() / (ga.norm() + 1e-4)).clamp(0, 1e4).detach()
                    else:
                        lam = torch.tensor(1.0, device=DEVICE)
                    loss = loss + _args.adv_weight * lam * adv
                    metrics["adv"] = float(adv); metrics["lam"] = float(lam)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        scheduler.step()
        if epoch % LOG_EVERY == 0:
            # evaluate BEFORE printing, or the val losses reach the plot but not
            # the log, and a log-based analysis sees NaN for them
            metrics.update(evaluate(model, val_set))
            print({"epoch": epoch, "kl_w": round(kl_weight, 5),
                   "lr": round(scheduler.get_last_lr()[0], 6), **metrics})
            wandb.log({"epoch": epoch, "kl_w": kl_weight,
                       "lr": scheduler.get_last_lr()[0], **metrics}, step=epoch)
            history["epoch"].append(epoch)
            history["kl_w"].append(kl_weight)
            history["lr"].append(scheduler.get_last_lr()[0])
            for k in ("loss", "coeff", "vec", "pres", "kl", "syn",
                      "w_dist", "d_sep", "adv", "lam",
                      "val_coeff", "val_vec", "val_pres", "val_kl"):
                if k in metrics:
                    history.setdefault(k, []).append(metrics[k])
            plot_loss_curves(history, SAVE_DIR, SAVE_DIR.name)
        if epoch % SAVE_EVERY == 0:
            save_checkpoint(model, optimizer, epoch)
        if epoch % SANITY_EVERY == 0:
            sanity_check(model, val_set, multi_recon, epoch, SAVE_DIR)
            if SANITY_SYNTHETIC and pool is not None:
                sanity_synthetic(model, pool, multi_recon, epoch, SAVE_DIR)

    wandb.finish()


if __name__ == "__main__":
    main()

"""
conda activate new
python scripts/train/train_branch_mlp_vae.py

"""
