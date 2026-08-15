"""
Train the claydoll branch VAE (models.branch_mlp_vae_claydoll[_attn]) — the
unclipped-centerline variant — on VesselSkeletonDatasetFourierClaydoll.

Adapted from scripts/train/train_branch_mlp_vae.py. Two differences follow
directly from claydoll's design (see models/branch_mlp_vae_claydoll.py and
models/branch_mlp_vae_claydoll_attn.py's module docstrings):

  - No SyntheticDirectionLoss / self-supervised term. The clipped model needed
    it to bridge a train/generate conditioning mismatch (real branch data at
    train time vs. compute_branch_conditions' mesh-opening lookup at generate
    time). Claydoll's per-branch start points are *predicted* by the model
    itself from mesh geometry, identically at train and generate time, so that
    mismatch — and the loss that patched it — doesn't exist here. Only
    supervised losses (coeff/vec/presence/start_points/point_mse) + KL.
  - Two extra loss terms beyond the clipped model's: start_points_loss (masked
    MSE, predicted vs. real per-branch start) and point_mse (masked MSE in
    physical point-space, reconstructing both predicted and real branches from
    the same real start point — see models.branch_mlp_vae_claydoll.
    reconstruct_branch_torch — since coeff_loss/vec_loss operate on normalized,
    per-harmonic-weighted parameters and don't track geometric reconstruction
    error 1:1). No compute_branch_conditions dependency anywhere, including in
    the fully-generative sanity panel.

conda activate new
python scripts/train/train_branch_mlp_vae_claydoll.py             # base variant, cuda:1
python scripts/train/train_branch_mlp_vae_claydoll.py --attn      # cross-attention variant
python scripts/train/train_branch_mlp_vae_claydoll.py --device cuda:0
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path = [path for path in sys.path if Path(path or ".").resolve() != ROOT]
sys.path.insert(0, str(ROOT))

from dataset.skeleton_dataset_fourier_claydoll import VesselSkeletonDatasetFourierClaydoll, reconstruct_branch
from models.branch_mlp_vae_claydoll import BranchConditionsClaydoll, BranchFourierVAE_GCNConditionerClaydoll
from models.branch_mlp_vae_claydoll_attn import BranchConditionsClaydollAttn, BranchFourierVAE_GCNConditionerClaydollAttn
from utils.generate_synthetic import load_ghd_vae


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--attn", action="store_true",
                   help="Use the local cross-attention variant (branch_mlp_vae_claydoll_attn) "
                        "instead of the base global-pooled one (branch_mlp_vae_claydoll).")
    p.add_argument("--device", default=None, help="e.g. cuda:0, cuda:1, cpu (default: cuda:1).")
    # Only parse real argv when run as a script — importing this module for
    # smoke tests/reuse (as the training scripts before it are sometimes
    # imported) shouldn't choke on pytest/ipython's own argv.
    return p.parse_args() if __name__ == "__main__" else p.parse_args([])


_args = _parse_args()

# ── variant ───────────────────────────────────────────────────────────────────
USE_ATTN = _args.attn   # True → local cross-attention over intermediate GCN tokens (claydoll_attn);
                        # False → single global-pooled ghd_embed, differentiated by slot + predicted start (claydoll)

PROCESSED_ROOT = ROOT / "runtime" / "dataset" / "processed_claydoll"
CANONICAL_ROOT = ROOT / "dataset" / "canonical"

DEVICE     = torch.device(_args.device or ("cuda:1" if torch.cuda.is_available() else "cpu"))
EPOCHS     = 3000
BATCH_SIZE = 64
LR         = 1e-3
LR_MIN     = 1e-5

K            = 8           # Fourier harmonics per coordinate (must match the dataset)
HIDDEN_DIM   = 512
LATENT_DIM   = 8
NUM_LAYERS   = 4            # MLP depth in encoder / decoder
COND_DIM     = 64
GHD_DIM      = 16
MAX_BRANCHES = 3
NUM_TYPES    = 3

GCN_HIDDEN     = 32
GCN_POOL_RATIO = 0.5
LOCAL_DIM      = 32          # only used when USE_ATTN=True
ATTN_HEADS     = 4           # only used when USE_ATTN=True

KL_BASE      = 1e-3
KL_MULT      = 2             # integer multiplier; goes into SAVE_DIR name
KL_WEIGHT    = KL_BASE * KL_MULT
KL_ANNEAL_EPOCHS   = 300
VEC_WEIGHT          = 1.0    # weight of the chord (branch_vector) loss vs coeff loss
PRESENCE_WEIGHT     = 1.0    # weight of the per-branch presence (BCE) loss
START_POINTS_WEIGHT = 1.0    # weight of the per-branch predicted-start-point loss
POINT_MSE_WEIGHT    = 2.0    # weight of the point-space shape loss (reconstructed curve
                              # MSE, see models.branch_mlp_vae_claydoll.reconstruct_branch_torch)
                              # — deliberately bigger than coeff_loss's implicit weight of 1.0,
                              # since coeff_loss (normalized, per-harmonic) doesn't track
                              # physical-space reconstruction error 1:1.

# Candidate openings per aneurysm type (matches the canonical openings.npz counts,
# used for the presence-loss mask and sanity-panel display only — claydoll's
# conditioning itself has no dependency on openings.npz, see model docstrings).
TYPE_N_OPEN = {0: 3, 1: 2, 2: 2}
CANONICAL_TYPE_NAME = {0: "bifurcated", 1: "sidewall", 2: "sidewall"}


def opening_mask_from_types(aneurysm_type, max_branches):
    """[B, max_branches] bool — which slots are candidate openings for each type."""
    n = torch.tensor([TYPE_N_OPEN.get(int(t), max_branches) for t in aneurysm_type.tolist()])
    return torch.arange(max_branches).unsqueeze(0) < n.unsqueeze(1)


_VARIANT = "branch_mlp_vae_claydoll_attn" if USE_ATTN else "branch_mlp_vae_claydoll"
# _pmse{POINT_MSE_WEIGHT} disambiguates from runs launched before the point_mse
# loss existed (same h/l/z/kl hyperparameters, different SAVE_DIR — no collision).
SAVE_DIR = ROOT / "runtime" / "tr_checkpoints" / "v2" / "stage2" / "branch_mlp_vae_claydoll" / f"{_VARIANT}_h{HIDDEN_DIM}_l{NUM_LAYERS}_z{LATENT_DIM}_kl{KL_MULT}_pmse{POINT_MSE_WEIGHT:g}"

# Stage-1 GHD VAE, frozen — only used to sample synthetic phi for the fully-
# generative sanity panel (same checkpoint the clipped pipeline uses; stage 1
# is shared/unchanged between both pipelines).
GHD_VAE_CKPT = ROOT / "runtime" / "tr_checkpoints" / "v2" / "stage1" / "ghd_vae_h512_z16_kl2" / "epoch_05000.pth"

SAVE_EVERY   = 500
LOG_EVERY    = 10
SANITY_EVERY = 500
SANITY_SYNTHETIC = True
VAL_FRACTION = 0.1
VAL_SEED     = 42

WANDB_PROJECT = "AneuGv2_stage2_claydoll"
WANDB_RUN_NAME = SAVE_DIR.name

CondCls = BranchConditionsClaydollAttn if USE_ATTN else BranchConditionsClaydoll


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
    """aneurysm_type/scale/branch_mask only — start_points/ghd_embed (and
    local_embed for the attn variant) are filled in inside forward()/sample()
    from the model's own predictions, not supplied by the caller."""
    return CondCls(
        aneurysm_type = batch["aneurysm_type"].to(DEVICE),
        scale         = batch["scale"].to(DEVICE),
        branch_mask   = batch["branch_mask"].to(DEVICE),
    )


def prepare_batch(batch):
    return {
        "branch_vector":     batch["branch_vector"].to(DEVICE),     # [B, mb, 3]
        "fourier_coeffs":    batch["fourier_coeffs"].to(DEVICE),    # [B, mb, k, 3]
        "phi":               batch["phi"].to(DEVICE),               # [B, 144, 3]
        "branch_mask":       batch["branch_mask"].to(DEVICE),      # [B, mb]  present branches
        "start_points_real": batch["start_points"].to(DEVICE),     # [B, mb, 3]  start_points_loss target
        "opening_mask":      opening_mask_from_types(batch["aneurysm_type"], MAX_BRANCHES).to(DEVICE),
        "cond":              make_conditions(batch),
    }


def model_args():
    """Kwargs needed to reconstruct the branch model from a checkpoint alone.

    Excludes `multi_recon` (the model's first positional arg) since it isn't
    serializable — pass it in separately at reload time.
    """
    args = {
        "use_attn": USE_ATTN,
        "k": K,
        "hidden_dim": HIDDEN_DIM,
        "latent_dim": LATENT_DIM,
        "num_layers": NUM_LAYERS,
        "num_types": NUM_TYPES,
        "max_branches": MAX_BRANCHES,
        "ghd_dim": GHD_DIM,
        "cond_dim": COND_DIM,
        "gcn_hidden": GCN_HIDDEN,
        "gcn_pool_ratio": GCN_POOL_RATIO,
    }
    if USE_ATTN:
        args["local_dim"] = LOCAL_DIM
        args["attn_heads"] = ATTN_HEADS
    return args


def all_hparams():
    """Every module-level training hyperparameter, for record-keeping in the checkpoint."""
    return {
        "use_attn": USE_ATTN,
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
        "local_dim": LOCAL_DIM,
        "attn_heads": ATTN_HEADS,
        "kl_base": KL_BASE,
        "kl_mult": KL_MULT,
        "kl_weight": KL_WEIGHT,
        "kl_anneal_epochs": KL_ANNEAL_EPOCHS,
        "vec_weight": VEC_WEIGHT,
        "presence_weight": PRESENCE_WEIGHT,
        "start_points_weight": START_POINTS_WEIGHT,
        "point_mse_weight": POINT_MSE_WEIGHT,
        "ghd_vae_ckpt": str(GHD_VAE_CKPT),
        "save_every": SAVE_EVERY,
        "log_every": LOG_EVERY,
        "sanity_every": SANITY_EVERY,
        "sanity_synthetic": SANITY_SYNTHETIC,
        "val_fraction": VAL_FRACTION,
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
def sanity_check(model, val_set, epoch, save_dir, z_zero=True, n=8, ncols=4):
    """Ground truth (black, anchored at its real start) vs. prediction (dashed
    color, anchored at the model's OWN predicted start) — the gap between a
    black and dashed line's start visualizes start_points_loss directly."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from dataset.preprocess_ImperialNHS import _reconstruct_ghd_numpy, _set_axes_equal

    model.eval()
    n = min(n, len(val_set))
    nrows = (n + ncols - 1) // ncols
    fig = plt.figure(figsize=(4 * ncols, 4 * nrows))
    bright = ["#e6194B", "#3cb44b", "#4363d8"]

    for i in range(n):
        item = val_set[i]
        cond = CondCls(
            aneurysm_type = item["aneurysm_type"].unsqueeze(0).to(DEVICE),
            scale         = item["scale"].unsqueeze(0).to(DEVICE),
            branch_mask   = item["branch_mask"].unsqueeze(0).to(DEVICE),
        )
        phi = item["phi"].unsqueeze(0).to(DEVICE)
        z = torch.zeros(1, model.latent_dim, device=DEVICE) if z_zero else None
        vec_p, coeff_p, pres_p, start_p = model.sample(phi, cond, z=z)
        vec_p, coeff_p, pres_p, start_p = (vec_p[0].cpu().numpy(), coeff_p[0].cpu().numpy(),
                                           pres_p[0].cpu().numpy(), start_p[0].cpu().numpy())

        atype  = int(item["aneurysm_type"])
        n_open = TYPE_N_OPEN.get(atype, model.max_branches)
        verts, faces = _reconstruct_ghd_numpy(
            {"aneurysm_type": atype, "ghd": {"phi": item["phi"].numpy()}}, denormalize_shape=True)

        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces,
                        color="lightgray", edgecolor="none", alpha=0.12)
        start_g = item["start_points"].numpy()
        vec_g = item["branch_vector"].numpy()
        coeff_g = item["fourier_coeffs"].numpy()
        gt_mask = item["branch_mask"].numpy().astype(bool)
        all_pts, pres_str = [verts], []
        for b in range(model.max_branches):
            c = bright[b % len(bright)]
            # ground truth: branches actually present, anchored at their real start
            if gt_mask[b]:
                gt = reconstruct_branch(start_g[b], start_g[b] + vec_g[b], coeff_g[b])
                ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], color="black", lw=2.5)
                all_pts.append(gt)
            # prediction: candidate openings the model deems present, anchored
            # at the model's OWN predicted start (never the real one)
            if b < n_open:
                pres_str.append(f"{pres_p[b]:.2f}")
                if pres_p[b] > 0.5:
                    pr = reconstruct_branch(start_p[b], start_p[b] + vec_p[b], coeff_p[b])
                    ax.plot(pr[:, 0], pr[:, 1], pr[:, 2], color=c, lw=2.0, ls=(0, (4, 2)))
                    all_pts.append(pr)
        ax.set_axis_off()
        ax.set_title(f"{item['case']}\npres=[{','.join(pres_str)}]", fontsize=7)
        _set_axes_equal(ax, *all_pts)

    fig.suptitle(f"epoch {epoch}  (black=GT @ real start, dashed=pred @ predicted start, z={'0' if z_zero else '~N'})",
                fontsize=10)
    fig.tight_layout()
    save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / f"sanity_epoch_{epoch:05d}.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    model.train()
    return out


@torch.no_grad()
def sanity_synthetic(model, ghd_vae, ghd_mean, ghd_std, ghd_input_dim, epoch, save_dir,
                     z_zero=True, n=8, ncols=4, seed=0):
    """Fully-generative sanity: synthetic GHD phi -> generated branches, with
    start points ALSO predicted by the model — no mesh-opening lookup needed
    (contrast with the clipped model's version of this function, which needs
    multi_recon.compute_branch_conditions to seed start/direction)."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from dataset.preprocess_ImperialNHS import _reconstruct_ghd_numpy, _set_axes_equal

    model.eval()
    g = torch.Generator(device="cpu").manual_seed(seed)
    types = torch.randint(0, NUM_TYPES, (n,), generator=g).to(DEVICE)
    z_ghd = torch.randn(n, ghd_vae.latent_dim, generator=g).to(DEVICE)
    ghd_n, _ = ghd_vae.decode(z_ghd, types)
    phi = (ghd_n * ghd_std[:, :ghd_input_dim] + ghd_mean[:, :ghd_input_dim]).reshape(n, -1, 3)

    cond = CondCls(
        aneurysm_type = types,
        scale         = torch.ones(n, device=DEVICE),
        branch_mask   = torch.ones(n, MAX_BRANCHES, dtype=torch.bool, device=DEVICE),
    )
    z = torch.zeros(n, model.latent_dim, device=DEVICE) if z_zero else None
    vec_p, coeff_p, pres_p, start_p = model.sample(phi, cond, z=z)

    start_np = start_p.cpu().numpy()
    vec_np, coeff_np, pres_np = vec_p.cpu().numpy(), coeff_p.cpu().numpy(), pres_p.cpu().numpy()
    bright = ["#e6194B", "#3cb44b", "#4363d8"]
    nrows = (n + ncols - 1) // ncols
    fig = plt.figure(figsize=(4 * ncols, 4 * nrows))
    for i in range(n):
        atype  = int(types[i])
        n_open = TYPE_N_OPEN.get(atype, MAX_BRANCHES)
        verts, faces = _reconstruct_ghd_numpy(
            {"aneurysm_type": atype, "ghd": {"phi": phi[i].cpu().numpy()}}, denormalize_shape=True)
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces,
                        color="lightgray", edgecolor="none", alpha=0.12)
        all_pts, pres_str = [verts], []
        for b in range(n_open):
            pres_str.append(f"{pres_np[i, b]:.2f}")
            if pres_np[i, b] > 0.5:
                pr = reconstruct_branch(start_np[i, b], start_np[i, b] + vec_np[i, b], coeff_np[i, b])
                ax.plot(pr[:, 0], pr[:, 1], pr[:, 2], color=bright[b % len(bright)], lw=2.0)
                all_pts.append(pr)
        ax.set_axis_off()
        ax.set_title(f"type {atype}: {CANONICAL_TYPE_NAME.get(atype, '?')}  pres=[{','.join(pres_str)}]", fontsize=7)
        _set_axes_equal(ax, *all_pts)

    fig.suptitle(f"epoch {epoch}  synthetic generation, predicted start points (z={'0' if z_zero else '~N'})",
                fontsize=10)
    fig.tight_layout()
    save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / f"sanity_synth_epoch_{epoch:05d}.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    model.train()
    return out


def build_model(multi_recon):
    args = model_args()
    use_attn = args.pop("use_attn")
    cls = BranchFourierVAE_GCNConditionerClaydollAttn if use_attn else BranchFourierVAE_GCNConditionerClaydoll
    return cls(multi_recon, **args).to(DEVICE)


def main():
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config=all_hparams(),
    )

    dataset = VesselSkeletonDatasetFourierClaydoll(PROCESSED_ROOT, max_branches=MAX_BRANCHES,
                                                   k=K, normalize=False)   # model owns normalization
    n_val   = max(1, int(len(dataset) * VAL_FRACTION))
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(VAL_SEED))
    loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)

    # multi_recon is needed unconditionally (mesh reconstruction for the GCN
    # encoder); the frozen GHD VAE is only needed for the synthetic sanity panel.
    from models.multi_canonical_ghd_reconstruct import MultiCanonicalGHDReconstruct
    multi_recon = MultiCanonicalGHDReconstruct(CANONICAL_ROOT, device=DEVICE)

    model = build_model(multi_recon)
    model.set_normalization_stats(*compute_target_stats(dataset))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=LR_MIN / LR, total_iters=EPOCHS
    )

    ghd_vae = ghd_mean = ghd_std = ghd_input_dim = None
    if SANITY_SYNTHETIC:
        ghd_vae, ghd_mean, ghd_std, ghd_input_dim = load_ghd_vae(GHD_VAE_CKPT, DEVICE)
        for p in ghd_vae.parameters():
            p.requires_grad_(False)
        print(f"Loaded GHD VAE: {GHD_VAE_CKPT.name}")

    print(f"{_VARIANT}: {len(train_set)} train / {len(val_set)} val  | coeff_dim={dataset.get_coeff_dim()}")

    for epoch in range(EPOCHS + 1):
        model.train()
        kl_weight = KL_WEIGHT * min(1.0, epoch / KL_ANNEAL_EPOCHS)
        metrics = {}
        for batch in loader:
            data = prepare_batch(batch)
            out, presence_logit, mu, logvar, start_points_pred = model(
                data["branch_vector"], data["fourier_coeffs"], data["phi"], data["cond"])
            coeff_loss, vec_loss, presence_loss, kl_loss, start_points_loss, point_mse = model.get_loss(
                out, data["branch_vector"], data["fourier_coeffs"],
                data["branch_mask"], data["opening_mask"], presence_logit, mu, logvar,
                start_points_pred, data["start_points_real"])
            loss = (coeff_loss + VEC_WEIGHT * vec_loss + PRESENCE_WEIGHT * presence_loss
                    + kl_weight * kl_loss + START_POINTS_WEIGHT * start_points_loss
                    + POINT_MSE_WEIGHT * point_mse)
            metrics = {
                "loss": float(loss), "coeff": float(coeff_loss), "vec": float(vec_loss),
                "pres": float(presence_loss), "kl": float(kl_loss), "start": float(start_points_loss),
                "point_mse": float(point_mse),
            }

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        scheduler.step()
        if epoch % LOG_EVERY == 0:
            print({"epoch": epoch, "kl_w": round(kl_weight, 5),
                   "lr": round(scheduler.get_last_lr()[0], 6), **metrics})
            wandb.log({"epoch": epoch, "kl_w": kl_weight,
                       "lr": scheduler.get_last_lr()[0], **metrics}, step=epoch)
        if epoch % SAVE_EVERY == 0:
            save_checkpoint(model, optimizer, epoch)
        if epoch % SANITY_EVERY == 0:
            sanity_check(model, val_set, epoch, SAVE_DIR)
            if SANITY_SYNTHETIC and ghd_vae is not None:
                sanity_synthetic(model, ghd_vae, ghd_mean, ghd_std, ghd_input_dim, epoch, SAVE_DIR)

    wandb.finish()


if __name__ == "__main__":
    main()

"""
conda activate new
python scripts/train/train_branch_mlp_vae_claydoll.py
"""
