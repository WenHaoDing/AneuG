"""
Train the claydoll multi-branch transformer (models.branch_transformer_claydoll)
— the unclipped-centerline variant — on VesselSkeletonDatasetClaydoll.

Adapted from scripts/train/train_branch_transformer.py. Differences follow
directly from claydoll's design (see models/branch_transformer_claydoll.py's
module docstring, and models/branch_mlp_vae_claydoll.py's for the same idea
applied to the other model family):

  - No synthetic-direction / self-supervised loss. The clipped model needed one
    to bridge a train/generate conditioning mismatch (real branch data at train
    time vs. compute_branch_conditions' mesh-opening lookup at generate time).
    Claydoll's per-branch start points are *predicted* by the model itself from
    mesh geometry, identically at train and generate time, so that mismatch —
    and the loss that patched it (compute_synthetic_dir_loss, already disabled
    in the clipped script) — doesn't exist here. Only supervised losses
    (recon/length/start_points) + KL.
  - One extra loss term beyond the clipped model's: start_points_loss (masked
    MSE, predicted vs. real per-branch start). No point_mse-style extra term is
    needed here the way the Fourier-MLP family needed one — this model's own
    recon_loss already IS a direct point-space MSE, not a normalized-coefficient
    loss, so it already tracks geometric reconstruction error 1:1.
  - Only the GCN-mesh-encoder variant exists (no GHDTokenEncoder/USE_GCN=False
    option): the start-point head needs real mesh geometry to predict from, so
    a flat-MLP-on-phi-tokens variant has nowhere to derive it from.
  - A cross-attention conditioning variant IS available via --attn
    (models/branch_transformer_claydoll_attn.py), reusing
    models/branch_transformer.py's SlotQueryCrossAttention / GCNMeshEncoder-
    WithLocalAttention — same idea as models/branch_mlp_vae_claydoll_attn.py,
    applied to this architecture's per-token conditioning instead of a
    per-branch MLP's. Kept as a flag (not a separate script) so both variants
    share the rest of this file's training loop verbatim.

conda activate new
python scripts/train/train_branch_transformer_claydoll.py             # base variant
python scripts/train/train_branch_transformer_claydoll.py --attn      # cross-attention variant
python scripts/train/train_branch_transformer_claydoll.py --device cuda:0
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

from dataset.skeleton_dataset_claydoll import VesselSkeletonDatasetClaydoll
from models.branch_transformer import BranchConditionsClaydoll, BranchConditionsClaydollAttn
from models.branch_transformer_claydoll import MultiBranchVAEClaydollGCNConditioner
from models.branch_transformer_claydoll_attn import MultiBranchVAEClaydollAttnGCNConditioner
from utils.generate_synthetic import load_ghd_vae


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--attn", action="store_true",
                   help="Use the local cross-attention variant (branch_transformer_claydoll_attn) "
                        "instead of the base global-pooled one (branch_transformer_claydoll).")
    p.add_argument("--device", default=None, help="e.g. cuda:0, cuda:1, cpu (default: cuda:1).")
    # Only parse real argv when run as a script — importing this module for
    # smoke tests/reuse shouldn't choke on pytest/ipython's own argv.
    return p.parse_args() if __name__ == "__main__" else p.parse_args([])


_args = _parse_args()

# ── variant ───────────────────────────────────────────────────────────────────
USE_ATTN = _args.attn   # True → local cross-attention over intermediate GCN tokens (claydoll_attn);
                        # False → single global-pooled ghd_embed, differentiated by slot + predicted start (claydoll)

PROCESSED_ROOT = ROOT / "dataset" / "processed_claydoll"
CANONICAL_ROOT = ROOT / "dataset" / "canonical"

DEVICE     = torch.device(_args.device or ("cuda:1" if torch.cuda.is_available() else "cpu"))
EPOCHS     = 2000
BATCH_SIZE = 256
LR         = 1e-3
LR_MIN     = 1e-5

HIDDEN_DIM            = 64
LATENT_DIM            = 8
NUM_LAYERS            = 4
NHEAD                 = 4
GHD_DIM               = 16
MAX_BRANCHES          = 3
MAX_POINTS_PER_BRANCH = 128   # max_local_points = MAX_POINTS_PER_BRANCH - 1 = 127
NUM_TYPES             = 3

GCN_HIDDEN     = 32
GCN_POOL_RATIO = 0.5
LOCAL_DIM      = 32          # only used when USE_ATTN=True
ATTN_HEADS     = 4           # only used when USE_ATTN=True

KL_BASE          = 3e-4
KL_MULT          = 1
KL_WEIGHT        = KL_BASE * KL_MULT
KL_ANNEAL_EPOCHS = 200
LEN_WEIGHT          = 0.1
START_POINTS_WEIGHT = 1.0   # weight of the per-branch predicted-start-point loss

# Curvature-weighted reconstruction: weight each point by local curvature so
# bends/hooks count more. USE_POINT_WEIGHTS=False → plain (uniform) recon loss.
USE_POINT_WEIGHTS = True
CURVATURE_WEIGHT  = 0.25

# Candidate openings per aneurysm type — a fixed per-type topological constant
# (matches the canonical openings.npz counts), used ONLY for the synthetic
# sanity panel's branch_mask and display. Not MultiCanonicalGHDReconstruct.
# compute_branch_conditions — this variant has zero dependency on that function.
TYPE_N_OPEN = {0: 3, 1: 2, 2: 2}
CANONICAL_TYPE_NAME = {0: "bifurcated", 1: "sidewall", 2: "sidewall"}

_VARIANT = "branch_transformer_claydoll_attn" if USE_ATTN else "branch_transformer_claydoll"
SAVE_DIR = ROOT / "tr_checkpoints" / "v2" / "branch_transformer_claydoll" / f"{_VARIANT}_h{HIDDEN_DIM}_z{LATENT_DIM}_kl{KL_MULT}_start{START_POINTS_WEIGHT:g}"

# Stage-1 GHD VAE, frozen — only used to sample synthetic phi for the fully-
# generative sanity panel (same checkpoint the other claydoll pipeline uses;
# stage 1 is shared/unchanged between all v2/claydoll variants).
GHD_VAE_CKPT = ROOT / "tr_checkpoints" / "v2" / "stage1" / "ghd_vae_h512_z16_kl2" / "epoch_05000.pth"

SAVE_EVERY   = 200
LOG_EVERY    = 10
SANITY_EVERY = 500
SANITY_SYNTHETIC = True
VAL_FRACTION = 0.1
VAL_SEED     = 42

WANDB_PROJECT = "AneuGv2_stage2_claydoll"
WANDB_RUN_NAME = SAVE_DIR.name

CondCls = BranchConditionsClaydollAttn if USE_ATTN else BranchConditionsClaydoll


def branch_mask_from_types(types, max_branches, device):
    n = torch.tensor([TYPE_N_OPEN.get(int(t), max_branches) for t in types.tolist()], device=device)
    return torch.arange(max_branches, device=device).unsqueeze(0) < n.unsqueeze(1)


def model_args():
    """Kwargs needed to reconstruct the branch model from a checkpoint alone.

    Excludes `multi_recon` (the model's first positional arg) since it isn't
    serializable — pass it in separately at reload time.
    """
    args = {
        "use_attn": USE_ATTN,
        "hidden_dim": HIDDEN_DIM,
        "latent_dim": LATENT_DIM,
        "num_layers": NUM_LAYERS,
        "nhead": NHEAD,
        "max_local_points": MAX_POINTS_PER_BRANCH - 1,
        "num_types": NUM_TYPES,
        "max_branches": MAX_BRANCHES,
        "ghd_dim": GHD_DIM,
        "gcn_hidden": GCN_HIDDEN,
        "gcn_pool_ratio": GCN_POOL_RATIO,
    }
    if USE_ATTN:
        args["local_dim"] = LOCAL_DIM
        args["attn_heads"] = ATTN_HEADS
    return args


def build_model(multi_recon):
    args = model_args()
    use_attn = args.pop("use_attn")
    cls = MultiBranchVAEClaydollAttnGCNConditioner if use_attn else MultiBranchVAEClaydollGCNConditioner
    return cls(multi_recon, **args).to(DEVICE)


def all_hparams():
    """Every module-level training hyperparameter, for record-keeping in the checkpoint."""
    return {
        "use_attn": USE_ATTN,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "lr_min": LR_MIN,
        "hidden_dim": HIDDEN_DIM,
        "latent_dim": LATENT_DIM,
        "num_layers": NUM_LAYERS,
        "nhead": NHEAD,
        "ghd_dim": GHD_DIM,
        "max_branches": MAX_BRANCHES,
        "num_types": NUM_TYPES,
        "max_points_per_branch": MAX_POINTS_PER_BRANCH,
        "gcn_hidden": GCN_HIDDEN,
        "gcn_pool_ratio": GCN_POOL_RATIO,
        "local_dim": LOCAL_DIM,
        "attn_heads": ATTN_HEADS,
        "kl_base": KL_BASE,
        "kl_mult": KL_MULT,
        "kl_weight": KL_WEIGHT,
        "kl_anneal_epochs": KL_ANNEAL_EPOCHS,
        "len_weight": LEN_WEIGHT,
        "start_points_weight": START_POINTS_WEIGHT,
        "use_point_weights": USE_POINT_WEIGHTS,
        "curvature_weight": CURVATURE_WEIGHT,
        "ghd_vae_ckpt": str(GHD_VAE_CKPT),
        "save_every": SAVE_EVERY,
        "log_every": LOG_EVERY,
        "sanity_every": SANITY_EVERY,
        "sanity_synthetic": SANITY_SYNTHETIC,
        "val_fraction": VAL_FRACTION,
        "val_seed": VAL_SEED,
    }


def save_checkpoint(model, optimizer, dataset, epoch):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "point_mean": dataset.point_mean,
            "point_std": dataset.point_std,
            "args": model_args(),
            "hparams": all_hparams(),
        },
        SAVE_DIR / f"epoch_{epoch:05d}.pth",
    )


def prepare_batch(batch):
    """Reshape per-branch fields and move to DEVICE.

    local_points:  [B, max_branches, max_local_points, 3] → [B, seq_len, 3]
    token_mask:    [B, seq_len]     already flat in dataset (= point_mask.flatten())
    branch_length: [B, max_branches]
    branch_mask:   [B, max_branches]
    phi:           [B, 144, 3]
    start_points (real, from the dataset): loss target only, NOT fed to the
    model as conditioning — the model uses its own predicted start instead.
    """
    cond = CondCls(
        aneurysm_type = batch["aneurysm_type"].to(DEVICE),
        scale         = batch["scale"].to(DEVICE),
        branch_mask   = batch["branch_mask"].to(DEVICE),
    )
    return {
        "local_points":      batch["local_points"].flatten(1, 2).to(DEVICE),
        "token_mask":        batch["token_mask"].to(DEVICE),
        "token_weights":     batch["token_weights"].to(DEVICE),
        "branch_length":     batch["branch_length"].to(DEVICE),
        "branch_mask":       batch["branch_mask"].to(DEVICE),
        "phi":               batch["phi"].to(DEVICE),
        "start_points_real": batch["start_points"].to(DEVICE),
        "cond":              cond,
    }


@torch.no_grad()
def sanity_check(model, val_set, epoch, save_dir, n=8, ncols=4):
    """Ground truth (black, anchored at its real start) vs. prediction (dashed
    color, anchored at the model's OWN predicted start) — the gap between a
    black and dashed line's start visualizes start_points_loss directly."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from dataset.preprocess_ImperialNHS import _reconstruct_ghd_numpy, _set_axes_equal

    base_dataset = val_set.dataset   # unwrap Subset → VesselSkeletonDatasetClaydoll
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
        z = torch.zeros(1, model.latent_dim, device=DEVICE)
        outputs, lengths, _, start_p = model.sample(phi, cond, z=z)
        outputs = base_dataset.denormalize_local_points(outputs[0].cpu())
        outputs = outputs.view(model.max_branches, model.max_local_points, 3)
        lengths = lengths[0].cpu()
        start_p = start_p[0].cpu().numpy()

        atype = int(item["aneurysm_type"])
        verts, faces = _reconstruct_ghd_numpy(
            {"aneurysm_type": atype, "ghd": {"phi": item["phi"].numpy()}}, denormalize_shape=True)

        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces,
                        color="lightgray", edgecolor="none", alpha=0.12)

        gt_local = base_dataset.denormalize_local_points(item["local_points"])
        all_pts = [verts]
        for b in range(model.max_branches):
            if not item["branch_mask"][b]:
                continue
            c = bright[b % len(bright)]
            n_gt = int(item["branch_length"][b])
            start_g = item["start_points"][b].numpy()
            gt = np.concatenate([start_g[None], start_g[None] + gt_local[b, :n_gt].numpy()], axis=0)
            ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], color="black", lw=2.5)
            all_pts.append(gt)

            n_pr = int(lengths[b].clamp(min=1))
            pr = np.concatenate([start_p[b][None], start_p[b][None] + outputs[b, :n_pr].numpy()], axis=0)
            ax.plot(pr[:, 0], pr[:, 1], pr[:, 2], color=c, lw=2.0, ls=(0, (4, 2)))
            all_pts.append(pr)

        ax.set_axis_off()
        ax.set_title(f"{item['case']} (type {atype})", fontsize=8)
        _set_axes_equal(ax, *all_pts)

    fig.suptitle(f"epoch {epoch}  (black=GT @ real start, dashed=pred @ predicted start, z=0)", fontsize=10)
    fig.tight_layout()
    save_dir = Path(save_dir) / "sanity"; save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / f"epoch_{epoch:05d}.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    model.train()
    return out


@torch.no_grad()
def sanity_synthetic(model, ghd_vae, ghd_mean, ghd_std, ghd_input_dim, epoch, save_dir, n=8, ncols=4, seed=0):
    """Fully-generative sanity: synthetic GHD phi -> generated branches, with
    start points ALSO predicted by the model — no mesh-opening lookup needed
    (contrast with the clipped model's version of this function, which needs
    multi_recon.compute_branch_conditions to seed start/direction). branch_mask
    (which candidate slots to generate) comes from TYPE_N_OPEN, a fixed
    per-type topological constant, not from real or mesh-derived data."""
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
        branch_mask   = branch_mask_from_types(types, MAX_BRANCHES, DEVICE),
    )
    z = torch.zeros(n, model.latent_dim, device=DEVICE)
    outputs, lengths, _, start_p = model.sample(phi, cond, z=z)
    outputs = (outputs.cpu() * model.point_std.cpu() + model.point_mean.cpu())
    outputs = outputs.view(n, MAX_BRANCHES, MAX_POINTS_PER_BRANCH - 1, 3)
    lengths = lengths.cpu()
    start_np = start_p.cpu().numpy()

    bright = ["#e6194B", "#3cb44b", "#4363d8"]
    nrows = (n + ncols - 1) // ncols
    fig = plt.figure(figsize=(4 * ncols, 4 * nrows))
    for i in range(n):
        atype = int(types[i])
        verts, faces = _reconstruct_ghd_numpy(
            {"aneurysm_type": atype, "ghd": {"phi": phi[i].cpu().numpy()}}, denormalize_shape=True)
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces,
                        color="lightgray", edgecolor="none", alpha=0.12)
        all_pts = [verts]
        n_open = TYPE_N_OPEN.get(atype, MAX_BRANCHES)
        for b in range(n_open):
            n_pr = int(lengths[i, b].clamp(min=1))
            pr = np.concatenate(
                [start_np[i, b][None], start_np[i, b][None] + outputs[i, b, :n_pr].numpy()], axis=0)
            ax.plot(pr[:, 0], pr[:, 1], pr[:, 2], color=bright[b % len(bright)], lw=2.0)
            all_pts.append(pr)
        ax.set_axis_off()
        ax.set_title(f"type {atype}: {CANONICAL_TYPE_NAME.get(atype, '?')}", fontsize=8)
        _set_axes_equal(ax, *all_pts)

    fig.suptitle(f"epoch {epoch}  synthetic generation, predicted start points (z=0)", fontsize=10)
    fig.tight_layout()
    save_dir = Path(save_dir) / "sanity"; save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / f"sanity_synth_epoch_{epoch:05d}.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    model.train()
    return out


def main():
    wandb.init(project=WANDB_PROJECT, name=WANDB_RUN_NAME, config=all_hparams())

    dataset = VesselSkeletonDatasetClaydoll(
        PROCESSED_ROOT,
        max_branches=MAX_BRANCHES,
        max_points_per_branch=MAX_POINTS_PER_BRANCH,
        curvature_weight=CURVATURE_WEIGHT,
    )
    n_val   = max(1, int(len(dataset) * VAL_FRACTION))
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(VAL_SEED),
    )
    loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)

    from models.multi_canonical_ghd_reconstruct import MultiCanonicalGHDReconstruct
    multi_recon = MultiCanonicalGHDReconstruct(CANONICAL_ROOT, device=DEVICE)

    model = build_model(multi_recon)
    model.set_normalization_stats(dataset.point_mean, dataset.point_std)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR_MIN)

    ghd_vae = ghd_mean = ghd_std = ghd_input_dim = None
    if SANITY_SYNTHETIC:
        ghd_vae, ghd_mean, ghd_std, ghd_input_dim = load_ghd_vae(GHD_VAE_CKPT, DEVICE)
        for p in ghd_vae.parameters():
            p.requires_grad_(False)
        print(f"Loaded GHD VAE: {GHD_VAE_CKPT.name}")

    print(f"{_VARIANT}: {len(train_set)} train / {len(val_set)} val")

    for epoch in range(EPOCHS + 1):
        model.train()
        kl_weight = KL_WEIGHT * min(1.0, epoch / KL_ANNEAL_EPOCHS)
        metrics = {}
        for batch in loader:
            data = prepare_batch(batch)

            recon, length_logit, mu, logvar, start_points_pred = model(
                data["local_points"], data["token_mask"], data["phi"], data["cond"],
            )
            recon_loss, kl_loss, length_loss, start_points_loss = model.get_loss(
                recon, data["local_points"], length_logit,
                data["branch_length"], data["token_mask"], data["branch_mask"],
                mu, logvar, start_points_pred, data["start_points_real"],
                point_weights=data["token_weights"] if USE_POINT_WEIGHTS else None,
            )
            loss = (recon_loss + kl_weight * kl_loss + LEN_WEIGHT * length_loss
                    + START_POINTS_WEIGHT * start_points_loss)

            metrics = {
                "loss":   float(loss),
                "recon":  float(recon_loss),
                "kl":     float(kl_loss),
                "length": float(length_loss),
                "start":  float(start_points_loss),
            }

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        scheduler.step()

        if epoch % LOG_EVERY == 0:
            print({"epoch": epoch, "kl_weight": round(kl_weight, 6), "lr": scheduler.get_last_lr()[0], **metrics})
            wandb.log({"epoch": epoch, "kl_weight": kl_weight, "lr": scheduler.get_last_lr()[0], **metrics}, step=epoch)
        if epoch % SAVE_EVERY == 0:
            save_checkpoint(model, optimizer, dataset, epoch)
        if epoch % SANITY_EVERY == 0:
            sanity_check(model, val_set, epoch, SAVE_DIR)
            if SANITY_SYNTHETIC and ghd_vae is not None:
                sanity_synthetic(model, ghd_vae, ghd_mean, ghd_std, ghd_input_dim, epoch, SAVE_DIR)

    wandb.finish()


if __name__ == "__main__":
    main()


"""
conda activate new
python scripts/train/train_branch_transformer_claydoll.py

"""
