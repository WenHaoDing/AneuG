"""
Train the endcap predictor (models/endcap_predictor.py) on
dataset/processed_endcaps (see dataset/preprocess_endcaps.py).

Predicts, per branch slot, a point on the mesh surface where a parent vessel
should be capped (soft-argmax over real mesh-vertex positions, weighted by a
per-vertex classifier head) and the tangent direction there — see
models/endcap_predictor.py's module docstring for the full design rationale
(GCN stem + GPSConv, no SAGPooling/FPS/attention modules, decoupled
localization vs. tangent heads, soft-argmax grounding, graph-distance patch
auxiliary loss).

Three losses, no self-supervised/synthetic term (same reasoning as the
claydoll scripts: nothing here depends on real-vs-generated conditioning
mismatch, since the model only ever sees mesh geometry reconstructed from phi
— identical at train and generation time):
  - endpoint_loss: masked MSE, predicted vs. real endpoint (mm^2).
  - tangent_loss:  masked (1 - cosine similarity), predicted vs. real tangent.
  - patch_loss:    cross-entropy of loc_head's per-vertex distribution against
                   a uniform target over each branch's in_patch vertices —
                   auxiliary signal so that distribution has something to
                   learn from directly, not just indirectly through the
                   endpoint regression loss.

Random SO(3) rotation augmentation (models.endcap_predictor.RandomSO3Rotation)
is applied here, in prepare_batch, not inside the model — rotating only the
mesh input would leave it inconsistent with the ground-truth endpoint/tangent
targets (only ever known in the canonical frame), so the SAME per-sample
rotations are applied to both the mesh and those targets before either
reaches the model, keeping model.forward() itself augmentation-agnostic.

conda activate new
python scripts/train/train_endcap_predictor.py
python scripts/train/train_endcap_predictor.py --device cuda:0
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

from dataset.endcap_dataset import EndcapDataset, collate_endcaps
from models.endcap_predictor import EndcapPredictor, RandomSO3Rotation
from models.multi_canonical_ghd_reconstruct import MultiCanonicalGHDReconstruct
from utils.generate_synthetic import load_ghd_vae


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default=None, help="e.g. cuda:0, cuda:1, cpu (default: cuda:1).")
    return p.parse_args() if __name__ == "__main__" else p.parse_args([])


_args = _parse_args()

PROCESSED_ROOT = ROOT / "runtime" / "dataset" / "processed_endcaps"
CANONICAL_ROOT = ROOT / "dataset" / "canonical"

DEVICE     = torch.device(_args.device or ("cuda:1" if torch.cuda.is_available() else "cpu"))
EPOCHS     = 2000
BATCH_SIZE = 8   # GPSConv's global attention is O(N^2) per graph, up to N=4143 (Bifurcated) —
                 # a shared, tight GPU OOM'd at batch_size=32; reduced 4x for headroom.
LR         = 1e-3
LR_MIN     = 1e-5

MAX_BRANCHES = 3
NUM_TYPES    = 3
HIDDEN       = 32
STEM_LAYERS  = 2   # plain GCNConv layers before GPS attention (see models/endcap_predictor.py)
GPS_LAYERS   = 2   # GPSConv layers (local MPNN + global self-attention), on top of the stem
GPS_HEADS    = 4
USE_NORMALS  = True   # default on for this model — see MultiCanonicalGHDReconstruct.to_pyg_batch
ROTATION_AUGMENT = True   # random SO(3) rotation of mesh + targets together, applied in prepare_batch

TANGENT_WEIGHT = 1.0
PATCH_WEIGHT   = 0.1   # auxiliary — the main regression loss carries precision, this just helps it get there

_VARIANT = "endcap_predictor"
SAVE_DIR = ROOT / "runtime" / "tr_checkpoints" / "v2_1" / _VARIANT / f"h{HIDDEN}_gps{GPS_LAYERS}_tw{TANGENT_WEIGHT:g}_pw{PATCH_WEIGHT:g}"

# Frozen stage-1 GHD VAE — only used to sample synthetic phi for the fully-
# generative sanity panel. Post type2->type1 merge (confirmed via its own
# saved hparams["merge_type2_into_type1"] == True).
GHD_VAE_CKPT = ROOT / "runtime" / "tr_checkpoints" / "v2_1" / "stage1" / "ghd_vae_h512_z16_kl0.5" / "epoch_05000.pth"

SAVE_EVERY   = 200
LOG_EVERY    = 10
SANITY_EVERY = 200
SANITY_SYNTHETIC = True

WANDB_PROJECT = "AneuGv2_endcap"
WANDB_RUN_NAME = SAVE_DIR.name


def model_args():
    return {
        "max_branches": MAX_BRANCHES,
        "hidden": HIDDEN,
        "stem_layers": STEM_LAYERS,
        "gps_layers": GPS_LAYERS,
        "gps_heads": GPS_HEADS,
        "use_normals": USE_NORMALS,
    }


def all_hparams():
    return {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "lr_min": LR_MIN,
        "max_branches": MAX_BRANCHES,
        "num_types": NUM_TYPES,
        "hidden": HIDDEN,
        "stem_layers": STEM_LAYERS,
        "gps_layers": GPS_LAYERS,
        "gps_heads": GPS_HEADS,
        "use_normals": USE_NORMALS,
        "rotation_augment": ROTATION_AUGMENT,
        "tangent_weight": TANGENT_WEIGHT,
        "patch_weight": PATCH_WEIGHT,
        "ghd_vae_ckpt": str(GHD_VAE_CKPT),
        "save_every": SAVE_EVERY,
        "log_every": LOG_EVERY,
        "sanity_every": SANITY_EVERY,
        "sanity_synthetic": SANITY_SYNTHETIC,
    }


def save_checkpoint(model, optimizer, epoch):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": model_args(),
            "hparams": all_hparams(),
        },
        SAVE_DIR / f"epoch_{epoch:05d}.pth",
    )


def prepare_batch(batch, multi_recon, augment):
    """Builds the PyG mesh batch here (not inside the model — see module
    docstring) and, when augmenting, rotates it AND the ground-truth
    endpoint/tangent targets by the exact same per-sample rotations, so the
    model (which stays augmentation-agnostic) is always training against a
    self-consistent (mesh, targets) pair, whichever frame that happens to be in."""
    phi           = batch["phi"].to(DEVICE)
    aneurysm_type = batch["aneurysm_type"].to(DEVICE)
    endpoints     = batch["endpoints"].to(DEVICE)
    tangents      = batch["tangents"].to(DEVICE)

    pyg_batch = multi_recon.to_pyg_batch(phi, aneurysm_type, point_std=None, include_normals=USE_NORMALS)

    if augment:
        rot = RandomSO3Rotation.sample(phi.size(0), phi.dtype, DEVICE)   # [B, 3, 3]
        pyg_batch = RandomSO3Rotation.apply_to_data(pyg_batch, rot)
        endpoints = RandomSO3Rotation.apply_to_points(rot, endpoints)
        tangents  = RandomSO3Rotation.apply_to_points(rot, tangents)

    return {
        "pyg_batch":     pyg_batch,
        "endpoints":     endpoints,
        "tangents":      tangents,
        "branch_mask":   batch["branch_mask"].to(DEVICE),
        "in_patch":      batch["in_patch"].to(DEVICE),
        "node_batch":    batch["node_batch"].to(DEVICE),
    }


def plot_loss_curves(history, save_dir):
    """Overwrites SAVE_DIR/loss_curves.png each call with the full history so
    far — lets you check progress on a run without opening wandb. Log-scale
    y-axis since endpoint/tangent/patch losses span orders of magnitude over
    training."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    keys = ["loss", "endpoint", "tangent", "patch"]
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4))
    epochs = history["epoch"]
    for ax, key in zip(axes, keys):
        ax.plot(epochs, history[key])
        ax.set_yscale("log")
        ax.set_title(key)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3, which="both")
    fig.suptitle(SAVE_DIR.name)
    fig.tight_layout()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / "loss_curves.png", dpi=130)
    plt.close(fig)


@torch.no_grad()
def sanity_check(model, dataset, epoch, save_dir, n=8, ncols=4):
    """Ground truth (black) vs. prediction (dashed color) on a few real
    (TRAINING — no held-out split, see main()) cases — the gap between a
    black and dashed marker visualizes endpoint_loss directly; yellow arrows
    are the tangents (real vs. predicted). Real-data fit is a sanity check on
    whether the model is learning at all; sanity_synthetic (generated shapes)
    is the one that actually matters for this model's purpose."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from dataset.preprocess_ImperialNHS import _reconstruct_ghd_numpy, _set_axes_equal

    model.eval()
    n = min(n, len(dataset))
    nrows = (n + ncols - 1) // ncols
    fig = plt.figure(figsize=(4 * ncols, 4 * nrows))
    bright = ["#e6194B", "#3cb44b", "#4363d8"]

    for i in range(n):
        item = dataset[i]
        phi = item["phi"].unsqueeze(0).to(DEVICE)
        atype_t = item["aneurysm_type"].unsqueeze(0).to(DEVICE)
        endpoint_pred, tangent_pred, _, _ = model(phi, atype_t)
        endpoint_pred = endpoint_pred[0].cpu().numpy()
        tangent_pred = tangent_pred[0].cpu().numpy()

        atype = int(item["aneurysm_type"])
        verts, faces = _reconstruct_ghd_numpy(
            {"aneurysm_type": atype, "ghd": {"phi": item["phi"].numpy()}}, denormalize_shape=True)

        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces,
                        color="lightgray", edgecolor="none", alpha=0.2)
        all_pts = [verts]
        for b in range(model.max_branches):
            if not item["branch_mask"][b]:
                continue
            c = bright[b % len(bright)]
            ep_real = item["endpoints"][b].numpy()
            tg_real = item["tangents"][b].numpy()
            ep_pred = endpoint_pred[b]
            tg_pred = tangent_pred[b]

            ax.scatter(*ep_real, s=60, marker="o", color="black")
            ax.scatter(*ep_pred, s=60, marker="x", color=c)
            ax.plot(*zip(ep_real, ep_pred), color=c, lw=1.0, ls=":")
            ax.quiver(*ep_real, *(tg_real * 2.0), color="black", linewidth=1.5, arrow_length_ratio=0.3)
            ax.quiver(*ep_pred, *(tg_pred * 2.0), color="yellow", linewidth=1.5, arrow_length_ratio=0.3)
            all_pts += [ep_real[None], ep_pred[None]]

        ax.set_axis_off()
        ax.set_title(f"{item['case']} (type {atype})", fontsize=8)
        _set_axes_equal(ax, *all_pts)

    fig.suptitle(f"epoch {epoch}  (black=real, colored=predicted)", fontsize=10)
    fig.tight_layout()
    save_dir = Path(save_dir) / "sanity"; save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / f"epoch_{epoch:05d}.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    model.train()
    return out


@torch.no_grad()
def sanity_synthetic(model, ghd_vae, ghd_mean, ghd_std, ghd_input_dim,
                     epoch, save_dir, n=8, ncols=4, seed=0):
    """Fully-generative sanity: synthetic GHD phi -> predicted endpoints/
    tangents. No ground truth exists for these, so this just checks the
    prediction looks like a sane vessel-cap location — the real point of this
    whole redesign was making that hold for GENERATED, not just real, shapes."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from dataset.preprocess_ImperialNHS import _reconstruct_ghd_numpy, _set_axes_equal

    model.eval()
    g = torch.Generator(device="cpu").manual_seed(seed)
    # Type 2 is merged into type 1 everywhere in this pipeline (see
    # EndcapDataset.__getitem__) — never sample it here either, regardless of
    # whether the loaded GHD_VAE_CKPT happens to have its own type-2 row
    # trained (an old, pre-merge checkpoint would).
    types = torch.randint(0, NUM_TYPES - 1, (n,), generator=g).to(DEVICE)
    z_ghd = torch.randn(n, ghd_vae.latent_dim, generator=g).to(DEVICE)
    ghd_n, _ = ghd_vae.decode(z_ghd, types)
    phi = (ghd_n * ghd_std[:, :ghd_input_dim] + ghd_mean[:, :ghd_input_dim]).reshape(n, -1, 3)

    endpoint_pred, tangent_pred, _, _ = model(phi, types)
    endpoint_pred, tangent_pred = endpoint_pred.cpu().numpy(), tangent_pred.cpu().numpy()

    bright = ["#e6194B", "#3cb44b", "#4363d8"]
    nrows = (n + ncols - 1) // ncols
    fig = plt.figure(figsize=(4 * ncols, 4 * nrows))
    for i in range(n):
        atype = int(types[i])
        verts, faces = _reconstruct_ghd_numpy(
            {"aneurysm_type": atype, "ghd": {"phi": phi[i].cpu().numpy()}}, denormalize_shape=True)
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces,
                        color="lightgray", edgecolor="none", alpha=0.2)
        all_pts = [verts]
        for b in range(model.max_branches):
            c = bright[b % len(bright)]
            ep, tg = endpoint_pred[i, b], tangent_pred[i, b]
            ax.scatter(*ep, s=60, marker="x", color=c)
            ax.quiver(*ep, *(tg * 2.0), color="yellow", linewidth=1.5, arrow_length_ratio=0.3)
            all_pts.append(ep[None])
        ax.set_axis_off()
        ax.set_title(f"type {atype}", fontsize=8)
        _set_axes_equal(ax, *all_pts)

    fig.suptitle(f"epoch {epoch}  synthetic generation, predicted endcaps", fontsize=10)
    fig.tight_layout()
    save_dir = Path(save_dir) / "sanity"; save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / f"sanity_synth_epoch_{epoch:05d}.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    model.train()
    return out


def main():
    wandb.init(project=WANDB_PROJECT, name=WANDB_RUN_NAME, config=all_hparams())

    # No held-out split — this model's actual target is unseen SYNTHETIC
    # shapes (sanity_synthetic), not held-out real cases, so a real-data val
    # split wouldn't measure the thing that matters; use all real data to train.
    dataset = EndcapDataset(PROCESSED_ROOT, max_branches=MAX_BRANCHES)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=4, drop_last=True, collate_fn=collate_endcaps)

    multi_recon = MultiCanonicalGHDReconstruct(CANONICAL_ROOT, device=DEVICE)
    model = EndcapPredictor(multi_recon, **model_args()).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR_MIN)

    ghd_vae = ghd_mean = ghd_std = ghd_input_dim = None
    if SANITY_SYNTHETIC:
        ghd_vae, ghd_mean, ghd_std, ghd_input_dim = load_ghd_vae(GHD_VAE_CKPT, DEVICE)
        for p in ghd_vae.parameters():
            p.requires_grad_(False)
        print(f"Loaded GHD VAE: {GHD_VAE_CKPT.name}")

    print(f"{_VARIANT}: {len(dataset)} samples (no held-out split)")

    history = {"epoch": [], "loss": [], "endpoint": [], "tangent": [], "patch": []}

    for epoch in range(EPOCHS + 1):
        model.train()
        metrics = {}
        for batch in loader:
            data = prepare_batch(batch, multi_recon, augment=ROTATION_AUGMENT)
            endpoint_pred, tangent_pred, loc_weights, token_mask = model(pyg_batch=data["pyg_batch"])
            endpoint_loss, tangent_loss, patch_loss = model.get_loss(
                endpoint_pred, tangent_pred, loc_weights, token_mask,
                data["endpoints"], data["tangents"], data["branch_mask"],
                data["in_patch"], data["node_batch"])
            loss = endpoint_loss + TANGENT_WEIGHT * tangent_loss + PATCH_WEIGHT * patch_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            metrics = {
                "loss": float(loss), "endpoint": float(endpoint_loss),
                "tangent": float(tangent_loss), "patch": float(patch_loss),
            }

        scheduler.step()
        if epoch % LOG_EVERY == 0:
            print({"epoch": epoch, "lr": round(scheduler.get_last_lr()[0], 6), **metrics})
            wandb.log({"epoch": epoch, "lr": scheduler.get_last_lr()[0], **metrics}, step=epoch)
            history["epoch"].append(epoch)
            for key in ("loss", "endpoint", "tangent", "patch"):
                history[key].append(metrics[key])
            plot_loss_curves(history, SAVE_DIR)
        if epoch % SAVE_EVERY == 0:
            save_checkpoint(model, optimizer, epoch)
        if epoch % SANITY_EVERY == 0:
            sanity_check(model, dataset, epoch, SAVE_DIR)
            if SANITY_SYNTHETIC and ghd_vae is not None:
                sanity_synthetic(model, ghd_vae, ghd_mean, ghd_std, ghd_input_dim, epoch, SAVE_DIR)

    wandb.finish()


if __name__ == "__main__":
    main()


"""
conda activate new
python scripts/train/train_endcap_predictor.py
"""
