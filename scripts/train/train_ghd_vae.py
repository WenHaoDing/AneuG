import argparse
import shutil
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path = [path for path in sys.path if Path(path or ".").resolve() != ROOT]
sys.path.insert(0, str(ROOT))

from dataset.ghd_datasets import ProcessedGHDDataset
from dataset.preprocess_ImperialNHS import get_ghd_reconstruct, _set_axes_equal
from models.vae_models import KL_divergence
from models.ghd_vae import TypeConditionalVAE


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--kl-mult", type=float, default=2.0,
                   help="KL_WEIGHT = KL_BASE * kl_mult; goes into SAVE_DIR name.")
    p.add_argument("--kl-anneal-epochs", type=int, default=0,
                   help="Ramp KL weight linearly over this many epochs (0 = full strength from epoch 0).")
    p.add_argument("--save-root", default=None,
                   help="Root under which SAVE_DIR = <save-root>/stage1/ghd_vae_h..._z..._kl... is built "
                        "(default: tr_checkpoints/v2).")
    p.add_argument("--device", default=None, help="e.g. cuda:0, cuda:1, cpu (default: cuda:1).")
    # Only parse real argv when run as a script — importing this module for
    # smoke tests/reuse shouldn't choke on pytest/ipython's own argv.
    return p.parse_args() if __name__ == "__main__" else p.parse_args([])


_args = _parse_args()

PROCESSED_ROOT = ROOT / "dataset" / "processed"

DEVICE = torch.device(_args.device or ("cuda:1" if torch.cuda.is_available() else "cpu"))
EPOCHS = 5000
BATCH_SIZE = 64
LR = 1e-3
HIDDEN_DIM = _args.hidden_dim
LATENT_DIM = _args.latent_dim

KL_BASE = 2.5e-4
KL_MULT = _args.kl_mult   # multiplier (e.g. 0.5, 1, 2, 4); goes into SAVE_DIR name
KL_WEIGHT = KL_BASE * KL_MULT
KL_ANNEAL_EPOCHS = _args.kl_anneal_epochs   # 0 = no annealing (full KL_WEIGHT from epoch 0)

SAVE_ROOT = Path(_args.save_root) if _args.save_root else ROOT / "tr_checkpoints" / "v2"
_anneal_suffix = f"_anneal{KL_ANNEAL_EPOCHS}" if KL_ANNEAL_EPOCHS > 0 else ""
SAVE_DIR = SAVE_ROOT / "stage1" / f"ghd_vae_h{HIDDEN_DIM}_z{LATENT_DIM}_kl{KL_MULT:g}{_anneal_suffix}"
WITH_SCALE = True
SCALE_WEIGHT = 0.25
MESH_WEIGHT = 100.0
SAVE_EVERY = 1000
LOG_EVERY = 10
SANITY_EVERY = 500
NUM_TYPES = 3
TYPE_CONDITION_DIM = 8

# Types 1 and 2 both resolve to the same canonical reconstructor
# (MultiCanonicalGHDReconstruct.specs: 1 and 2 -> dataset/canonical/Sidewall) —
# they're anatomically the same "sidewall" template, just two different real
# case sub-populations. Treat them as one conditioning class by default so the
# type embedding isn't needlessly split across two rows for the same shape.
# NUM_TYPES stays 3 (unused row 2) rather than 2, to keep checkpoints
# compatible with downstream scripts that hardcode num_types=3.
MERGE_TYPE2_INTO_TYPE1 = True

WANDB_PROJECT = "AneuGv2_stage1"
WANDB_RUN_NAME = SAVE_DIR.name


def all_hparams():
    """Every module-level training hyperparameter, for record-keeping in the checkpoint."""
    return {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "hidden_dim": HIDDEN_DIM,
        "latent_dim": LATENT_DIM,
        "kl_base": KL_BASE,
        "kl_mult": KL_MULT,
        "kl_weight": KL_WEIGHT,
        "kl_anneal_epochs": KL_ANNEAL_EPOCHS,
        "with_scale": WITH_SCALE,
        "scale_weight": SCALE_WEIGHT,
        "mesh_weight": MESH_WEIGHT,
        "save_every": SAVE_EVERY,
        "log_every": LOG_EVERY,
        "num_types": NUM_TYPES,
        "type_condition_dim": TYPE_CONDITION_DIM,
        "merge_type2_into_type1": MERGE_TYPE2_INTO_TYPE1,
    }


def model_args(dataset):
    """Kwargs needed to reconstruct TypeConditionalVAE from a checkpoint alone."""
    return {
        "input_dim": dataset.get_dim(),
        "hidden_dim": HIDDEN_DIM,
        "latent_dim": LATENT_DIM,
        "withscale": WITH_SCALE,
        "num_types": NUM_TYPES,
        "condition_dim": TYPE_CONDITION_DIM,
    }


def _merge_type2(aneurysm_type):
    if not MERGE_TYPE2_INTO_TYPE1:
        return aneurysm_type
    return torch.where(aneurysm_type == 2, torch.ones_like(aneurysm_type), aneurysm_type)


def save_checkpoint(model, optimizer, dataset, epoch):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).resolve(), SAVE_DIR / Path(__file__).name)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "mean": dataset.mean,
            "std": dataset.std,
            "cases": dataset.cases,
            # merged to match what the model was actually trained/conditioned on
            "aneurysm_type": _merge_type2(torch.stack(dataset.aneurysm_type)),
            "withscale": dataset.withscale,
            "num_types": NUM_TYPES,
            "type_condition_dim": TYPE_CONDITION_DIM,
            "args": model_args(dataset),
            "hparams": all_hparams(),
        },
        SAVE_DIR / f"epoch_{epoch:05d}.pth",
    )


def mesh_reconstruction_loss(ghd, ghd_recon, aneurysm_type, ghd_reconstruct, mean, std):
    total = ghd.new_tensor(0.0)
    count = 0
    for type_id in aneurysm_type.unique(sorted=True):
        mask = aneurysm_type == type_id
        if not mask.any():
            continue
        reconstructor = ghd_reconstruct.get(int(type_id.item()))
        real_meshes = reconstructor.ghd_forward_as_Meshes(ghd[mask], mean=mean, std=std)
        recon_meshes = reconstructor.ghd_forward_as_Meshes(ghd_recon[mask], mean=mean, std=std)
        total = total + F.mse_loss(
            recon_meshes.verts_padded(),
            real_meshes.verts_padded(),
            reduction="mean",
        ) * int(mask.sum())
        count += int(mask.sum())
    return total / max(count, 1)


@torch.no_grad()
def sanity_synthetic(model, mean, std, ghd_reconstruct, epoch, save_dir, n=16, ncols=4, seed=0):
    """Fully-generative sanity: z ~ N(0,1) -> decode -> reconstruct mesh, per
    random type. The only quality signal available right now is eyeballing
    this panel (no FID-equivalent), so it needs to exist during training —
    otherwise checking a config means a separate manual generation pass per
    checkpoint.

    Every generated mesh comes out in the same fixed canonical orientation
    (all deformations of the same template), so a fixed camera would show the
    exact same view angle in every panel — a random rotation per shape avoids
    that systematic bias and gives a better sense of overall shape quality.
    """
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.spatial.transform import Rotation

    model.eval()
    g = torch.Generator(device="cpu").manual_seed(seed)
    # type 2's embedding row is never trained when merged into type 1 (see
    # MERGE_TYPE2_INTO_TYPE1) -- sampling it here would decode a random,
    # untrained condition vector, not a meaningful "sidewall" generation.
    n_sampled_types = NUM_TYPES - 1 if MERGE_TYPE2_INTO_TYPE1 else NUM_TYPES
    types = torch.randint(0, n_sampled_types, (n,), generator=g).to(DEVICE)
    z = torch.randn(n, model.latent_dim, generator=g).to(DEVICE)
    ghd_n = model.decode(z, types, strip_scale=True) if WITH_SCALE else model.decode(z, types)

    rng = np.random.default_rng(seed)
    nrows = (n + ncols - 1) // ncols
    fig = plt.figure(figsize=(4 * ncols, 4 * nrows))
    for i in range(n):
        atype = int(types[i])
        reconstructor = ghd_reconstruct.get(atype)
        # mean/std (from dataset.get_mean_std()) are already sliced to the
        # phi-only dimension — same convention mesh_reconstruction_loss uses.
        mesh = reconstructor.ghd_forward_as_Meshes(ghd_n[i:i + 1], mean=mean, std=std)
        verts = mesh.verts_padded()[0].detach().cpu().numpy()
        faces = mesh.faces_padded()[0].detach().cpu().numpy()
        verts = verts @ Rotation.random(random_state=rng).as_matrix().T

        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces,
                        color="lightgray", edgecolor="none")
        ax.set_axis_off()
        ax.set_title(f"type {atype}", fontsize=9)
        _set_axes_equal(ax, verts)

    fig.suptitle(f"epoch {epoch}  synthetic generation (z~N(0,1))", fontsize=10)
    fig.tight_layout()
    save_dir = Path(save_dir) / "sanity"
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / f"sanity_synth_epoch_{epoch:05d}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    model.train()
    return out


def plot_loss_curves(history, save_dir):
    """Overwrites SAVE_DIR/loss_curves.png each call with the full history so
    far — lets you check progress on a run without opening wandb."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    keys = ["loss", "recon", "mesh", "kl", "scale", "kl_weight"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    epochs = history["epoch"]
    for ax, key in zip(axes.flat, keys):
        ax.plot(epochs, history[key])
        ax.set_title(key)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    fig.suptitle(SAVE_DIR.name)
    fig.tight_layout()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / "loss_curves.png", dpi=130)
    plt.close(fig)


def main():
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config=all_hparams(),
    )

    dataset = ProcessedGHDDataset(PROCESSED_ROOT, withscale=WITH_SCALE, normalize=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=False)
    mean, std = dataset.get_mean_std()

    model = TypeConditionalVAE(**model_args(dataset)).to(DEVICE)
    ghd_reconstruct = get_ghd_reconstruct(device=DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.05, total_iters=EPOCHS
    )

    history = {"epoch": [], "loss": [], "recon": [], "mesh": [], "kl": [], "scale": [], "kl_weight": []}

    for epoch in range(EPOCHS + 1):
        model.train()
        metrics = {}
        kl_weight = (KL_WEIGHT * min(1.0, epoch / KL_ANNEAL_EPOCHS)
                    if KL_ANNEAL_EPOCHS > 0 else KL_WEIGHT)
        for batch, aneurysm_type in loader:
            batch = batch.to(DEVICE)
            aneurysm_type = _merge_type2(aneurysm_type.to(DEVICE))
            if WITH_SCALE:
                ghd, scale = batch[:, :-1], batch[:, -1:]
                ghd_recon, scale_recon, mu, logvar = model(ghd, aneurysm_type, scale)
                scale_loss = F.huber_loss(scale_recon, scale, delta=2.0)
            else:
                ghd = batch
                scale = None
                ghd_recon, mu, logvar = model(ghd, aneurysm_type, scale)
                scale_loss = ghd.new_tensor(0.0)

            recon_loss = F.mse_loss(ghd_recon, ghd)
            mesh_loss = mesh_reconstruction_loss(
                ghd,
                ghd_recon,
                aneurysm_type,
                ghd_reconstruct,
                mean,
                std,
            )
            kl_loss = KL_divergence(mu, logvar)
            loss = recon_loss + MESH_WEIGHT * mesh_loss + kl_weight * kl_loss + SCALE_WEIGHT * scale_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metrics = {
                "loss": float(loss.detach().cpu()),
                "recon": float(recon_loss.detach().cpu()),
                "mesh": float(mesh_loss.detach().cpu()),
                "kl": float(kl_loss.detach().cpu()),
                "scale": float(scale_loss.detach().cpu()),
            }

        scheduler.step()
        if epoch % LOG_EVERY == 0:
            print({"epoch": epoch, "kl_weight": round(kl_weight, 6), **metrics})
            wandb.log({"epoch": epoch, "lr": scheduler.get_last_lr()[0],
                       "kl_weight": kl_weight, **metrics}, step=epoch)
            history["epoch"].append(epoch)
            history["kl_weight"].append(kl_weight)
            for key in ("loss", "recon", "mesh", "kl", "scale"):
                history[key].append(metrics[key])
            plot_loss_curves(history, SAVE_DIR)
        if epoch % SAVE_EVERY == 0:
            save_checkpoint(model, optimizer, dataset, epoch)
        if epoch % SANITY_EVERY == 0:
            sanity_synthetic(model, mean, std, ghd_reconstruct, epoch, SAVE_DIR)

    wandb.finish()


if __name__ == "__main__":
    main()


"""
conda activate new
python scripts/train/train_ghd_vae.py

"""
