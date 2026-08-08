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
from dataset.preprocess_ImperialNHS import get_ghd_reconstruct
from models.vae_models import KL_divergence
from models.ghd_vae import TypeConditionalVAE


PROCESSED_ROOT = ROOT / "dataset" / "processed"

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
EPOCHS = 5000
BATCH_SIZE = 128
LR = 1e-3
HIDDEN_DIM = 512
LATENT_DIM = 64

KL_BASE = 2.5e-4
KL_MULT = 2  # integer multiplier (e.g. 1, 2, 4); goes into SAVE_DIR name
KL_WEIGHT = KL_BASE * KL_MULT

SAVE_DIR = ROOT / "tr_checkpoints" / "v2" / "stage1"/ f"ghd_vae_h{HIDDEN_DIM}_z{LATENT_DIM}_kl{KL_MULT}"
WITH_SCALE = True
SCALE_WEIGHT = 0.25
MESH_WEIGHT = 100.0
SAVE_EVERY = 1000
LOG_EVERY = 10
NUM_TYPES = 3
TYPE_CONDITION_DIM = 8

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
        "with_scale": WITH_SCALE,
        "scale_weight": SCALE_WEIGHT,
        "mesh_weight": MESH_WEIGHT,
        "save_every": SAVE_EVERY,
        "log_every": LOG_EVERY,
        "num_types": NUM_TYPES,
        "type_condition_dim": TYPE_CONDITION_DIM,
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
            "aneurysm_type": torch.stack(dataset.aneurysm_type),
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

    for epoch in range(EPOCHS + 1):
        model.train()
        metrics = {}
        for batch, aneurysm_type in loader:
            batch = batch.to(DEVICE)
            aneurysm_type = aneurysm_type.to(DEVICE)
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
            loss = recon_loss + MESH_WEIGHT * mesh_loss + KL_WEIGHT * kl_loss + SCALE_WEIGHT * scale_loss

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
            print({"epoch": epoch, **metrics})
            wandb.log({"epoch": epoch, "lr": scheduler.get_last_lr()[0], **metrics}, step=epoch)
        if epoch % SAVE_EVERY == 0:
            save_checkpoint(model, optimizer, dataset, epoch)

    wandb.finish()


if __name__ == "__main__":
    main()


"""
conda activate new
python scripts/train/train_ghd_vae.py

"""
