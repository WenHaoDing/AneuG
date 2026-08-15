import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path = [path for path in sys.path if Path(path or ".").resolve() != ROOT]
sys.path.insert(0, str(ROOT))

from dataset.ghd_datasets import ProcessedGHDDataset
from dataset.preprocess_ImperialNHS import get_ghd_reconstruct
from models.vae_models import KL_divergence
from models.ghd_vae import TypeConditionalVAE


PROCESSED_ROOT = ROOT / "runtime" / "dataset" / "processed"
SAVE_DIR = ROOT / "checkpoints" / "v2" / "ghd_vae_tune2"

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
EPOCHS = 3000            # baseline 10000
BATCH_SIZE = 128
LR = 1e-3                # baseline 2e-3
HIDDEN_DIM = 512
LATENT_DIM = 108
WITH_SCALE = True
KL_WEIGHT = 1e-3         # baseline 2.5e-4
SCALE_WEIGHT = 0.25
MESH_WEIGHT = 50.0       # baseline 100.0, tune1 10.0 (regressed); tune2 = 50.0
SAVE_EVERY = 500
LOG_EVERY = 10
NUM_TYPES = 3
TYPE_CONDITION_DIM = 8


def save_checkpoint(model, optimizer, dataset, epoch):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
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


def plot_losses(log_path, plot_path):
    epochs, losses = [], {"loss": [], "recon": [], "mesh": [], "kl": [], "scale": []}
    with open(log_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            for key in losses:
                losses[key].append(float(row[key]))
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, values in losses.items():
        ax.plot(epochs, values, label=key, linewidth=1.2)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (log scale)")
    ax.set_title(f"VAE tune2 | mesh={MESH_WEIGHT}, kl={KL_WEIGHT}, lr={LR}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    log_path = SAVE_DIR / "loss_log.csv"
    plot_path = SAVE_DIR / "loss_curves.png"

    dataset = ProcessedGHDDataset(PROCESSED_ROOT, withscale=WITH_SCALE, normalize=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=False)
    mean, std = dataset.get_mean_std()

    model = TypeConditionalVAE(
        dataset.get_dim(),
        HIDDEN_DIM,
        LATENT_DIM,
        withscale=WITH_SCALE,
        num_types=NUM_TYPES,
        condition_dim=TYPE_CONDITION_DIM,
    ).to(DEVICE)
    ghd_reconstruct = get_ghd_reconstruct(device=DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)

    csv_file = open(log_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "loss", "recon", "mesh", "kl", "scale"])
    csv_file.flush()

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

        csv_writer.writerow([epoch, metrics["loss"], metrics["recon"], metrics["mesh"], metrics["kl"], metrics["scale"]])
        csv_file.flush()

        if epoch % LOG_EVERY == 0:
            print({"epoch": epoch, **metrics}, flush=True)
        if epoch % SAVE_EVERY == 0:
            save_checkpoint(model, optimizer, dataset, epoch)

    csv_file.close()
    plot_losses(log_path, plot_path)
    print(f"Done. Logs: {log_path}", flush=True)
    print(f"Plot: {plot_path}", flush=True)


if __name__ == "__main__":
    main()


"""
conda activate new
python scripts/experimental/train_ghd_vae_tune2.py
"""
