import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from pytorch3d.ops import cot_laplacian

ROOT = Path(__file__).resolve().parents[1]
sys.path = [path for path in sys.path if Path(path or ".").resolve() != ROOT]
sys.path.insert(0, str(ROOT))

from dataset.ghd_datasets import ProcessedGHDDataset
from dataset.preprocess import get_ghd_reconstruct
from models.vae_models import KL_divergence
from v2_trial.vae_models import TypeConditionalVAE


PROCESSED_ROOT = ROOT / "dataset" / "processed"
SAVE_DIR = ROOT / "checkpoints" / "v2_trial" / "ghd_vae_tune6"

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
EPOCHS = 3000
BATCH_SIZE = 128
LR = 2e-3
HIDDEN_DIM = 512
LATENT_DIM = 108         # reverted from tune5; Fix 2B abandoned
WITH_SCALE = True
KL_WEIGHT = 2.5e-4
SCALE_WEIGHT = 0.25
MESH_WEIGHT = 100.0
LAP_WEIGHT = 1.0        # Fix 2C: Laplacian curvature-matching weight
SAVE_EVERY = 500
LOG_EVERY = 10
NUM_TYPES = 3
TYPE_CONDITION_DIM = 32  # Fix 1A (kept)
TYPE_WEIGHTS = [1.0, 1.5, 4.0]  # Fix 1C (kept)


def save_checkpoint(model, optimizer, dataset, epoch):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
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
        "type_weights": TYPE_WEIGHTS,
        "lap_weight": LAP_WEIGHT,
    }, SAVE_DIR / f"epoch_{epoch:05d}.pth")


def precompute_canonical_laplacians(ghd_reconstruct, types, device):
    """One-shot cotangent Laplacian per canonical type, kept dense on GPU."""
    laps = {}
    for t in types:
        reconstructor = ghd_reconstruct.get(t)
        verts = reconstructor.canonical_Meshes.verts_packed()
        faces = reconstructor.canonical_Meshes.faces_packed()
        L, _ = cot_laplacian(verts, faces)
        laps[t] = L.to_dense().to(device)
        mb = laps[t].element_size() * laps[t].numel() / 1e6
        print(f"  type {t} Laplacian: shape={tuple(laps[t].shape)}, mem~{mb:.1f} MB", flush=True)
    return laps


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


def laplacian_diff_loss(ghd, ghd_recon, aneurysm_type, ghd_reconstruct, laplacians, mean, std):
    total = ghd.new_tensor(0.0)
    count = 0
    for type_id in aneurysm_type.unique(sorted=True):
        mask = aneurysm_type == type_id
        if not mask.any():
            continue
        reconstructor = ghd_reconstruct.get(int(type_id.item()))
        L = laplacians[int(type_id.item())]
        real_meshes = reconstructor.ghd_forward_as_Meshes(ghd[mask], mean=mean, std=std)
        recon_meshes = reconstructor.ghd_forward_as_Meshes(ghd_recon[mask], mean=mean, std=std)
        v_real = real_meshes.verts_padded()
        v_recon = recon_meshes.verts_padded()
        lap_real = torch.einsum("ij,bjk->bik", L, v_real)
        lap_recon = torch.einsum("ij,bjk->bik", L, v_recon)
        total = total + F.mse_loss(lap_recon, lap_real, reduction="mean") * int(mask.sum())
        count += int(mask.sum())
    return total / max(count, 1)


def plot_losses(log_path, plot_path):
    epochs, losses = [], {"loss": [], "recon": [], "mesh": [], "lap": [], "kl": [], "scale": []}
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
    ax.set_title(f"VAE tune6 (Fix 1A + 1C + 2C, lap_w={LAP_WEIGHT}) | mesh={MESH_WEIGHT}, kl={KL_WEIGHT}, lr={LR}")
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
    sample_weights = torch.tensor(
        [TYPE_WEIGHTS[int(t)] for t in dataset.aneurysm_type], dtype=torch.float32
    )
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(dataset), replacement=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, drop_last=False)
    counts = [int((torch.stack(dataset.aneurysm_type) == t).sum()) for t in range(NUM_TYPES)]
    print(f"Fix 1C: TYPE_WEIGHTS={TYPE_WEIGHTS}, dataset size={len(dataset)}, raw counts per type: {counts}", flush=True)
    mean, std = dataset.get_mean_std()

    model = TypeConditionalVAE(
        dataset.get_dim(), HIDDEN_DIM, LATENT_DIM,
        withscale=WITH_SCALE, num_types=NUM_TYPES, condition_dim=TYPE_CONDITION_DIM,
    ).to(DEVICE)
    ghd_reconstruct = get_ghd_reconstruct(device=DEVICE)

    print(f"Fix 2C: precomputing canonical Laplacians for types {list(range(NUM_TYPES))}...", flush=True)
    laplacians = precompute_canonical_laplacians(ghd_reconstruct, list(range(NUM_TYPES)), DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)

    csv_file = open(log_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "loss", "recon", "mesh", "lap", "kl", "scale"])
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
            mesh_loss = mesh_reconstruction_loss(ghd, ghd_recon, aneurysm_type, ghd_reconstruct, mean, std)
            lap_loss = laplacian_diff_loss(ghd, ghd_recon, aneurysm_type, ghd_reconstruct, laplacians, mean, std)
            kl_loss = KL_divergence(mu, logvar)
            loss = (recon_loss + MESH_WEIGHT * mesh_loss + LAP_WEIGHT * lap_loss
                    + KL_WEIGHT * kl_loss + SCALE_WEIGHT * scale_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metrics = {
                "loss": float(loss.detach().cpu()),
                "recon": float(recon_loss.detach().cpu()),
                "mesh": float(mesh_loss.detach().cpu()),
                "lap": float(lap_loss.detach().cpu()),
                "kl": float(kl_loss.detach().cpu()),
                "scale": float(scale_loss.detach().cpu()),
            }

        scheduler.step()

        csv_writer.writerow([epoch, metrics["loss"], metrics["recon"], metrics["mesh"],
                             metrics["lap"], metrics["kl"], metrics["scale"]])
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
python v2_trial/train_ghd_vae_tune6.py
"""
