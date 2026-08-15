import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path = [path for path in sys.path if Path(path or ".").resolve() != ROOT]
sys.path.insert(0, str(ROOT))

from dataset.ghd_datasets import ProcessedGHDDataset
from dataset.preprocess_ImperialNHS import get_ghd_reconstruct
from models.ghd_vqvae import ConditionalGHDVQVAE


PROCESSED_ROOT = ROOT / "runtime" / "dataset" / "processed"
SAVE_DIR = ROOT / "checkpoints" / "v2" / "ghd_vqvae"

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
EPOCHS = 10000
BATCH_SIZE = 128
LR = 2e-3
WITH_SCALE = True
SEQ_LEN = 144
CHANNELS = 3
NUM_TYPES = 3
HIDDEN_DIM = 128
EMBEDDING_DIM = 64
NUM_CODES = 128

X_WEIGHT = 1.0
SCALE_WEIGHT = 0.25
MESH_WEIGHT = 100.0
COMMITMENT_WEIGHT = 0.25
SAVE_EVERY = 1000
LOG_EVERY = 10


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
            "seq_len": SEQ_LEN,
            "channels": CHANNELS,
            "num_types": NUM_TYPES,
            "hidden_dim": HIDDEN_DIM,
            "embedding_dim": EMBEDDING_DIM,
            "num_codes": NUM_CODES,
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


def split_batch(batch):
    if WITH_SCALE:
        ghd, scale = batch[:, :-1], batch[:, -1:]
    else:
        ghd = batch
        scale = batch.new_zeros((batch.shape[0], 1))
    return ghd, ghd.view(batch.shape[0], SEQ_LEN, CHANNELS), scale


def main():
    dataset = ProcessedGHDDataset(PROCESSED_ROOT, withscale=WITH_SCALE, normalize=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=False)
    mean, std = dataset.get_mean_std()

    model = ConditionalGHDVQVAE(
        seq_len=SEQ_LEN,
        channels=CHANNELS,
        num_types=NUM_TYPES,
        hidden_dim=HIDDEN_DIM,
        embedding_dim=EMBEDDING_DIM,
        num_codes=NUM_CODES,
    ).to(DEVICE)
    ghd_reconstruct = get_ghd_reconstruct(device=DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)

    for epoch in range(EPOCHS + 1):
        model.train()
        metrics = {}
        for batch, aneurysm_type in loader:
            batch = batch.to(DEVICE)
            aneurysm_type = aneurysm_type.to(DEVICE)
            ghd, x, scale = split_batch(batch)

            out = model(x, scale, aneurysm_type)
            x_recon = out["x_recon"]
            ghd_recon = x_recon.reshape(x_recon.shape[0], -1)

            x_loss = F.mse_loss(x_recon, x)
            scale_loss = F.mse_loss(out["scale_recon"], scale)
            mesh_loss = mesh_reconstruction_loss(
                ghd,
                ghd_recon,
                aneurysm_type,
                ghd_reconstruct,
                mean,
                std,
            )
            loss = (
                X_WEIGHT * x_loss
                + SCALE_WEIGHT * scale_loss
                + MESH_WEIGHT * mesh_loss
                + out["codebook_loss"]
                + COMMITMENT_WEIGHT * out["commitment_loss"]
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metrics = {
                "loss": float(loss.detach().cpu()),
                "x": float(x_loss.detach().cpu()),
                "scale": float(scale_loss.detach().cpu()),
                "mesh": float(mesh_loss.detach().cpu()),
                "codebook": float(out["codebook_loss"].detach().cpu()),
                "commit": float(out["commitment_loss"].detach().cpu()),
                "perplexity": float(out["perplexity"].detach().cpu()),
            }

        scheduler.step()
        if epoch % LOG_EVERY == 0:
            print({"epoch": epoch, **metrics})
        if epoch % SAVE_EVERY == 0:
            save_checkpoint(model, optimizer, dataset, epoch)


if __name__ == "__main__":
    main()


"""
conda activate new
python scripts/experimental/train_ghd_vqvae.py

"""
