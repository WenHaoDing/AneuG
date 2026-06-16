import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path = [path for path in sys.path if Path(path or ".").resolve() != ROOT]
sys.path.insert(0, str(ROOT))

from dataset.skeleton_dataset import VesselSkeletonDataset
from v2.branch_transformer import MultiBranchVAE, MultiBranchVAE_GCNConditioner, MultiBranchConditions
from v2.sanity_check import sanity_check

# ── variant ───────────────────────────────────────────────────────────────────
USE_GCN        = True   # True → MultiBranchVAE_GCNConditioner (GCN mesh encoder)
                          # False → MultiBranchVAE (flat MLP on phi tokens)

PROCESSED_ROOT = ROOT / "dataset" / "processed"
CANONICAL_ROOT = ROOT / "dataset" / "canonical"   # only used when USE_GCN=True
SAVE_DIR       = ROOT / "checkpoints" / "v2" / ("branch_transformer_gcn" if USE_GCN else "branch_transformer")

DEVICE    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPOCHS    = 2000
BATCH_SIZE = 32
LR        = 1e-3

HIDDEN_DIM            = 32
LATENT_DIM            = 32
NUM_LAYERS            = 4
NHEAD                 = 4
GHD_DIM               = 16
MAX_BRANCHES          = 3
MAX_POINTS_PER_BRANCH = 128   # max_local_points = MAX_POINTS_PER_BRANCH - 1 = 127
NUM_TYPES             = 3

# GCN encoder (only used when USE_GCN=True)
GCN_HIDDEN     = 32
GCN_POOL_RATIO = 0.5

KL_WEIGHT        = 1e-3
KL_ANNEAL_EPOCHS = 200
LEN_WEIGHT       = 0.1
DIR_WEIGHT       = 0.0

LR_MIN = 1e-5   # cosine annealing floor

SAVE_EVERY   = 200
LOG_EVERY    = 10
SANITY_EVERY = 500
VAL_FRACTION = 0.1
VAL_SEED     = 42


def save_checkpoint(model, optimizer, dataset, epoch):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch":      epoch,
            "model":      model.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "point_mean": dataset.point_mean,
            "point_std":  dataset.point_std,
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
    """
    cond = MultiBranchConditions(
        aneurysm_type    = batch["aneurysm_type"].to(DEVICE),              # [B]
        scale            = batch["scale"].to(DEVICE),                       # [B]
        start_points     = batch["start_points"].to(DEVICE),               # [B, max_branches, 3]
        branch_direction = batch["branch_direction"].to(DEVICE),           # [B, max_branches, 3]
        branch_mask      = batch["branch_mask"].to(DEVICE),                # [B, max_branches]
    )
    return {
        "local_points":  batch["local_points"].flatten(1, 2).to(DEVICE),  # [B, seq_len, 3]
        "token_mask":    batch["token_mask"].to(DEVICE),                   # [B, seq_len]
        "branch_length": batch["branch_length"].to(DEVICE),               # [B, max_branches]
        "branch_mask":   batch["branch_mask"].to(DEVICE),                 # [B, max_branches]
        "phi":           batch["phi"].to(DEVICE),                          # [B, 144, 3]
        "cond":          cond,
    }


def main():
    dataset = VesselSkeletonDataset(
        PROCESSED_ROOT,
        max_branches=MAX_BRANCHES,
        max_points_per_branch=MAX_POINTS_PER_BRANCH,
    )
    n_val   = max(1, int(len(dataset) * VAL_FRACTION))
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(VAL_SEED),
    )
    loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)

    if USE_GCN:
        from v2.ghd_reconstruct import MultiCanonicalGHDReconstruct
        multi_recon = MultiCanonicalGHDReconstruct(CANONICAL_ROOT, device=DEVICE)
        model = MultiBranchVAE_GCNConditioner(
            multi_recon,
            hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM, num_layers=NUM_LAYERS, nhead=NHEAD,
            max_local_points=MAX_POINTS_PER_BRANCH - 1,
            num_types=NUM_TYPES, max_branches=MAX_BRANCHES, ghd_dim=GHD_DIM,
            gcn_hidden=GCN_HIDDEN, gcn_pool_ratio=GCN_POOL_RATIO,
        ).to(DEVICE)
    else:
        model = MultiBranchVAE(
            hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM, num_layers=NUM_LAYERS, nhead=NHEAD,
            max_local_points=MAX_POINTS_PER_BRANCH - 1,
            num_types=NUM_TYPES, max_branches=MAX_BRANCHES, ghd_dim=GHD_DIM,
        ).to(DEVICE)
    model.set_normalization_stats(dataset.point_mean, dataset.point_std)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR_MIN)

    for epoch in range(EPOCHS + 1):
        model.train()
        kl_weight = KL_WEIGHT * min(1.0, epoch / KL_ANNEAL_EPOCHS)
        metrics = {}
        for batch in loader:
            data = prepare_batch(batch)

            recon, length_logit, mu, logvar = model(
                data["local_points"], data["token_mask"], data["phi"], data["cond"],
            )
            recon_loss, kl_loss, length_loss, dir_loss = model.get_loss(
                recon, data["local_points"], length_logit,
                data["branch_length"], data["token_mask"], data["branch_mask"],
                mu, logvar, branch_direction=data["cond"].branch_direction,
            )
            loss = recon_loss + kl_weight * kl_loss + LEN_WEIGHT * length_loss + DIR_WEIGHT * dir_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metrics = {
                "loss":   float(loss),
                "recon":  float(recon_loss),
                "kl":     float(kl_loss),
                "length": float(length_loss),
                "dir":    float(dir_loss),
            }

        scheduler.step()

        if epoch % LOG_EVERY == 0:
            print({"epoch": epoch, "kl_weight": round(kl_weight, 6), "lr": scheduler.get_last_lr()[0], **metrics})
        if epoch % SAVE_EVERY == 0:
            save_checkpoint(model, optimizer, dataset, epoch)
        if epoch % SANITY_EVERY == 0:
            sanity_check(model, val_set, epoch, SAVE_DIR, DEVICE,
                         MultiBranchConditions, MAX_BRANCHES, MAX_POINTS_PER_BRANCH - 1)


if __name__ == "__main__":
    main()


"""
conda activate new
python v2/train_branch_transformer.py

"""
