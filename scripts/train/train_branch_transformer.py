import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path = [path for path in sys.path if Path(path or ".").resolve() != ROOT]
sys.path.insert(0, str(ROOT))

from dataset.skeleton_dataset import VesselSkeletonDataset
from models.branch_transformer import MultiBranchVAE, MultiBranchVAE_GCNConditioner, MultiBranchConditions
from visualization.sanity_check import sanity_check
from models.multi_canonical_ghd_reconstruct import MultiCanonicalGHDReconstruct

# ── variant ───────────────────────────────────────────────────────────────────
USE_GCN        = True   # True → MultiBranchVAE_GCNConditioner (GCN mesh encoder)
                          # False → MultiBranchVAE (flat MLP on phi tokens)

PROCESSED_ROOT = ROOT / "dataset" / "processed"
CANONICAL_ROOT = ROOT / "dataset" / "canonical"   # only used when USE_GCN=True

DEVICE    = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
EPOCHS    = 2000
BATCH_SIZE = 256
LR        = 1e-3

HIDDEN_DIM            = 64
LATENT_DIM            = 8
NUM_LAYERS            = 4
NHEAD                 = 4
GHD_DIM               = 16
MAX_BRANCHES          = 3
MAX_POINTS_PER_BRANCH = 128   # max_local_points = MAX_POINTS_PER_BRANCH - 1 = 127
NUM_TYPES             = 3   # conditioning labels 0/1/2 (0=bifurcated, 1/2=sidewall); must match GHD VAE

# GCN encoder (only used when USE_GCN=True)
GCN_HIDDEN     = 32
GCN_POOL_RATIO = 0.5

KL_BASE          = 3e-4
KL_MULT          = 1     # integer multiplier (e.g. 1, 2, 4); goes into SAVE_DIR name
KL_WEIGHT        = KL_BASE * KL_MULT
KL_ANNEAL_EPOCHS = 200
LEN_WEIGHT       = 0.1
SYN_DIR_WEIGHT   = 0.0   # synthetic GHD direction loss weight (0 = disabled)
B_SYN            = 8     # synthetic batch size; must be divisible by NUM_TYPES

# Curvature-weighted reconstruction: weight each point by local curvature so
# bends/hooks count more. USE_POINT_WEIGHTS=False → plain (uniform) recon loss.
USE_POINT_WEIGHTS = True
CURVATURE_WEIGHT  = 0.25   # dataset curvature-boost strength (0 = uniform weights)

_VARIANT = "branch_transformer_gcn" if USE_GCN else "branch_transformer"
SAVE_DIR = ROOT / "tr_checkpoints" / "v2" / "branch_transformer" / f"{_VARIANT}_h{HIDDEN_DIM}_z{LATENT_DIM}_kl{KL_MULT}_weighted_weak"

LR_MIN = 1e-5   # cosine annealing floor

# pre-trained GHD VAE (TypeConditionalVAE from train_ghd_vae.py)
GHD_VAE_CKPT    = ROOT / "tr_checkpoints" / "v2" / "ghd_vae" / "epoch_10000.pth"
GHD_INPUT_DIM   = 432   # 144 * 3
GHD_HIDDEN_DIM  = 512
GHD_LATENT_DIM  = 108
GHD_COND_DIM    = 8

SAVE_EVERY   = 200
LOG_EVERY    = 10
SANITY_EVERY = 500
VAL_FRACTION = 0.1
VAL_SEED     = 42


def load_ghd_vae(ckpt_path, device):
    """Load pre-trained TypeConditionalVAE GHD model. Returned in eval/no-grad mode."""
    from models.ghd_vae import TypeConditionalVAE
    ckpt  = torch.load(ckpt_path, map_location=device)
    model = TypeConditionalVAE(
        input_dim     = GHD_INPUT_DIM,
        hidden_dim    = GHD_HIDDEN_DIM,
        latent_dim    = GHD_LATENT_DIM,
        withscale     = ckpt.get("withscale", True),
        num_types     = NUM_TYPES,
        condition_dim = GHD_COND_DIM,
    )
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    mean = ckpt["mean"].to(device)   # [1, 433]
    std  = ckpt["std"].to(device)    # [1, 433]
    return model, mean, std


def compute_synthetic_dir_loss(model, ghd_vae, ghd_mean, ghd_std, multi_recon: MultiCanonicalGHDReconstruct):
    """Direction loss on branches generated from synthetic GHD.

    1. Sample random phi from the frozen GHD VAE (no grad).
    2. Compute branch conditions (start_points, directions) from mesh openings (no grad).
    3. Forward the branch decoder with z ~ N(0,I) and dummy (zero) local_points.
    4. Return direction loss — gradients flow through the branch transformer only.
    """
    assert B_SYN % NUM_TYPES == 0, f"B_SYN ({B_SYN}) must be divisible by NUM_TYPES ({NUM_TYPES})"
    n_per_type = B_SYN // NUM_TYPES

    with torch.no_grad():
        types    = torch.arange(NUM_TYPES, device=DEVICE).repeat_interleave(n_per_type)
        z_ghd    = torch.randn(B_SYN, ghd_vae.latent_dim, device=DEVICE)
        ghd_n, _ = ghd_vae.decode(z_ghd, types)                           # [B_SYN, 432]
        phi      = (ghd_n * ghd_std[:, :432] + ghd_mean[:, :432]).reshape(B_SYN, -1, 3)

        starts_l, dirs_l, mask_l = [], [], []
        for atype in range(NUM_TYPES):
            sl = slice(atype * n_per_type, (atype + 1) * n_per_type)
            s, d, m = multi_recon.compute_branch_conditions(phi[sl], atype, MAX_BRANCHES)
            starts_l.append(s); dirs_l.append(d); mask_l.append(m)
        starts = torch.cat(starts_l)
        dirs   = torch.cat(dirs_l)
        b_mask = torch.cat(mask_l)

    if not b_mask.any():
        return None

    ghd_embed = model.encode_phi(phi, types)                               # [B_SYN, ghd_dim]
    syn_cond  = MultiBranchConditions(
        aneurysm_type    = types,
        scale            = torch.ones(B_SYN, device=DEVICE),
        start_points     = starts,
        branch_direction = dirs,
        branch_mask      = b_mask,
        ghd_embed        = ghd_embed,
    )

    seq_len   = MAX_BRANCHES * (MAX_POINTS_PER_BRANCH - 1)
    dummy     = torch.zeros(B_SYN, seq_len, 3, device=DEVICE)
    b_ids     = torch.arange(MAX_BRANCHES, device=DEVICE).repeat_interleave(MAX_POINTS_PER_BRANCH - 1)
    tok_mask  = b_mask[:, b_ids]                                           # [B_SYN, seq_len]

    z         = torch.randn(B_SYN, model.latent_dim, device=DEVICE)
    recon, length_logit = model.decoder(z, dummy, tok_mask, syn_cond)
    syn_len = (length_logit.argmax(-1) + 1).clamp(max=MAX_POINTS_PER_BRANCH - 1)
    return model.get_dir_loss(recon, syn_len, b_mask, dirs)


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
        "token_weights": batch["token_weights"].to(DEVICE),               # [B, seq_len]
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

    if USE_GCN:
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

    ghd_vae = ghd_mean = ghd_std = None
    if SYN_DIR_WEIGHT > 0:
        ghd_vae, ghd_mean, ghd_std = load_ghd_vae(GHD_VAE_CKPT, DEVICE)
        print(f"Loaded GHD VAE from {GHD_VAE_CKPT}")

    for epoch in range(EPOCHS + 1):
        model.train()
        kl_weight = KL_WEIGHT * min(1.0, epoch / KL_ANNEAL_EPOCHS)
        metrics = {}
        for batch in loader:
            data = prepare_batch(batch)

            recon, length_logit, mu, logvar = model(
                data["local_points"], data["token_mask"], data["phi"], data["cond"],
            )
            recon_loss, kl_loss, length_loss = model.get_loss(
                recon, data["local_points"], length_logit,
                data["branch_length"], data["token_mask"], data["branch_mask"],
                mu, logvar,
                point_weights=data["token_weights"] if USE_POINT_WEIGHTS else None,
            )
            loss = recon_loss + kl_weight * kl_loss + LEN_WEIGHT * length_loss

            metrics = {
                "loss":   float(loss),
                "recon":  float(recon_loss),
                "kl":     float(kl_loss),
                "length": float(length_loss),
            }

            # if SYN_DIR_WEIGHT > 0:
            #     syn_dir = compute_synthetic_dir_loss(model, ghd_vae, ghd_mean, ghd_std, multi_recon)
            #     if syn_dir is not None:
            #         loss = loss + SYN_DIR_WEIGHT * syn_dir
            #         metrics["syn_dir"] = float(syn_dir)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

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
python scripts/train/train_branch_transformer.py

"""