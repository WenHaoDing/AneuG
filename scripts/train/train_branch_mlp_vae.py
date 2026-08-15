"""
Train the conditional MLP-VAE over per-branch Fourier coefficients
(models.branch_mlp_vae.BranchFourierVAE) on VesselSkeletonDatasetFourier.

Adapted from scripts/train/train_branch_transformer.py.

conda activate new
python scripts/train/train_branch_mlp_vae.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path = [path for path in sys.path if Path(path or ".").resolve() != ROOT]
sys.path.insert(0, str(ROOT))

from dataset.skeleton_dataset_fourier import VesselSkeletonDatasetFourier, reconstruct_branch
from models.branch_transformer import MultiBranchConditions
from models.branch_mlp_vae import BranchFourierVAE, BranchFourierVAE_GCNConditioner
from utils.generate_synthetic import load_ghd_vae

# ── variant ───────────────────────────────────────────────────────────────────
USE_GCN        = True   # True → GCN mesh encoder for phi;  False → GHDTokenEncoder (simple MLP)

PROCESSED_ROOT = ROOT / "runtime" / "dataset" / "processed"
CANONICAL_ROOT = ROOT / "dataset" / "canonical"   # only used when USE_GCN=True

DEVICE     = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
EPOCHS     = 3000
BATCH_SIZE = 64
LR         = 1e-3
LR_MIN     = 1e-5

K            = 8           # Fourier harmonics per coordinate (must match the dataset)
HIDDEN_DIM   = 512
LATENT_DIM   = 8
NUM_LAYERS   = 4           # MLP depth in encoder / decoder
COND_DIM     = 64
GHD_DIM      = 16
MAX_BRANCHES = 3
NUM_TYPES    = 3

# GCN encoder (only used when USE_GCN=True)
GCN_HIDDEN     = 32
GCN_POOL_RATIO = 0.5

KL_BASE      = 1e-3
KL_MULT      = 2           # integer multiplier; goes into SAVE_DIR name
KL_WEIGHT    = KL_BASE * KL_MULT
KL_ANNEAL_EPOCHS = 300
VEC_WEIGHT      = 1.0      # weight of the chord (branch_vector) loss vs coeff loss
PRESENCE_WEIGHT = 1.0      # weight of the per-branch presence (BCE) loss

# Candidate openings per aneurysm type (matches the canonical openings.npz counts
# and compute_branch_conditions): bifurcated has 3 openings, sidewall has 2.
TYPE_N_OPEN = {0: 3, 1: 2, 2: 2}
CANONICAL_TYPE_NAME = {0: "bifurcated", 1: "sidewall", 2: "sidewall"}


def opening_mask_from_types(aneurysm_type, max_branches):
    """[B, max_branches] bool — which slots are candidate openings for each type."""
    n = torch.tensor([TYPE_N_OPEN.get(int(t), max_branches) for t in aneurysm_type.tolist()])
    return torch.arange(max_branches).unsqueeze(0) < n.unsqueeze(1)

_VARIANT = "branch_mlp_vae_gcn" if USE_GCN else "branch_mlp_vae"
SAVE_DIR = ROOT / "runtime" / "tr_checkpoints" / "v2" / "stage2" / "branch_mlp_vae" / f"{_VARIANT}_h{HIDDEN_DIM}_l{NUM_LAYERS}_z{LATENT_DIM}_kl{KL_MULT}"

# Synthetic-direction loss: sample phi from a frozen GHD VAE, reconstruct the
# mesh openings (start + outward direction) via multi_recon, generate Fourier
# branches off that condition, and constrain each branch's *initial tangent*
# (analytic from branch_vector + coeffs) to match the opening's outward direction.
SYN_DIR_WEIGHT = 0.5      # 0 disables the synthetic-direction loss
B_SYN          = 24       # synthetic batch size; must be divisible by NUM_TYPES
SYN_DIR_GHD_Z_AMP = 3.0   # synthetic GHD latent sampled ~ U(-SYN_DIR_GHD_Z_AMP, SYN_DIR_GHD_Z_AMP)

GHD_VAE_CKPT   = ROOT / "runtime" / "tr_checkpoints" / "v2" / "stage1" / "ghd_vae_h512_z16_kl2" / "epoch_05000.pth"
# GHD VAE architecture is loaded from GHD_VAE_CKPT's own "args" (see utils.generate_synthetic.load_ghd_vae)

SAVE_EVERY   = 500
LOG_EVERY    = 10
SANITY_EVERY = 500
SANITY_SYNTHETIC = True    # also render fully-generated branches from synthetic GHD phi
VAL_FRACTION = 0.1
VAL_SEED     = 42

WANDB_PROJECT = "AneuGv2_stage2"
WANDB_RUN_NAME = SAVE_DIR.name


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
    return MultiBranchConditions(
        aneurysm_type    = batch["aneurysm_type"].to(DEVICE),
        scale            = batch["scale"].to(DEVICE),
        start_points     = batch["start_points"].to(DEVICE),
        branch_direction = batch["branch_direction"].to(DEVICE),
        branch_mask      = batch["branch_mask"].to(DEVICE),
    )


def prepare_batch(batch):
    return {
        "branch_vector":  batch["branch_vector"].to(DEVICE),    # [B, mb, 3]
        "fourier_coeffs": batch["fourier_coeffs"].to(DEVICE),   # [B, mb, k, 3]
        "phi":            batch["phi"].to(DEVICE),              # [B, 144, 3]
        "branch_mask":    batch["branch_mask"].to(DEVICE),     # [B, mb]  present branches
        "opening_mask":   opening_mask_from_types(batch["aneurysm_type"], MAX_BRANCHES).to(DEVICE),  # candidates
        "cond":           make_conditions(batch),
    }


def model_args():
    """Kwargs needed to reconstruct the branch model from a checkpoint alone.

    Excludes `multi_recon` (the GCN variant's first positional arg) since it
    isn't serializable — pass it in separately when use_gcn is True.
    """
    args = {
        "use_gcn": USE_GCN,
        "k": K,
        "hidden_dim": HIDDEN_DIM,
        "latent_dim": LATENT_DIM,
        "num_layers": NUM_LAYERS,
        "num_types": NUM_TYPES,
        "max_branches": MAX_BRANCHES,
        "ghd_dim": GHD_DIM,
        "cond_dim": COND_DIM,
    }
    if USE_GCN:
        args["gcn_hidden"] = GCN_HIDDEN
        args["gcn_pool_ratio"] = GCN_POOL_RATIO
    return args


def all_hparams():
    """Every module-level training hyperparameter, for record-keeping in the checkpoint."""
    return {
        "use_gcn": USE_GCN,
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
        "kl_base": KL_BASE,
        "kl_mult": KL_MULT,
        "kl_weight": KL_WEIGHT,
        "kl_anneal_epochs": KL_ANNEAL_EPOCHS,
        "vec_weight": VEC_WEIGHT,
        "presence_weight": PRESENCE_WEIGHT,
        "syn_dir_weight": SYN_DIR_WEIGHT,
        "b_syn": B_SYN,
        "syn_dir_ghd_z_amp": SYN_DIR_GHD_Z_AMP,
        "ghd_vae_ckpt": str(GHD_VAE_CKPT),  # GHD VAE's own architecture is in its checkpoint's "args"
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
        cond = MultiBranchConditions(
            aneurysm_type    = item["aneurysm_type"].unsqueeze(0).to(DEVICE),
            scale            = item["scale"].unsqueeze(0).to(DEVICE),
            start_points     = item["start_points"].unsqueeze(0).to(DEVICE),
            branch_direction = item["branch_direction"].unsqueeze(0).to(DEVICE),
            branch_mask      = item["branch_mask"].unsqueeze(0).to(DEVICE),
        )
        phi = item["phi"].unsqueeze(0).to(DEVICE)
        z = torch.zeros(1, model.latent_dim, device=DEVICE) if z_zero else None
        vec_p, coeff_p, pres_p = model.sample(phi, cond, z=z)
        vec_p, coeff_p, pres_p = vec_p[0].cpu().numpy(), coeff_p[0].cpu().numpy(), pres_p[0].cpu().numpy()

        atype  = int(item["aneurysm_type"])
        n_open = TYPE_N_OPEN.get(atype, model.max_branches)
        verts, faces = _reconstruct_ghd_numpy(
            {"aneurysm_type": atype, "ghd": {"phi": item["phi"].numpy()}}, denormalize_shape=True)

        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces,
                        color="lightgray", edgecolor="none", alpha=0.12)
        start = item["start_points"].numpy()
        vec_g = item["branch_vector"].numpy()
        coeff_g = item["fourier_coeffs"].numpy()
        gt_mask = item["branch_mask"].numpy().astype(bool)
        all_pts, pres_str = [verts], []
        for b in range(model.max_branches):
            c = bright[b % len(bright)]
            # ground truth: branches actually present in the data
            if gt_mask[b]:
                gt = reconstruct_branch(start[b], start[b] + vec_g[b], coeff_g[b])
                ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], color="black", lw=2.5)
                all_pts.append(gt)
            # prediction: only candidate openings the model deems present
            if b < n_open:
                pres_str.append(f"{pres_p[b]:.2f}")
                if pres_p[b] > 0.5:
                    pr = reconstruct_branch(start[b], start[b] + vec_p[b], coeff_p[b])
                    ax.plot(pr[:, 0], pr[:, 1], pr[:, 2], color=c, lw=2.0, ls=(0, (4, 2)))
                    all_pts.append(pr)
        ax.set_axis_off()
        ax.set_title(f"{item['case']}\npres=[{','.join(pres_str)}]", fontsize=7)
        _set_axes_equal(ax, *all_pts)

    fig.suptitle(f"epoch {epoch}  (black=GT, dashed=pred, z={'0' if z_zero else '~N'})", fontsize=10)
    fig.tight_layout()
    save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / f"sanity_epoch_{epoch:05d}.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    model.train()
    return out


@torch.no_grad()
def sanity_synthetic(model, ghd_vae, ghd_mean, ghd_std, ghd_input_dim, multi_recon, epoch, save_dir,
                     z_zero=True, n=8, ncols=4, seed=0):
    """Fully-generative sanity: synthetic GHD phi → mesh openings → generated
    branches (presence head decides which openings get a branch)."""
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

    # branch conditions (start + outward dir + candidate openings) from the mesh
    starts = phi.new_zeros(n, MAX_BRANCHES, 3)
    dirs   = phi.new_zeros(n, MAX_BRANCHES, 3)
    omask  = torch.zeros(n, MAX_BRANCHES, dtype=torch.bool, device=DEVICE)
    for atype in sorted(set(int(t) for t in types.tolist())):
        idx = (types == atype).nonzero(as_tuple=True)[0]
        s, d, m = multi_recon.compute_branch_conditions(phi[idx], atype, MAX_BRANCHES)
        starts[idx], dirs[idx], omask[idx] = s, d, m
    cond = MultiBranchConditions(types, torch.ones(n, device=DEVICE), starts, dirs, omask)
    z = torch.zeros(n, model.latent_dim, device=DEVICE) if z_zero else None
    vec_p, coeff_p, pres_p = model.sample(phi, cond, z=z)

    starts_np = starts.cpu().numpy()
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
                pr = reconstruct_branch(starts_np[i, b], starts_np[i, b] + vec_np[i, b], coeff_np[i, b])
                ax.plot(pr[:, 0], pr[:, 1], pr[:, 2], color=bright[b % len(bright)], lw=2.0)
                all_pts.append(pr)
        ax.set_axis_off()
        ax.set_title(f"type {atype}: {CANONICAL_TYPE_NAME.get(atype, '?')}  pres=[{','.join(pres_str)}]", fontsize=7)
        _set_axes_equal(ax, *all_pts)

    fig.suptitle(f"epoch {epoch}  synthetic generation (z={'0' if z_zero else '~N'})", fontsize=10)
    fig.tight_layout()
    save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / f"sanity_synth_epoch_{epoch:05d}.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    model.train()
    return out


class SyntheticDirectionLoss:
    """Constrain generated branches' initial tangent to the GHD mesh opening direction.

    For a batch of synthetic phi sampled from the frozen GHD VAE:
      1. (no grad) decode phi, reconstruct the mesh openings via multi_recon →
         per-branch start points + outward directions + valid mask.
      2. (grad) inject those as the condition, decode Fourier branches with z~N(0,I),
         compute each branch's analytic initial tangent, and penalise its angular
         deviation from the conditioned outward direction.
    Gradients flow through the branch model's ghd encoder / cond embed / decoder only.
    """

    def __init__(self, ghd_vae, ghd_mean, ghd_std, ghd_input_dim, multi_recon,
                 num_types=NUM_TYPES, max_branches=MAX_BRANCHES, b_syn=B_SYN,
                 ghd_z_amp=SYN_DIR_GHD_Z_AMP, device=DEVICE):
        assert b_syn % num_types == 0, f"B_SYN ({b_syn}) must be divisible by NUM_TYPES ({num_types})"
        self.ghd_vae = ghd_vae
        self.ghd_mean = ghd_mean
        self.ghd_std = ghd_std
        self.ghd_input_dim = ghd_input_dim
        self.multi_recon = multi_recon
        self.num_types = num_types
        self.max_branches = max_branches
        self.b_syn = b_syn
        self.ghd_z_amp = ghd_z_amp
        self.device = device

    def __call__(self, model):
        n_per = self.b_syn // self.num_types
        with torch.no_grad():
            types = torch.arange(self.num_types, device=self.device).repeat_interleave(n_per)
            z_ghd = torch.empty(self.b_syn, self.ghd_vae.latent_dim, device=self.device).uniform_(
                -self.ghd_z_amp, self.ghd_z_amp)
            ghd_n, _ = self.ghd_vae.decode(z_ghd, types)
            phi = (ghd_n * self.ghd_std[:, :self.ghd_input_dim]
                   + self.ghd_mean[:, :self.ghd_input_dim]).reshape(self.b_syn, -1, 3)
            starts, dirs, masks = [], [], []
            for atype in range(self.num_types):
                sl = slice(atype * n_per, (atype + 1) * n_per)
                s, d, m = self.multi_recon.compute_branch_conditions(phi[sl], atype, self.max_branches)
                starts.append(s); dirs.append(d); masks.append(m)
            starts = torch.cat(starts); dirs = torch.cat(dirs); b_mask = torch.cat(masks)
        if not b_mask.any():
            return None

        # grad path: condition on the (frozen) opening geometry, generate branches
        ghd_embed = model.encode_phi(phi, types)
        cond = MultiBranchConditions(
            aneurysm_type    = types,
            scale            = torch.ones(self.b_syn, device=self.device),
            start_points     = starts,
            branch_direction = dirs,
            branch_mask      = b_mask,
            ghd_embed        = ghd_embed,
        )
        cond_emb = model.cond_embed(cond)
        z = torch.randn(self.b_syn, model.latent_dim, device=self.device)
        vec, coeffs = model._denorm_out(model._decode(z, cond_emb))
        pred_dir = model.initial_tangent(vec, coeffs)                 # [B, mb, 3]
        cos = (pred_dir * dirs).sum(dim=-1)                           # [B, mb]
        return (1.0 - cos)[b_mask].mean()


def build_model(multi_recon=None):
    args = model_args()
    use_gcn = args.pop("use_gcn")
    if use_gcn:
        return BranchFourierVAE_GCNConditioner(multi_recon, **args).to(DEVICE)
    return BranchFourierVAE(**args).to(DEVICE)


def main():
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config=all_hparams(),
    )

    dataset = VesselSkeletonDatasetFourier(PROCESSED_ROOT, max_branches=MAX_BRANCHES,
                                           k=K, normalize=False)   # model owns normalization
    n_val   = max(1, int(len(dataset) * VAL_FRACTION))
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(VAL_SEED))
    loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)

    # multi_recon + GHD VAE are needed for the GCN conditioner, the synthetic-
    # direction loss, and/or the synthetic-generation sanity panel.
    need_recon = USE_GCN or SYN_DIR_WEIGHT > 0 or SANITY_SYNTHETIC
    need_ghd_vae = SYN_DIR_WEIGHT > 0 or SANITY_SYNTHETIC
    multi_recon = None
    if need_recon:
        from models.multi_canonical_ghd_reconstruct import MultiCanonicalGHDReconstruct
        multi_recon = MultiCanonicalGHDReconstruct(CANONICAL_ROOT, device=DEVICE)

    model = build_model(multi_recon)
    model.set_normalization_stats(*compute_target_stats(dataset))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=LR_MIN / LR, total_iters=EPOCHS
    )

    ghd_vae = ghd_mean = ghd_std = ghd_input_dim = None
    if need_ghd_vae:
        ghd_vae, ghd_mean, ghd_std, ghd_input_dim = load_ghd_vae(GHD_VAE_CKPT, DEVICE)
        for p in ghd_vae.parameters():
            p.requires_grad_(False)
        print(f"Loaded GHD VAE: {GHD_VAE_CKPT.name}")

    syn_dir_loss = None
    if SYN_DIR_WEIGHT > 0:
        syn_dir_loss = SyntheticDirectionLoss(ghd_vae, ghd_mean, ghd_std, ghd_input_dim, multi_recon)
        print(f"Synthetic-direction loss enabled (B_SYN={B_SYN})")
    print(f"{_VARIANT}: {len(train_set)} train / {len(val_set)} val  | coeff_dim={dataset.get_coeff_dim()}")

    for epoch in range(EPOCHS + 1):
        model.train()
        kl_weight = KL_WEIGHT * min(1.0, epoch / KL_ANNEAL_EPOCHS)
        metrics = {}
        for batch in loader:
            data = prepare_batch(batch)
            out, presence_logit, mu, logvar = model(
                data["branch_vector"], data["fourier_coeffs"], data["phi"], data["cond"])
            coeff_loss, vec_loss, presence_loss, kl_loss = model.get_loss(
                out, data["branch_vector"], data["fourier_coeffs"],
                data["branch_mask"], data["opening_mask"], presence_logit, mu, logvar)
            loss = (coeff_loss + VEC_WEIGHT * vec_loss
                    + PRESENCE_WEIGHT * presence_loss + kl_weight * kl_loss)
            metrics = {
                "loss": float(loss), "coeff": float(coeff_loss), "vec": float(vec_loss),
                "pres": float(presence_loss), "kl": float(kl_loss),
            }

            if syn_dir_loss is not None:
                syn = syn_dir_loss(model)
                if syn is not None:
                    loss = loss + SYN_DIR_WEIGHT * syn
                    metrics["syn"] = float(syn)

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
                sanity_synthetic(model, ghd_vae, ghd_mean, ghd_std, ghd_input_dim, multi_recon, epoch, SAVE_DIR)

    wandb.finish()


if __name__ == "__main__":
    main()

"""
conda activate new
python scripts/train/train_branch_mlp_vae.py

"""
