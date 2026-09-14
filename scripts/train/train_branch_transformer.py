import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]


def _parse_args():
    import argparse
    q = argparse.ArgumentParser()
    q.add_argument("--no-rotation", dest="rotation", action="store_false", default=True,
                   help="Disable random SO(3) on mesh + conditions + targets.")
    q.add_argument("--curv-weight", type=float, default=None,
                   help="Weight on the curvature-matching term. 0 disables it.")
    q.add_argument("--tag", default="",
                   help="Suffix appended to the run directory name.")
    q.add_argument("--tangent-weight", type=float, default=None,
                   help="Weight on the soft-threshold first-segment tangent penalty. "
                        "0 disables it.")
    q.add_argument("--lr", type=float, default=None,
                   help="Initial learning rate before the warmup ramp.")
    q.add_argument("--device", default=None)
    q.add_argument("--epochs", type=int, default=None)
    return q.parse_args() if __name__ == "__main__" else q.parse_args([])


_args = _parse_args()
sys.path = [path for path in sys.path if Path(path or ".").resolve() != ROOT]
sys.path.insert(0, str(ROOT))

from dataset.skeleton_dataset import VesselSkeletonDataset
from models.branch_transformer import MultiBranchVAE, MultiBranchVAE_GCNConditioner, MultiBranchConditions
from visualization.sanity_check import sanity_check
from models.multi_canonical_ghd_reconstruct import MultiCanonicalGHDReconstruct

# ── variant ───────────────────────────────────────────────────────────────────
USE_GCN        = True   # True → MultiBranchVAE_GCNConditioner (GCN mesh encoder)
                          # False → MultiBranchVAE (flat MLP on phi tokens)

PROCESSED_ROOT = ROOT / "runtime_dataset" / "AneuG_processed"
CANONICAL_ROOT = ROOT / "dataset" / "canonical"   # only used when USE_GCN=True

DEVICE    = torch.device(_args.device or ("cuda:1" if torch.cuda.is_available() else "cpu"))
ROTATION_AUGMENT = _args.rotation
EPOCHS    = _args.epochs if _args.epochs is not None else 3000
SANITY_N_CASES = 6   # val cases rendered per sanity step
# Curvature matching. MSE on point positions is invariant to whether consecutive
# errors are correlated, so a curve wobbling +-e about the truth scores the same
# as one smoothly offset by e. Nothing in the objective prefers the smooth one,
# and the model duly produces independent per-point noise: measured at epoch 200,
# roughness (mean |2nd diff| / mean step) was 0.752 teacher-forced against 0.034
# for real centerlines, 22x. Teacher-forced being ROUGHER than free-running
# (0.516) rules out exposure bias -- the model is noisy even given a perfect
# history -- so a training-time term is the right lever.
#
# Matching the ground truth's 2nd difference rather than minimising the
# prediction's, so the term cannot bias toward straight lines.
#
# Weight chosen by measuring both terms on the epoch-600 checkpoint of the
# no-curvature norot run (whole train split, normalised units):
#   recon 0.0177, curvature mismatch 0.0033, GT curvature energy 4e-7.
# The mismatch is ~8000x the ground truth's own 2nd-difference energy, i.e.
# essentially all of it is prediction noise. At w=0.5 the term contributes
# 9.4% of recon; w=2.0 would contribute 38%, enough to trade away position
# accuracy. Early on the term is ~30x larger (0.106 at epoch 0), so it mainly
# does its work late.
CURV_WEIGHT = _args.curv_weight if _args.curv_weight is not None else 0.5

# First-segment tangent penalty.
#
# THE DEFECT. Generated branches leave their opening sideways: the first segment
# sits ~55 degrees off the tangent the model was conditioned on, then the curve
# turns and proceeds correctly. Decomposed against the tangent, the outward
# component is right (+0.13 against a true +0.11) while the in-plane component is
# 0.20 -- three times what isotropic error of that size would give, so it is a
# systematic sideways step across the cap, not noise. Real centerlines leave
# along the tangent to 0.35 degrees, so the data is clean, and the curvature
# weight is irrelevant (the cw0 and cw0.5 runs are the same here).
#
# WHY. The model learned to predict each point by extrapolating from the previous
# one, which is unavailable at the first token of the sequence. Per branch slot,
# teacher-forced against free-running on real data:
#       slot 0:  55.0 deg  /  54.9      slot 0 is the sequence's first token
#       slot 1:  11.7      /  36.3      correct given a TRUE history
#       slot 2:  10.5      / 106.4      worst: inherits a corrupted history
# Branch 0's first point is the only token in all 381 with no history to attend
# to, so it is the only one that must come from conditioning alone, and nothing
# ever forced the model to learn that path. In generation the other two branches
# attend to its corrupted output and inherit the error.
#
# The tangent IS supplied and IS obeyed elsewhere: rotate the conditioned tangent
# by 60 degrees and the body of the curve follows within 4. What was missing is
# any loss scoring the model against it. (get_dir_loss exists but constrains the
# average of the first ten offsets, which is already correct to 3 degrees.)
#
# SHAPE OF THE PENALTY. A soft hinge on the angle: inside the threshold the loss
# decays exponentially toward zero so a well-aimed branch is left alone, outside
# it grows linearly and fast. softplus gives exactly that in one expression. The
# free cone matters -- a hard alignment term would fight reconstruction over
# where the curve sits, rather than only stepping in when the error is visible.
TANGENT_WEIGHT = _args.tangent_weight if _args.tangent_weight is not None else 1.0
TANGENT_THRESH_DEG = 30.0   # free inside this cone
TANGENT_SOFT_DEG   = 5.0    # width of the transition; smaller = harder hinge
# 64, was 256. With 471 training cases and drop_last=True, a batch of 256 gave
# the model ONE optimizer step per epoch and threw away 215 of the 471 cases
# every epoch. "5000 epochs" was therefore 5000 gradient steps, which is the
# real reason this model looked unconverged next to the MLP VAE (batch 64, seven
# steps per epoch). At 64 an epoch is seven steps over 448 cases, so the same
# epoch count buys seven times the optimisation, and activation memory for the
# GCN conditioner over 4143-vertex meshes drops by about the same factor.
BATCH_SIZE = 64
# 3e-4, was 1e-3. At 1e-3 the rotation-augmented run diverged to NaN by epoch
# 20 and then produced NaN for 480 more epochs before a plotting call finally
# failed. Rotation makes the task harder, so early gradients are much larger
# than in the unrotated run, which trained fine at 1e-3.
# Back to 1e-4. At 3e-4 the rotation run trained without diverging, but the
# generated centerlines carried high-frequency twists the real ones do not have.
# A lower rate is the cheap thing to try first; note the model has no smoothness
# prior at all (it predicts free points, unlike the Fourier model which is
# band-limited to 8 harmonics and cannot produce them), so if the twists survive
# this, the parameterisation is the cause, not the step size.
#
# They survived. At 1e-4 and epoch 4200 the prediction's mean |2nd difference|
# is still 10.7x the ground truth's (0.0348 vs 0.0033), so the step size was
# never what made 3e-4 look twisty at epoch 1000 -- that run was simply not
# converged. Positional accuracy at 4200 is fine (RMS 0.130 mesh units, 1.15
# point spacings, 1.1% of branch arc length, val matching train), which is why
# the twists are what remain visible. Prefer 3e-4 for the next round to reach
# the same place sooner; 1e-3 is still out, it diverged to NaN by epoch 20.
LR        = _args.lr if _args.lr is not None else 1e-4
WARMUP_EPOCHS = 50   # linear ramp from LR/20; the divergence happened at epoch 20

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
SAVE_DIR = (ROOT / "runtime_train" / "branch_transformer"
            / f"{_VARIANT}_h{HIDDEN_DIM}_z{LATENT_DIM}_kl{KL_MULT}"
              f"{'' if CURV_WEIGHT <= 0 else f'_cw{CURV_WEIGHT:g}'}"
              f"{'' if _args.rotation else '_norot'}"
              f"{_args.tag}")

LR_MIN = 1e-5   # cosine annealing floor

# pre-trained GHD VAE (TypeConditionalVAE from train_ghd_vae.py)
# Best of 26 stage-1 runs by KPD (0.0076): the WGAN-GP variant.
GHD_VAE_CKPT    = (ROOT / "runtime_train" / "ghd_vae" / "stage1"
                   / "ghd_vae_gan_h256_z16_kl2_adv0.3" / "epoch_05000.pth")
SENSOR_CKPT     = (ROOT / "runtime_train" / "morphoformer" / "morphology_sensor"
                   / "h128_gps4_tw1_pw2_dome1_phi0.5_rot0.5_ft_syn0.3_lr0.0001"
                   / "epoch_01000.pth")
SANITY_SYNTHETIC = True
GHD_INPUT_DIM   = 432   # 144 * 3
GHD_HIDDEN_DIM  = 512
GHD_LATENT_DIM  = 108
GHD_COND_DIM    = 8

# Sparser than before because the runs are far longer: at 10000 epochs the old
# every-200 cadence would leave 50 checkpoints per run.
SAVE_EVERY   = 1000
LOG_EVERY    = 10
SANITY_EVERY = 1000
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
    # drop_last=False: with 471 cases the remainder is a real part of the corpus,
    # and discarding it meant a different slice went unseen every epoch -- 23 cases
    # at batch 64, and 215 of 471 back when the batch was 256. Nothing here needs
    # equal-sized batches: normalization is per layer, not per batch.
    # num_workers is overridable because the workers share /dev/shm: with 16
    # concurrent runs at 4 workers each, one run died mid-training with
    # "DataLoader worker killed by signal: Bus error", which is shared-memory
    # exhaustion rather than anything wrong with the run itself.
    loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=int(os.environ.get("NUM_WORKERS", 4)),
                        drop_last=False)

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
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(EPOCHS - WARMUP_EPOCHS, 1), eta_min=LR_MIN)
    warm = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.05, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warm, cos], milestones=[WARMUP_EPOCHS])

    ghd_vae = ghd_mean = ghd_std = ghd_input_dim = uncapper = None
    if SYN_DIR_WEIGHT > 0 or SANITY_SYNTHETIC:
        # utils version, not the local loader: it also returns input_dim, which
        # the synthetic panel needs to split phi from the scale channel.
        from utils.generate_synthetic import load_ghd_vae as _load_ghd_vae
        ghd_vae, ghd_mean, ghd_std, ghd_input_dim = _load_ghd_vae(GHD_VAE_CKPT, DEVICE)
        print(f"Loaded GHD VAE: {GHD_VAE_CKPT.parent.name}")
    if SANITY_SYNTHETIC:
        from models.morphoformer import SensorUncapper
        uncapper = SensorUncapper.from_checkpoint(SENSOR_CKPT, multi_recon, device=DEVICE)
        print(f"Morphology sensor: {SENSOR_CKPT.parent.name}")

    @torch.no_grad()
    def sanity_synthetic(epoch, n=8, ncols=4, seed=0):
        """Fully generative: synthetic phi from the frozen stage-1 GHD VAE, cap
        conditions from the morphology sensor, branches sampled from the prior.

        The val panel only shows reconstruction of real cases. This is the one
        that shows what the model will actually be asked to do.
        """
        import matplotlib; matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from dataset.preprocess_ImperialNHS import _set_axes_equal
        if ghd_vae is None or uncapper is None:
            return
        was = model.training; model.eval()
        g = torch.Generator(device="cpu").manual_seed(seed)
        types = torch.randint(0, 2, (n,), generator=g).to(DEVICE)   # type 2 merged into 1
        z = torch.randn(n, ghd_vae.latent_dim, generator=g).to(DEVICE)
        out = ghd_vae.decode(z, types, strip_scale=True) if ghd_mean.numel() > ghd_input_dim \
            else ghd_vae.decode(z, types)
        phi_n = out[0] if isinstance(out, tuple) else out
        m_, s_ = ghd_mean[..., :ghd_input_dim], ghd_std[..., :ghd_input_dim]
        phi = (phi_n * s_ + m_).reshape(n, -1, 3)

        starts, dirs, bmask = uncapper.branch_conditions(phi, types, MAX_BRANCHES,
                                                        skip_failed=True)
        cond = MultiBranchConditions(
            aneurysm_type=types, scale=torch.ones(n, device=DEVICE),
            start_points=starts, branch_direction=dirs, branch_mask=bmask)
        outputs, lengths, _ = model.sample(phi, cond)
        outputs = dataset.denormalize_local_points(outputs.cpu()).view(
            n, MAX_BRANCHES, MAX_POINTS_PER_BRANCH - 1, 3)
        lengths, sp = lengths.cpu(), starts.cpu()

        nrows = (n + ncols - 1) // ncols
        fig = plt.figure(figsize=(4 * ncols, 4 * nrows))
        bright = ["#e6194B", "#3cb44b", "#4363d8"]
        for i in range(n):
            at = int(types[i])
            verts = multi_recon._reconstruct_verts_np(phi[i].cpu().numpy(), at)
            faces = multi_recon.get(at).canonical_Meshes.faces_packed().cpu().numpy()
            ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
            ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces,
                            color="lightgray", edgecolor="none", alpha=0.18)
            pts_all, lens = [verts], []
            for b in range(MAX_BRANCHES):
                if not bool(bmask[i, b]):
                    continue
                k = int(lengths[i, b].clamp(min=1))
                pts = np.concatenate([sp[i, b][None].numpy(),
                                      sp[i, b][None].numpy() + outputs[i, b, :k].numpy()])
                ax.plot(*pts.T, color=bright[b % 3], lw=2.2)
                pts_all.append(pts); lens.append(k)
            ax.set_axis_off()
            ax.set_title(f"type {at}  len={lens}", fontsize=7)
            _set_axes_equal(ax, *pts_all)
        fig.suptitle(f"epoch {epoch}  synthetic generation (z~N(0,1))", fontsize=10)
        fig.tight_layout()
        d = Path(SAVE_DIR) / "sanity"; d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"synth_epoch_{epoch:05d}.png", dpi=130); plt.close(fig)
        if was: model.train()

    def tangent_loss(pred, cond, branch_mask,
                     thresh=math.radians(TANGENT_THRESH_DEG),
                     soft=math.radians(TANGENT_SOFT_DEG)):
        """Soft-threshold penalty on the first segment's direction, per branch.

        pred[:, b, 0] IS the first segment: targets are offsets from the branch
        start, so the offset at index 0 is exactly (first point - start point).

        Denormalized before taking a direction. The per-axis std is anisotropic
        (4.10 / 4.05 / 3.76), so normalizing a vector in the standardized space
        would skew the angle it makes with the tangent.

        acos is clamped off its endpoints: its gradient is unbounded at cos = +-1,
        and cos = 1 is exactly where a well-aimed branch sits.
        """
        B = pred.shape[0]
        first = pred.view(B, MAX_BRANCHES, MAX_POINTS_PER_BRANCH - 1, 3)[:, :, 0]
        first = first * model.point_std + model.point_mean
        u = F.normalize(first, dim=-1)
        cos = (u * cond.branch_direction).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6)
        ang = torch.acos(cos)
        # softplus((ang - thresh)/soft): ~exp decay below the threshold, linear above
        pen = soft * F.softplus((ang - thresh) / soft)
        return pen[branch_mask].mean(), torch.rad2deg(ang)[branch_mask].mean().detach()

    def curvature_loss(pred, target, token_mask, branch_mask):
        """Match the ground truth's second difference, per branch.

        Reshaped to [B, max_branches, L, 3] first: the sequence is three branches
        flattened together, so differencing the flat axis would compute curvature
        across a branch boundary, between the end of one branch and the start of
        the next. A triple counts only when all three of its points are valid AND
        the branch is present.
        """
        B = pred.shape[0]
        L = MAX_POINTS_PER_BRANCH - 1
        p3 = pred.view(B, MAX_BRANCHES, L, 3)
        t3 = target.view(B, MAX_BRANCHES, L, 3)
        m3 = token_mask.view(B, MAX_BRANCHES, L) & branch_mask.unsqueeze(-1)
        d2p = p3[:, :, 2:] - 2 * p3[:, :, 1:-1] + p3[:, :, :-2]
        d2t = t3[:, :, 2:] - 2 * t3[:, :, 1:-1] + t3[:, :, :-2]
        valid = (m3[:, :, 2:] & m3[:, :, 1:-1] & m3[:, :, :-2]).unsqueeze(-1).float()
        return ((d2p - d2t) ** 2 * valid).sum() / valid.sum().clamp(min=1) / 3.0

    def plot_loss_curves(hist, save_dir, title):
        """Overwrite loss_curves.png each log step."""
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        keys = ["loss", "recon", "kl", "length", "curv", "tan", "tan_deg", "val_recon", "val_kl",
                "val_length", "kl_weight", "lr"]
        fig, axes = plt.subplots(3, 4, figsize=(21, 11))
        for ax, k in zip(axes.flat, keys):
            v = hist.get(k, [])
            if len(v) != len(hist["epoch"]):
                ax.set_axis_off(); continue
            ax.plot(hist["epoch"], v, lw=1.2)
            base = k[4:] if k.startswith("val_") else None
            if base and len(hist.get(base, [])) == len(hist["epoch"]):
                ax.plot(hist["epoch"], hist[base], lw=1.0, ls="--", color="crimson", label="train")
                ax.legend(fontsize=7)
            ax.set_title(k); ax.set_xlabel("epoch"); ax.grid(alpha=0.3)
            if k in ("loss", "recon", "kl", "val_recon", "val_kl"):
                ax.set_yscale("log")
        fig.suptitle(title); fig.tight_layout()
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(save_dir) / "loss_curves.png", dpi=130); plt.close(fig)

    @torch.no_grad()
    def evaluate(subset):
        """Same losses on held-out cases. Never rotated: the model is used in the
        canonical frame, and a rotating val set would add variance to the very
        number you are watching for a train/val gap."""
        was = model.training; model.eval()
        tot, n = {}, 0
        for b in DataLoader(subset, batch_size=BATCH_SIZE):
            d = prepare_batch(b)
            rc, ll, mu, lv = model(d["local_points"], d["token_mask"], d["phi"], d["cond"])
            # argument ORDER matters here and I had it wrong: get_loss takes
            # (recon, local_points, length_logit, branch_length, token_mask,
            #  branch_mask, mu, logvar, point_weights=...)
            r, k, le = model.get_loss(
                rc, d["local_points"], ll, d["branch_length"], d["token_mask"],
                d["branch_mask"], mu, lv,
                point_weights=d["token_weights"] if USE_POINT_WEIGHTS else None)
            bs = b["phi"].size(0); n += bs
            for kk, vv in (("recon", r), ("kl", k), ("length", le)):
                tot[kk] = tot.get(kk, 0.0) + float(vv) * bs
        if was: model.train()
        return {f"val_{kk}": vv / max(n, 1) for kk, vv in tot.items()}

    def rotate_batch(batch):
        """Random SO(3) per sample on the whole condition/target system.

        local_points are already relative to each branch start, so they rotate
        as plain 3-vectors, as do start_points and branch_direction. phi is not
        rotated -- see encode_phi.
        """
        from models.morphoformer import RandomSO3Rotation
        R = RandomSO3Rotation.sample(batch["phi"].shape[0], batch["phi"].dtype, DEVICE)
        rv = lambda x: torch.einsum('bij,b...j->b...i', R, x.to(DEVICE))
        out = dict(batch)
        for k in ("start_points", "branch_direction", "local_points"):
            if k in batch:
                out[k] = rv(batch[k])
        return out, R

    history = {"epoch": [], "kl_weight": [], "lr": []}
    for epoch in range(EPOCHS + 1):
        model.train()
        kl_weight = KL_WEIGHT * min(1.0, epoch / KL_ANNEAL_EPOCHS)
        metrics = {}
        for batch in loader:
            rot = None
            if ROTATION_AUGMENT and USE_GCN:
                batch, rot = rotate_batch(batch)
            data = prepare_batch(batch)

            recon, length_logit, mu, logvar = model(
                data["local_points"], data["token_mask"], data["phi"], data["cond"], rot=rot,
            )
            recon_loss, kl_loss, length_loss = model.get_loss(
                recon, data["local_points"], length_logit,
                data["branch_length"], data["token_mask"], data["branch_mask"],
                mu, logvar,
                point_weights=data["token_weights"] if USE_POINT_WEIGHTS else None,
            )
            loss = recon_loss + kl_weight * kl_loss + LEN_WEIGHT * length_loss
            curv_loss = None
            if CURV_WEIGHT > 0:
                curv_loss = curvature_loss(recon, data["local_points"],
                                           data["token_mask"], data["branch_mask"])
                loss = loss + CURV_WEIGHT * curv_loss
            tan_loss = tan_deg = None
            if TANGENT_WEIGHT > 0:
                tan_loss, tan_deg = tangent_loss(recon, data["cond"], data["branch_mask"])
                loss = loss + TANGENT_WEIGHT * tan_loss
            if not torch.isfinite(loss):
                # Stop AT the divergence, not 480 epochs later on a plotting
                # call. The previous run printed nan from epoch 20 onward and
                # kept going, which wasted the whole run and hid the cause.
                raise RuntimeError(
                    f"non-finite loss at epoch {epoch}: recon={float(recon_loss)}, "
                    f"kl={float(kl_loss)}, length={float(length_loss)}")

            metrics = {
                "loss":   float(loss),
                "recon":  float(recon_loss),
                "kl":     float(kl_loss),
                "length": float(length_loss),
            }
            if curv_loss is not None:
                metrics["curv"] = float(curv_loss)
            if tan_loss is not None:
                metrics["tan"] = float(tan_loss)
                metrics["tan_deg"] = float(tan_deg)   # the readable one

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
            try:
                metrics.update(evaluate(val_set))
            except Exception as exc:
                # Do NOT swallow: a broken val call left the CUDA context in a
                # bad state and the run died several steps later with an opaque
                # device-side assert, far from the actual cause.
                raise RuntimeError(f"validation evaluation failed: {exc}") from exc
            print({"epoch": epoch, "kl_weight": round(kl_weight, 6), "lr": scheduler.get_last_lr()[0], **metrics})
            history["epoch"].append(epoch)
            history["kl_weight"].append(kl_weight)
            history["lr"].append(scheduler.get_last_lr()[0])
            for k in ("loss", "recon", "kl", "length", "curv", "tan", "tan_deg",
                      "val_recon", "val_kl", "val_length"):
                if k in metrics:
                    history.setdefault(k, []).append(metrics[k])
            plot_loss_curves(history, SAVE_DIR, SAVE_DIR.name)
        if epoch % SAVE_EVERY == 0:
            save_checkpoint(model, optimizer, dataset, epoch)
        if epoch % SANITY_EVERY == 0:
            # several cases, not one: a single random val case says almost nothing
            # about whether the model is working, and re-rolls the case each time
            # so two epochs are not comparable.
            for _i in range(SANITY_N_CASES):
                sanity_check(model, val_set, epoch, SAVE_DIR, DEVICE,
                             MultiBranchConditions, MAX_BRANCHES,
                             MAX_POINTS_PER_BRANCH - 1, case_index=_i,
                             multi_recon=multi_recon)
            sanity_synthetic(epoch)


if __name__ == "__main__":
    main()


"""
conda activate new
python scripts/train/train_branch_transformer.py

"""