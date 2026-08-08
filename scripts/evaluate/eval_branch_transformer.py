"""
Fully-generative evaluation for MultiBranchVAE / MultiBranchVAE_GCNConditioner.

Pipeline (no ground-truth data involved):
  1. Sample a random aneurysm type per sample.
  2. Sample z_ghd ~ N(0, I) and decode it through the pre-trained GHD VAE
     (TypeConditionalVAE), conditioned on the random type, to generate a GHD
     shape (phi) and its scale.
  3. Derive branch conditions (start points, outward directions, branch mask)
     from the reconstructed mesh openings.
  4. Generate branch points with the branch transformer using z = 0, i.e. the
     mean/most-likely branches given the condition (deterministic).

For each generated sample we render the reconstructed GHD mesh overlaid with
its generated branches, arranged in a grid.

conda activate new
python scripts/evaluate/eval_branch_transformer.py
"""

import re
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path = [path for path in sys.path if Path(path or ".").resolve() != ROOT]
sys.path.insert(0, str(ROOT))

from dataset.preprocess_ImperialNHS import _reconstruct_ghd_numpy, _set_axes_equal
from models.branch_transformer import MultiBranchVAE, MultiBranchVAE_GCNConditioner, MultiBranchConditions
from models.multi_canonical_ghd_reconstruct import MultiCanonicalGHDReconstruct
from scripts.train.train_branch_transformer import (
    CANONICAL_ROOT, DEVICE,
    HIDDEN_DIM, LATENT_DIM, NUM_LAYERS, NHEAD, GHD_DIM,
    MAX_BRANCHES, MAX_POINTS_PER_BRANCH, NUM_TYPES,
    GCN_HIDDEN, GCN_POOL_RATIO,
    load_ghd_vae,
)

# ── eval config ───────────────────────────────────────────────────────────────
CHECKPOINT = ROOT / "tr_checkpoints" / "v2" / "branch_transformer" / "branch_transformer_gcn_h64_z8_kl1_weighted" / "epoch_02000.pth"
# GHD VAE checkpoint — must match GHD_HIDDEN_DIM/GHD_LATENT_DIM in train_branch_transformer (h512/z108).
GHD_VAE_CKPT = ROOT / "tr_checkpoints" / "v2" / "ghd_vae_h512_z108_kl1" / "epoch_10000.pth"
USE_GCN    = True              # must match the checkpoint
Z_ZERO     = False             # True → branch z = 0 (mean/most-likely);  False → z ~ N(0, I)
Z_AMP = 1
N_GEN      = 12                # number of random samples to generate
NCOLS      = 4                 # grid columns
SEED       = 1                 # RNG seed for reproducible generation
CANONICAL_TYPE_NAME = {0: "bifurcated", 1: "sidewall", 2: "sidewall"}
SAVE_DIR   = CHECKPOINT.parent / "eval_generative"

# ── fusion config ───────────────────────────────────────────────────────────
FUSE           = True   # uncap the GHD mesh + glue tubular branches, then render/save the merged mesh
SAVE_OBJ       = True   # write each merged mesh as an .obj
EXTRUDE_LENGTH = 3.0    # straight stub length (mm) for short/invalid branches
MIN_BRANCH_ARC = 3.0    # generated branch shorter than this (mm) → extrude instead of tube
FUSE_SMOOTH    = True   # smooth the seam + planarize tube-end openings


# ── model loading ─────────────────────────────────────────────────────────────

def dims_from_checkpoint_name(checkpoint_path, default_hidden, default_latent):
    """Parse hidden/latent dims from the checkpoint folder name (…_h64_z8_kl1).

    Only hidden_dim and latent_dim are encoded in the name; other architecture
    hyper-parameters (nhead, num_layers, ghd_dim, GCN params, …) still come from
    the train config import.
    """
    name = Path(checkpoint_path).parent.name
    m = re.search(r"_h(\d+)_z(\d+)", name)
    if not m:
        print(f"[warn] no '_h<H>_z<Z>' in '{name}'; using defaults h{default_hidden}/z{default_latent}")
        return default_hidden, default_latent
    hidden, latent = int(m.group(1)), int(m.group(2))
    print(f"Parsed dims from '{name}': hidden_dim={hidden}, latent_dim={latent}")
    return hidden, latent


def num_layers_from_state_dict(state, default_layers, prefix="encoder.transformer.layers."):
    """Infer the transformer depth from a checkpoint state_dict.

    num_layers is not encoded in the folder name, so count the
    `encoder.transformer.layers.<i>.*` blocks present in the checkpoint.
    """
    idxs = [int(m.group(1)) for k in state
            for m in [re.match(re.escape(prefix) + r"(\d+)\.", k)] if m]
    if not idxs:
        print(f"[warn] could not infer num_layers from checkpoint; using default {default_layers}")
        return default_layers
    n = max(idxs) + 1
    print(f"Inferred num_layers={n} from checkpoint")
    return n


def load_branch_model(checkpoint_path, multi_recon):
    hidden_dim, latent_dim = dims_from_checkpoint_name(checkpoint_path, HIDDEN_DIM, LATENT_DIM)
    chk = torch.load(checkpoint_path, map_location=DEVICE)
    num_layers = num_layers_from_state_dict(chk["model"], NUM_LAYERS)
    if USE_GCN:
        model = MultiBranchVAE_GCNConditioner(
            multi_recon,
            hidden_dim=hidden_dim, latent_dim=latent_dim, num_layers=num_layers, nhead=NHEAD,
            max_local_points=MAX_POINTS_PER_BRANCH - 1,
            num_types=NUM_TYPES, max_branches=MAX_BRANCHES, ghd_dim=GHD_DIM,
            gcn_hidden=GCN_HIDDEN, gcn_pool_ratio=GCN_POOL_RATIO,
        ).to(DEVICE)
    else:
        model = MultiBranchVAE(
            hidden_dim=hidden_dim, latent_dim=latent_dim, num_layers=num_layers, nhead=NHEAD,
            max_local_points=MAX_POINTS_PER_BRANCH - 1,
            num_types=NUM_TYPES, max_branches=MAX_BRANCHES, ghd_dim=GHD_DIM,
        ).to(DEVICE)
    model.load_state_dict(chk["model"])
    model.set_normalization_stats(chk["point_mean"], chk["point_std"])
    model.eval()
    return model


# ── generation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_ghd(ghd_vae, ghd_mean, ghd_std, types, z_amp=1.0):
    """Sample z_ghd ~ N(0,I) and decode → (phi [B,144,3], scale [B])."""
    B        = types.size(0)
    z_ghd    = torch.randn(B, ghd_vae.latent_dim, device=DEVICE) * z_amp
    ghd_n, scale_n = ghd_vae.decode(z_ghd, types)                              # [B,432], [B,1]
    phi      = (ghd_n * ghd_std[:, :432] + ghd_mean[:, :432]).reshape(B, -1, 3)
    scale    = (scale_n * ghd_std[:, 432:] + ghd_mean[:, 432:]).squeeze(1)     # [B]
    return phi, scale


@torch.no_grad()
def build_conditions(multi_recon, phi, types, scale, max_branches):
    """Branch conditions from mesh openings, for a mixed-type batch."""
    B      = phi.size(0)
    starts = phi.new_zeros(B, max_branches, 3)
    dirs   = phi.new_zeros(B, max_branches, 3)
    mask   = torch.zeros(B, max_branches, dtype=torch.bool, device=phi.device)
    for atype in sorted(set(int(t) for t in types.tolist())):
        idx = (types == atype).nonzero(as_tuple=True)[0]
        s, d, m = multi_recon.compute_branch_conditions(phi[idx], atype, max_branches)
        starts[idx], dirs[idx], mask[idx] = s, d, m
    return MultiBranchConditions(
        aneurysm_type    = types,
        scale            = scale,
        start_points     = starts,
        branch_direction = dirs,
        branch_mask      = mask,
    )


def abs_branches(outputs_b, lengths_b, starts_b, mask_b, point_mean, point_std, max_branches):
    """One sample's absolute branch curves: start + denorm(local offsets[:n]).

    Skips masked-out branches. Used for plotting only.
    """
    curves = branch_curves(outputs_b, lengths_b, starts_b, point_mean, point_std, max_branches)
    return [c for b, c in enumerate(curves) if bool(mask_b[b])]


def branch_curves(outputs_b, lengths_b, starts_b, point_mean, point_std, max_branches):
    """Absolute generated centerline per branch (ALL branches, no mask filtering).

    Returns a list of length max_branches; entry b is the [n+1, 3] curve
    start + denorm(local offsets[:n]) in physical GHD space. Short branches stay
    short so the fusion step can decide to extrude them.
    """
    max_lp = outputs_b.shape[0] // max_branches
    offs   = (outputs_b.view(max_branches, max_lp, 3) * point_std + point_mean)
    curves = []
    for b in range(max_branches):
        n     = int(lengths_b[b].clamp(min=1))
        start = starts_b[b].cpu().numpy()
        o     = offs[b, :n].cpu().numpy()
        curves.append(np.concatenate([start[None], start[None] + o], axis=0))
    return curves


def polydata_tris(mesh):
    """(verts, faces) numpy arrays from a pyvista PolyData (triangles)."""
    return np.asarray(mesh.points), mesh.faces.reshape(-1, 4)[:, 1:]


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_branches(ax, verts, faces, branch_pts, title):
    ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces,
                    color="lightgray", edgecolor="none", alpha=0.2)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(branch_pts), 1)))
    for i, pts in enumerate(branch_pts):
        c = colors[i % len(colors)]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], linewidth=2, color=c)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=6, color=c)
        ax.scatter(*pts[0], s=50, marker="x", color=c)
    ax.set_title(title, fontsize=8)
    ax.set_axis_off()
    _set_axes_equal(ax, verts, *branch_pts)


# ── main ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    multi_recon = MultiCanonicalGHDReconstruct(CANONICAL_ROOT, device=DEVICE)
    model       = load_branch_model(CHECKPOINT, multi_recon)
    ghd_vae, ghd_mean, ghd_std = load_ghd_vae(GHD_VAE_CKPT, DEVICE)
    print(f"Loaded branch transformer: {CHECKPOINT}")
    print(f"Loaded GHD VAE:            {GHD_VAE_CKPT}")

    # 1. random types → 2. generate GHD shapes
    types = torch.randint(0, NUM_TYPES, (N_GEN,), device=DEVICE)
    phi, scale = generate_ghd(ghd_vae, ghd_mean, ghd_std, types, z_amp=Z_AMP)

    # 3. branch conditions from mesh openings
    cond = build_conditions(multi_recon, phi, types, scale, MAX_BRANCHES)

    # 4. generate branches
    #    Z_ZERO=True  → z = 0  (mean / most-likely branches given the condition)
    #    Z_ZERO=False → z ~ N(0, I)  (sample diverse branches)
    if Z_ZERO:
        z = torch.zeros(N_GEN, model.latent_dim, device=DEVICE)
    else:
        z = torch.randn(N_GEN, model.latent_dim, device=DEVICE) * Z_AMP
    outputs, lengths, _ = model.sample(phi, cond, z=z)

    point_mean = model.point_mean   # [1, 3]
    point_std  = model.point_std    # [1, 3]

    z_label = "z=0" if Z_ZERO else "z~N(0,I)"
    tag     = ("fused" if FUSE else "ghd") + ("_z0" if Z_ZERO else "_zrandn")
    obj_dir = SAVE_DIR / f"obj_{tag}"
    if FUSE and SAVE_OBJ:
        obj_dir.mkdir(parents=True, exist_ok=True)

    nrows = (N_GEN + NCOLS - 1) // NCOLS
    fig   = plt.figure(figsize=(4 * NCOLS, 4 * nrows))
    fig.suptitle(f"Generative samples ({z_label}, {'fused' if FUSE else 'ghd'})  |  {CHECKPOINT.name}", fontsize=11)

    for i in range(N_GEN):
        atype  = int(types[i])
        n_open = int(cond.branch_mask[i].sum())
        # generated centerlines per opening (physical GHD space)
        curves = branch_curves(outputs[i], lengths[i], cond.start_points[i],
                               point_mean, point_std, MAX_BRANCHES)
        display_pts = curves[:n_open]
        title = f"#{i}  type {atype}: {CANONICAL_TYPE_NAME.get(atype, '?')}  (scale {float(scale[i]):.2f})"

        if FUSE:
            try:
                obj_path = (obj_dir / f"{i:02d}_type{atype}.obj") if SAVE_OBJ else None
                merged = multi_recon.reconstruct_fused_mesh(
                    phi[i], atype,
                    branch_points=curves,
                    branch_mask=cond.branch_mask[i].cpu().numpy(),
                    extrude_length=EXTRUDE_LENGTH, min_branch_arc=MIN_BRANCH_ARC,
                    smooth=FUSE_SMOOTH, save_path=obj_path,
                )
                verts, faces = polydata_tris(merged)
                if obj_path is not None:
                    print(f"#{i}: saved {obj_path}")
            except Exception as exc:
                print(f"#{i}: fusion failed ({exc}); falling back to GHD mesh")
                verts, faces = _reconstruct_ghd_numpy(
                    {"aneurysm_type": atype, "ghd": {"phi": phi[i].cpu().numpy()}},
                    denormalize_shape=True,
                )
        else:
            verts, faces = _reconstruct_ghd_numpy(
                {"aneurysm_type": atype, "ghd": {"phi": phi[i].cpu().numpy()}},
                denormalize_shape=True,
            )

        ax = fig.add_subplot(nrows, NCOLS, i + 1, projection="3d")
        plot_branches(ax, verts, faces, display_pts, title)

    fig.tight_layout()
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    save_path = SAVE_DIR / f"generative_samples_{tag}.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {save_path}")


if __name__ == "__main__":
    main()


"""
conda activate new
python scripts/evaluate/eval_branch_transformer.py

"""
