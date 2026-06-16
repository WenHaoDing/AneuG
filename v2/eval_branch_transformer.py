"""
Evaluation and visualisation for MultiBranchVAE / MultiBranchVAE_GCNConditioner.

For each case produces a side-by-side figure:
  col 0  : ground-truth branches on GHD mesh
  col 1-N: N_SAMPLES independently generated branches on the same mesh

conda activate new
python v2/eval_branch_transformer.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path = [path for path in sys.path if Path(path or ".").resolve() != ROOT]
sys.path.insert(0, str(ROOT))

from dataset.skeleton_dataset import VesselSkeletonDataset
from dataset.preprocess import _reconstruct_ghd_numpy, _set_axes_equal
from v2.branch_transformer import MultiBranchVAE, MultiBranchVAE_GCNConditioner, MultiBranchConditions
from v2.train_branch_transformer import (
    PROCESSED_ROOT, CANONICAL_ROOT, DEVICE,
    HIDDEN_DIM, LATENT_DIM, NUM_LAYERS, NHEAD, GHD_DIM,
    MAX_BRANCHES, MAX_POINTS_PER_BRANCH, NUM_TYPES,
    GCN_HIDDEN, GCN_POOL_RATIO,
    VAL_FRACTION, VAL_SEED,
)

# ── eval config ───────────────────────────────────────────────────────────────
CHECKPOINT = ROOT / "checkpoints" / "v2" / "branch_transformer_gcn" / "epoch_00400.pth"
USE_GCN    = True    # must match the checkpoint
N_CASES    = 10      # number of cases to visualise
N_SAMPLES  = 3       # independently generated samples per case
FROM_VAL   = True    # True = val split only;  False = full dataset
SAVE_DIR   = CHECKPOINT.parent / "eval"


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(checkpoint_path):
    chk = torch.load(checkpoint_path, map_location=DEVICE)
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
    model.load_state_dict(chk["model"])
    model.set_normalization_stats(chk["point_mean"], chk["point_std"])
    model.eval()
    return model


# ── branch extraction helpers ─────────────────────────────────────────────────

def gt_branch_points(item, dataset):
    """Absolute GT branch curves: start + denorm(local_points[:n])."""
    local = dataset.denormalize_local_points(item["local_points"])  # [max_branches, max_lp, 3]
    result = []
    for b in range(dataset.max_branches):
        if not item["branch_mask"][b]:
            continue
        n     = int(item["branch_length"][b])
        start = item["start_points"][b].numpy()                     # [3]
        offs  = local[b, :n].numpy()                                # [n, 3]
        result.append(np.concatenate([start[None], start[None] + offs], axis=0))
    return result


def gen_branch_points(outputs, lengths, item, dataset):
    """Absolute generated branch curves from one model.sample() call.

    outputs: [seq_len, 3]  (CPU, zeroed for absent branches)
    lengths: [max_branches]
    """
    max_lp  = dataset.max_local_points
    outputs = dataset.denormalize_local_points(outputs).view(dataset.max_branches, max_lp, 3)
    result  = []
    for b in range(dataset.max_branches):
        if not item["branch_mask"][b]:
            continue
        n     = int(lengths[b].clamp(min=1))
        start = item["start_points"][b].numpy()
        offs  = outputs[b, :n].numpy()
        result.append(np.concatenate([start[None], start[None] + offs], axis=0))
    return result


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


# ── per-case evaluation ───────────────────────────────────────────────────────

@torch.no_grad()
def eval_case(model, item, dataset, save_path):
    phi  = item["phi"].unsqueeze(0).to(DEVICE)                             # [1, 144, 3]
    cond = MultiBranchConditions(
        aneurysm_type    = item["aneurysm_type"].unsqueeze(0).to(DEVICE),
        scale            = item["scale"].unsqueeze(0).to(DEVICE),
        start_points     = item["start_points"].unsqueeze(0).to(DEVICE),
        branch_direction = item["branch_direction"].unsqueeze(0).to(DEVICE),
        branch_mask      = item["branch_mask"].unsqueeze(0).to(DEVICE),
    )

    # GHD mesh (numpy reconstruction)
    verts, faces = _reconstruct_ghd_numpy(
        {"aneurysm_type": int(item["aneurysm_type"]), "ghd": {"phi": item["phi"].numpy()}},
        denormalize_shape=True,
    )

    gt_pts = gt_branch_points(item, dataset)
    gen_pts_list = []
    for _ in range(N_SAMPLES):
        outputs, lengths, _ = model.sample(phi, cond)
        gen_pts_list.append(gen_branch_points(outputs[0].cpu(), lengths[0].cpu(), item, dataset))

    # 1 GT column + N_SAMPLES generated columns
    ncols = 1 + N_SAMPLES
    fig   = plt.figure(figsize=(4 * ncols, 4))
    fig.suptitle(
        f"{item['case']}  (type {int(item['aneurysm_type'])}: {item['canonical_type']})",
        fontsize=9,
    )

    ax = fig.add_subplot(1, ncols, 1, projection="3d")
    plot_branches(ax, verts, faces, gt_pts, "Ground Truth")

    for s, gen_pts in enumerate(gen_pts_list):
        ax = fig.add_subplot(1, ncols, s + 2, projection="3d")
        plot_branches(ax, verts, faces, gen_pts, f"Sample {s + 1}")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {save_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    dataset = VesselSkeletonDataset(
        PROCESSED_ROOT,
        max_branches=MAX_BRANCHES,
        max_points_per_branch=MAX_POINTS_PER_BRANCH,
    )

    if FROM_VAL:
        n_val   = max(1, int(len(dataset) * VAL_FRACTION))
        n_train = len(dataset) - n_val
        _, val_set = torch.utils.data.random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(VAL_SEED),
        )
        indices = [val_set.indices[i] for i in range(min(N_CASES, len(val_set)))]
    else:
        indices = list(range(min(N_CASES, len(dataset))))

    model = load_model(CHECKPOINT)
    print(f"Loaded checkpoint: {CHECKPOINT}")
    print(f"Evaluating {len(indices)} cases  ({N_SAMPLES} samples each) → {SAVE_DIR}")

    for rank, idx in enumerate(indices):
        item = dataset[idx]
        eval_case(model, item, dataset, SAVE_DIR / f"{rank:03d}_{item['case']}.png")


if __name__ == "__main__":
    main()


"""
conda activate new
python v2/eval_branch_transformer.py

"""
