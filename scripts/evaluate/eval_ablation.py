"""
Ablation sweep over stage1 (GHD VAE) and stage2 (branch MLP-VAE) checkpoints.

For each stage1 checkpoint, pairs it with a fixed reference stage2 checkpoint and
generates N_GEN samples (24 by default). For each stage2 checkpoint, pairs it with
a fixed reference stage1 checkpoint and does the same. One image per checkpoint
under evaluation (the "ablated" stage), saved to runtime/tr_checkpoints/v2/ablation_eval/.

conda activate new
python scripts/evaluate/eval_ablation.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path = [path for path in sys.path if Path(path or ".").resolve() != ROOT]
sys.path.insert(0, str(ROOT))

from utils.generate_synthetic import generate_synthetic_shapes
from scripts.evaluate.eval_branch_mlp_vae import plot_branches, CANONICAL_TYPE_NAME

# ── eval config ───────────────────────────────────────────────────────────────
STAGE1_DIR = ROOT / "runtime" / "tr_checkpoints" / "v2" / "stage1"
STAGE2_DIR = ROOT / "runtime" / "tr_checkpoints" / "v2" / "stage2" / "branch_mlp_vae"

STAGE1_EPOCH = "epoch_05000.pth"   # final epoch for every stage1 config
STAGE2_EPOCH = "epoch_03000.pth"   # final epoch for every stage2 config

# reference checkpoint held fixed while the *other* stage is ablated
REF_STAGE1 = STAGE1_DIR / "ghd_vae_h512_z32_kl1" / STAGE1_EPOCH
REF_STAGE2 = STAGE2_DIR / "branch_mlp_vae_gcn_h512_l4_z8_kl2" / STAGE2_EPOCH

N_GEN  = 24
NCOLS  = 6
SEED   = 0
GEN_TYPE = 1                # aneurysm type held fixed across the sweep
Z_ZERO = True                # branch z = 0 (mode)
Z_AMP  = 1                   # GHD + branch latents sampled ~ N(0, Z_AMP^2)
PRESENCE_THRESH = 0.5

FUSE           = True
SAVE_OBJ       = True
EXTRUDE_LENGTH = 3.0
MIN_BRANCH_ARC = 3.0
FUSE_SMOOTH    = True

DEVICE   = "cuda:0"
SAVE_DIR = ROOT / "runtime" / "tr_checkpoints" / "v2" / "ablation_eval"


def discover_configs(stage_dir, epoch_name):
    """Every immediate subdir of stage_dir that has an epoch_name checkpoint."""
    configs = []
    for d in sorted(stage_dir.iterdir()):
        ckpt = d / epoch_name
        if d.is_dir() and ckpt.exists():
            configs.append((d.name, ckpt))
    return configs


def render(samples, title, save_path):
    nrows = (len(samples) + NCOLS - 1) // NCOLS
    fig = plt.figure(figsize=(4 * NCOLS, 4 * nrows))
    fig.suptitle(title, fontsize=11)
    for i, sample in enumerate(samples):
        atype = sample["type"]
        pres_str = ",".join(
            f"{sample['presence'][b]:.2f}" for b in range(len(sample["opening_mask"])) if sample["opening_mask"][b]
        )
        subtitle = f"#{i} type {atype}:{CANONICAL_TYPE_NAME.get(atype, '?')} pres=[{pres_str}]"
        ax = fig.add_subplot(nrows, NCOLS, i + 1, projection="3d")
        plot_branches(ax, sample["verts"], sample["faces"], sample["branches"], subtitle)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {save_path}")


def run_one(ghd_ckpt, branch_ckpt, tag, title):
    obj_dir = (SAVE_DIR / "obj" / tag) if SAVE_OBJ else None
    samples = generate_synthetic_shapes(
        ghd_vae_ckpt=ghd_ckpt,
        branch_vae_ckpt=branch_ckpt,
        canonical_root=ROOT / "dataset" / "canonical",
        n_samples=N_GEN,
        types=GEN_TYPE,
        ghd_z_amp=Z_AMP,
        branch_z_zero=Z_ZERO,
        branch_z_amp=Z_AMP,
        presence_thresh=PRESENCE_THRESH,
        fuse=FUSE,
        extrude_length=EXTRUDE_LENGTH,
        min_branch_arc=MIN_BRANCH_ARC,
        fuse_smooth=FUSE_SMOOTH,
        obj_dir=obj_dir,
        device=DEVICE,
        seed=SEED,
    )
    render(samples, title, SAVE_DIR / f"{tag}.png")


def main():
    print(f"Reference stage1: {REF_STAGE1}")
    print(f"Reference stage2: {REF_STAGE2}")

    stage1_configs = discover_configs(STAGE1_DIR, STAGE1_EPOCH)
    stage2_configs = discover_configs(STAGE2_DIR, STAGE2_EPOCH)
    print(f"Stage1 configs ({len(stage1_configs)}): {[n for n, _ in stage1_configs]}")
    print(f"Stage2 configs ({len(stage2_configs)}): {[n for n, _ in stage2_configs]}")

    # ablate stage1, hold stage2 fixed
    for name, ckpt in stage1_configs:
        title = f"[stage1 ablation] {name}  (stage2 fixed: {REF_STAGE2.parent.name})"
        run_one(ckpt, REF_STAGE2, f"stage1_{name}", title)

    # ablate stage2, hold stage1 fixed
    for name, ckpt in stage2_configs:
        title = f"[stage2 ablation] {name}  (stage1 fixed: {REF_STAGE1.parent.name})"
        run_one(REF_STAGE1, ckpt, f"stage2_{name}", title)


if __name__ == "__main__":
    main()
