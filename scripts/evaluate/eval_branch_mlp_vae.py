"""
Fully-generative evaluation for the Fourier branch MLP-VAE (models.branch_mlp_vae).

Generation itself lives in utils.generate_synthetic.generate_synthetic_shapes
(GHD VAE + branch MLP-VAE → fused vessel meshes); this script just renders the results.

conda activate new
python scripts/evaluate/eval_branch_mlp_vae.py
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

from dataset.preprocess_ImperialNHS import _set_axes_equal
from utils.generate_synthetic import generate_synthetic_shapes

# ── eval config ───────────────────────────────────────────────────────────────
CHECKPOINT = ROOT / "runtime" / "tr_checkpoints" / "v2" / "branch_mlp_vae" / "branch_mlp_vae_h128_z16_kl1" / "epoch_03000.pth"
GHD_VAE_CKPT = ROOT / "runtime" / "tr_checkpoints" / "v2" / "ghd_vae_h512_z32_kl1" / "epoch_10000.pth"
Z_ZERO     = True              # True → branch z = 0 (mean/most-likely);  False → z ~ N(0, Z_AMP^2)
N_GEN      = 12                # number of samples to generate
NCOLS      = 4
SEED       = 0
PRESENCE_THRESH = 0.5          # branch kept (tubed) iff predicted presence > this
GEN_TYPE   = 1                 # aneurysm type for every sample; None → random per sample

Z_AMP = 1              # GHD + branch latents sampled ~ N(0, Z_AMP^2)

# fusion config
FUSE           = True
SAVE_OBJ       = True
EXTRUDE_LENGTH = 3.0           # stub length (mm) for absent branches
MIN_BRANCH_ARC = 3.0
FUSE_SMOOTH    = True

CANONICAL_TYPE_NAME = {0: "bifurcated", 1: "sidewall", 2: "sidewall"}

DEVICE   = "cuda:0"
SAVE_DIR = CHECKPOINT.parent / "eval_generative"


def plot_branches(ax, verts, faces, branch_pts, title):
    ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces,
                    color="lightgray", edgecolor="none", alpha=0.2)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(branch_pts), 1))) if branch_pts else []
    for i, pts in enumerate(branch_pts):
        if pts is None:
            continue
        c = colors[i % len(colors)]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], linewidth=2, color=c)
        ax.scatter(*pts[0], s=40, marker="x", color=c)
    ax.set_title(title, fontsize=8)
    ax.set_axis_off()
    _set_axes_equal(ax, verts, *[p for p in branch_pts if p is not None])


def main():
    tag = ("fused" if FUSE else "ghd") + ("_z0" if Z_ZERO else "_zgaussian")
    obj_dir = (SAVE_DIR / f"obj_{tag}") if (FUSE and SAVE_OBJ) else None

    samples = generate_synthetic_shapes(
        ghd_vae_ckpt=GHD_VAE_CKPT,
        branch_vae_ckpt=CHECKPOINT,
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
    print(f"Loaded branch MLP-VAE: {CHECKPOINT}")
    print(f"Loaded GHD VAE:        {GHD_VAE_CKPT}")

    nrows = (N_GEN + NCOLS - 1) // NCOLS
    fig = plt.figure(figsize=(4 * NCOLS, 4 * nrows))
    fig.suptitle(f"Fourier MLP-VAE generation ({'z=0' if Z_ZERO else f'z~N(0,{Z_AMP}^2)'})  |  {CHECKPOINT.name}", fontsize=11)

    for i, sample in enumerate(samples):
        atype = sample["type"]
        pres_str = ",".join(
            f"{sample['presence'][b]:.2f}" for b in range(len(sample["opening_mask"])) if sample["opening_mask"][b]
        )
        title = f"#{i} type {atype}:{CANONICAL_TYPE_NAME.get(atype, '?')} pres=[{pres_str}]"
        ax = fig.add_subplot(nrows, NCOLS, i + 1, projection="3d")
        plot_branches(ax, sample["verts"], sample["faces"], sample["branches"], title)
        if sample["obj_path"] is not None:
            print(f"#{i}: saved {sample['obj_path']}")

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
python scripts/evaluate/eval_branch_mlp_vae.py

"""
