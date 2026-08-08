"""
Generate synthetic vessel meshes from a GHD VAE + branch MLP-VAE checkpoint pair.

conda activate new
python scripts/generate/generate_synthetic.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path = [p for p in sys.path if Path(p or ".").resolve() != ROOT]
sys.path.insert(0, str(ROOT))

from utils.generate_synthetic import generate_synthetic_shapes

GHD_VAE_CKPT    = ROOT / "tr_checkpoints" / "v2" / "stage1" / "ghd_vae_h512_z16_kl2" / "epoch_05000.pth"
BRANCH_VAE_CKPT = ROOT / "tr_checkpoints" / "v2" / "stage2" / "branch_mlp_vae" / "branch_mlp_vae_gcn_h512_l4_z4_kl5" / "epoch_01000.pth"
ANEURYSM_TYPE   = 1      # 0/1/2, or None for a random type per sample
Z_AMP           = 1    # GHD + branch latents sampled ~ N(0, Z_AMP^2)
N_SAMPLES       = 12

# OBJ_DIR = BRANCH_VAE_CKPT.parent / "generated"
OBJ_DIR = "v2/synthetic_debug"  # relative to ROOT

generate_synthetic_shapes(
    ghd_vae_ckpt=GHD_VAE_CKPT,
    branch_vae_ckpt=BRANCH_VAE_CKPT,
    canonical_root=ROOT / "dataset" / "canonical",
    n_samples=N_SAMPLES,
    types=ANEURYSM_TYPE,
    ghd_z_amp=Z_AMP,
    branch_z_zero=True,
    branch_z_amp=1,
    obj_dir=OBJ_DIR,
)
print(f"Saved {N_SAMPLES} meshes → {OBJ_DIR}")


"""
conda activate new
python scripts/generate/generate_synthetic.py

"""