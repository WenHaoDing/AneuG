# aneu_ghd — GHD Fitting for Sidewall Intracranial Aneurysm Meshes

Graph Harmonic Deformation (GHD) fitting pipeline for **sidewall intracranial aneurysm** surface meshes.
Deforms a canonical sidewall aneurysm template mesh to match a target patient mesh using a Laplacian eigenvector basis.

---

## Method Overview

```
Canonical mesh  ──► Normalise ──► Anatomy align ──► CPD align ──► GHD optimisation ──► Fitted mesh
                                        ▲                ▲                ▲
                                   Landmark file    Landmark file    Target mesh
Target mesh     ──► Normalise ───────────────────────────────────────────►┘
```

**Three-stage alignment:**
1. **Anatomy alignment** *(landmark-based, coarse)* — builds an anatomical coordinate frame from dome, neck and vessel cap landmarks (dome → +X, vessel → +Z, larger opening → +Z). Ensures the canonical mesh faces the same anatomical direction as the target before numerical registration. Critical for sidewall aneurysms where the dome protrusion direction varies across patients.
2. **CPD (Coherent Point Drift)** *(rigid, fine)* — corrects the small residual misalignment left after anatomy alignment using point-cloud registration (rotation + translation only).
3. **GHD optimisation** — deforms the canonical mesh using the Laplacian eigenvector basis `U` (pre-computed once per canonical mesh).

**Losses during GHD fitting:**
| Loss | Purpose |
|------|---------|
| Chamfer P0 | Surface distance (bidirectional) |
| Chamfer N1 | Normal consistency via Chamfer |
| DVS Occupancy | Winding-number volume overlap |
| Laplacian | Cotangent smoothness, prevents pinching |
| Normal consistency | Prevents face flipping |
| Edge length | Prevents edge collapse / stretching |
| ARAP rigid | Local rigidity (decays over iterations) |
| Landmark | Dome + cap alignment (tapered, optional) |

**Output metrics:**
- `chamfer_best` / `chamfer_final` — surface accuracy (lower = better)
- `Dice` — volumetric overlap vs target (0–1, higher = better)
- `GAR` — Good Angle Ratio, fraction of triangles with all angles in [30°, 120°] (higher = better)
- `converged` — True if `chamfer_final < chamfer_init × 0.95`

---

## Installation

```bash
# 1. Clone the repo
git clone <repo-url>
cd aneu_ghd

# 2. Install PyTorch (match your CUDA version)
# https://pytorch.org/get-started/locally/

# 3. Install pytorch3d (not on PyPI)
# https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md

# 4. Install remaining dependencies
pip install -r requirements.txt
```

---

## Quick Start

### Single case (script)

```bash
python scripts/fit_case.py \
    --canonical  path/to/canonical/final_aligned.obj \
    --target     path/to/target/final_aligned.obj \
    --out        results/case_id/ \
    --eigenvecs  path/to/canonical/eigenvectors.npy \
    --landmarks-dir path/to/landmarks/root \
    --canonical-id  canonical/p044 \
    --target-id     cases/C0002 \
    --device     cuda:0
```

### Batch run (loop over cases)

```bash
for case in data/cases/*/; do
    id=$(basename $case)
    python scripts/fit_case.py \
        --canonical     path/to/canonical/final_aligned.obj \
        --target        $case/final_aligned.obj \
        --eigenvecs     path/to/canonical/eigenvectors.npy \
        --landmarks-dir path/to/landmarks/root \
        --canonical-id  canonical/p044 \
        --target-id     cases/$id \
        --out           results/$id/ \
        --device        cuda:0
done
```

### Pre-compute eigenvectors (do once per canonical mesh)

Run the notebook `aneu_ghd/ghd.ipynb` through STEP 3, then execute the **"Save canonical eigenvectors"** cell. This saves `eigenvectors.npy` next to the canonical mesh and skips the ~1 min Laplacian build on every subsequent run.

---

## Outputs

Each run saves three files to `--out`:

| File | Contents |
|------|----------|
| `fitted_mesh.obj` | Fitted canonical mesh in normalised space |
| `metrics.json` | `chamfer_best`, `chamfer_final`, `dice`, `gar`, `converged`, `best_iter` |
| `loss_history.npz` | Per-iteration loss values (load with `np.load`) |

---

## Meshes without landmark files

If your cutting pipeline does not produce a `landmarks.npz` file, use `compute_landmarks_from_rings()` to compute landmarks from ring nodes and a segmented dome:

```python
from aneu_ghd import compute_landmarks_from_rings, normalize_lm

lm = compute_landmarks_from_rings(
    inlet_ring_nodes  = mesh.vertices[inlet_ring_ids],   # (N, 3)
    outlet_ring_nodes = mesh.vertices[outlet_ring_ids],  # (M, 3)
    dome_pts          = mesh.vertices[dome_mask],         # (K, 3)
)
lm_s = normalize_lm(lm, centroid, scale)
```

Inputs provided by the cutting pipeline:
- `inlet_ring_ids` — vertex indices of the inlet opening ring
- `outlet_ring_ids` — vertex indices of the outlet opening ring
- `dome_mask` — boolean mask or indices of dome-region vertices

---

## Package Structure

```
aneu_ghd/
├── __init__.py          # public API
├── io.py                # load_obj / save_obj
├── landmarks.py         # load_landmarks, normalize_lm, compute_landmarks_from_rings
├── alignment.py         # anatomy_align, cpd_align, build_frame
├── fitting.py           # FitConfig, FitResult, ghd_fit, prepare_dvs_samples
└── losses/
    ├── chamfer_loss.py
    ├── dice_loss.py
    ├── dvs_loss.py
    ├── landmark_loss.py
    ├── mesh_quality_loss.py
    └── rigid_loss.py

scripts/
└── fit_case.py          # CLI for single-case fitting

aneu_ghd/ghd.ipynb       # step-by-step notebook for development / debugging
```

---

## Key Parameters (FitConfig)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_iter` | 12000 | Optimisation iterations |
| `lr` | 1e-3 | Adam learning rate |
| `lambda_chamfer` | 1.0 | Chamfer point loss weight |
| `lambda_occupancy` | 1.0 | DVS volume loss weight |
| `lambda_rigid_start` | 3.0 | ARAP weight at start (decays to `lambda_rigid_end`) |
| `lambda_rigid_end` | 0.01 | ARAP weight at end |
| `lambda_landmark` | 0.1 | Landmark loss weight (0 = disabled) |
| `device` | `"cuda"` | Torch device string |

Override any parameter:
```python
from aneu_ghd import FitConfig
cfg = FitConfig(n_iter=8000, lambda_landmark=0.0, device="cuda:2")
```

---

## Citation

Based on GHD (Graph Harmonic Deformation) from [AneuG](https://github.com/...) and [GHDHeart](https://github.com/...).
