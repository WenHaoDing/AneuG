# Handoff: synthetic aneurysm masks from 1D vessel graphs

For the agent working in the image-generation repository
(`/media/yaplab2/HDD Storage/wenhao/image-generation`). It explains which AneuG model to load
and how to run it, so that, given the 1D vessel graph of an image-model patch, you get the
patch's label map with a synthetic aneurysm added. Turning those label maps into the masks
the image model needs is your part.

This follows `HANDOFF_inference_1d_graph.md` (same folder), which defined the graph format.
The graphs you already export in `debugging/graph1d_aneug_test/` are the right input, as they
are.

## 1. What the pipeline does

```
graph.npz (healthy vessel, mm)                         run_inference.py
  -> site on the graph (inside the core)
  -> ostium ring -> aneurysm type -> GHD complex mesh
  -> morphology sensor: which vertices are dome (p_dome)        -> sample_XX.npz
patch_seg.npy (artery = 1) + sample                    run_merge.py
  -> dome attached to the artery wall, voxelised on the patch grid
  -> label 2 = new sac, only inside the core                    -> merged_XX.npz
```

Both steps are command-line scripts in the AneuG repo. Run them from your clone of AneuG
(`image-generation/AneuG`, branch `vessel-conditioned`) with AneuG's Python env (§3). You do
not need to import or change AneuG code.

## 2. Which model to load

**Use:** `runtime_train/ostium_model/abl_mlp_crop_norot/epoch_1000.pth`
(through the `runtime_train` link of §3; the file itself is in the original AneuG folder)

- MLP heads, trained with off-centre crops, no rotation, no self-consistency or skeleton
  losses; training finished (epoch 1000).
- **The best one measured end to end on your test graphs** (generate, then merge; §6): the
  most samples accepted, and sac sizes closest to the real ones.

Alternative: `runtime_train/ostium_model/abl_mlp_crop_rot15_selfrim_graph_mix/epoch_1000.pth`.
- Adds ±15° rotation, ring–dome self-consistency, and losses tying the generated vessel to
  the graph skeleton.
- Its domes looked more physiological on the training-time validation cases. On your test
  graphs, though, fewer samples were accepted and the sacs were smaller (§6).
- Use it only if the project lead asks for it.

Do **not** use these yet (still training):
- `abl_mlp_crop_rot15_selfrim_graph_mix_w2`: stronger skeleton weights;
- `abl_mlp_crop_rot15_selfrim_graph_mix_morph`: lets the user request the dome volume and
  aspect ratio. It is used the same way, plus the `--volume_mm3` and `--aspect_ratio` options
  (§4.3).

You will be told when to switch. Do not mix samples from different checkpoints in one dataset
without recording which is which.

## 3. Setup, environment and rules

**Code: your clone.** Update it to the latest `vessel-conditioned` before every batch of
generation:

```bash
cd "/media/yaplab2/HDD Storage/wenhao/image-generation/AneuG"
git fetch origin && git checkout vessel-conditioned && git pull --ff-only origin vessel-conditioned
```

**Weights: one link to the original repo.** Checkpoints are not in git (`runtime_train/` is
ignored). Inference reads three of them, all under the original repo's `runtime_train/`:
- the ostium model (§2);
- the frozen VAE decoder:
  `runtime_train/ghd_vae/stage1/ghd_vae_gan_h256_z16_kl2_adv0.3/epoch_05000.pth`;
- the frozen morphology sensor:
  `runtime_train/morphoformer/morphology_sensor/h128_gps4_tw1_pw2_dome1_phi0.5_rot0.5/epoch_02000.pth`.

Link the whole folder once, in the clone's root:

```bash
cd "/media/yaplab2/HDD Storage/wenhao/image-generation/AneuG"
ln -sfn "/media/yaplab2/HDD Storage/wenhao/AneuG/runtime_train" runtime_train
```

Read it only: never write, move or delete anything through this link (training runs write
there). Everything else inference needs is in git: `dataset/canonical/`, and the
`dataset/ostium_*_reference.json` and `dataset/sac_shape_reference.json` files.

- Python: `/home/yaplab2/miniconda3/envs/new/bin/python`, the env the AneuG models run in.
- Run from the clone's root, with absolute paths for `--graphs` and `--out`.
- **CPU by default.** This is the intended mode; the GPUs are busy with training. If you ever
  use a GPU: `CUDA_VISIBLE_DEVICES=<0, 1 or 3> ... --device cuda`. **Never GPU 2: it crashes
  the workstation.**
- **Always pass `--out`** pointing into your own repo. Without it, outputs go to the clone's
  `runtime_infer/`. Do not edit AneuG code in the clone: it is overwritten by the next pull. If
  something in AneuG needs changing, report it.
- The geometry dataset `/media/yaplab2/wd8tb/wenhao/angioflow/data/geometry/` is read-only.
- Patient data stays local. Only pseudonymised case IDs, as in your current folders.

## 4. Running it

### 4.1 Input: one folder per patch

Same layout as `debugging/graph1d_aneug_test/<case>_<group>/`:

| file | needed by | content |
|---|---|---|
| `graph.npz` | both steps | `save_point_graph` format: `pos` (N, 3) mm, `radius` (N,) mm, `edge_index` (2, E), plus **`in_core`** (N,) bool. Sites are only placed on `in_core` points; without it every point is allowed. |
| `meta.json` | both steps | `origin_xyz`, `spacing_xyz`, `core_start_zyx`, `core_size`, `core_box_mm` (as you write them now) |
| `patch_seg.npy` | merge | `uint8` (z, y, x): 0 background, 1 artery, 2 real aneurysm. Label 2 is removed before merging, so an aneurysm patch gets only the generated sac. |

The frame is the one in your `DATASET_CARD.md`: mm = `origin + ijk * spacing`, no direction
matrix, i, j, k = array axes 2, 1, 0. The graph, the generated meshes and the merged masks all
share it.

### 4.2 Generate, then merge

```bash
cd "/media/yaplab2/HDD Storage/wenhao/image-generation/AneuG"
PY=/home/yaplab2/miniconda3/envs/new/bin/python
CKPT=runtime_train/ostium_model/abl_mlp_crop_norot/epoch_1000.pth
GRAPHS="/media/yaplab2/HDD Storage/wenhao/image-generation/<your graph folder>"
OUT="/media/yaplab2/HDD Storage/wenhao/image-generation/<your output folder>/norot_ep1000_seed0"

$PY scripts/ostium/run_inference.py --ckpt "$CKPT" --graphs "$GRAPHS" --out "$OUT" --n_sites 6 --seed 0 [--render]
$PY scripts/ostium/run_merge.py --infer "$OUT" --graphs "$GRAPHS" [--no_render]
```

- `--n_sites K`: aneurysms generated per patch, at sites the model scores as likely.
- `--seed`: fixes the site choice and the crops. For more samples per patch, run again with
  another `--seed` into another `--out`. The shapes are not bit-reproducible across runs.
- `--cases A B ...`: only these case folders.
- `--render` (inference) and the merge renders (on by default) write PNGs for checking by
  eye. They are slow. For bulk generation use `--no_render` and no `--render`, and render a
  subset.

### 4.3 Optional: choosing where the aneurysm goes

Instead of letting the model choose the site:
- `--prompts K`: K random graph points in the core, at least the training sites' typical
  distance from a vessel end, one aneurysm at each;
- `--prompt_mm X,Y,Z [X,Y,Z ...]`: given positions (mm, graph frame); the nearest graph
  point is used;
- `--prompt_sigma MM`: 0 (default) puts the site exactly at the prompt. A value > 0 lets the
  model pick a site within about that distance along the graph.

On the 10 test graphs, prompted and model-chosen sites were accepted equally often (92%
each, 120 samples per mode; `runtime_infer/ostium/site_mode_comparison.md`).

The morphology-conditioned model, once released, needs a requested dome per sample:
- `--volume_mm3 V --aspect_ratio A` for all samples, where the aspect ratio is dome height /
  **mean neck radius** (2× the clinical height / neck diameter);
- or neither option, and each sample's target is drawn from the real aneurysms.

## 5. Outputs, and which samples to use

Per case, in `$OUT/<case>/`:

| file | content |
|---|---|
| `sample_XX.npz` | the generated aneurysm in mm: `verts_mm`, `faces` (complex mesh), `p_dome` (per-vertex dome probability), `ring_mm`, `site`, `atype` (0/1/2), `metrics` (JSON string) |
| `merged_XX.npz` | **the result**: `seg`, `uint8` (z, y, x), the patch shape (96³), 0 background, 1 artery, **2 the generated sac** (inside the core only). Written only for samples that were built. |
| `merged_XX.png`, `patch_overview.png` | renders (voxels and surfaces; artery grey, generated sac blue, real sac orange) |
| `case.json`, `site_scores.npy` | per-sample checks, site score per graph point |

Plus `$OUT/summary.md` (generation) and `$OUT/merge_summary.md` / `merge_summary.json`
(merge, one record per sample).

**Use only samples with `"accepted": true` in `merge_summary.json`.**
- *Not built* (no `merged_XX.npz`): the dome could not be attached; for example
  `rim_move_too_large` or `detached_sac`.
- *Built but rejected* (`merged_XX.npz` exists, `"rejected"` non-empty): do not use. The
  reasons are `dome_inside_lumen`, `dome_into_other_vessel` and `not_dome_like`.
- `rim_inside_lumen` in `flags` is a warning only: the rim was projected out to the wall.
  Those samples are fine if accepted.

```python
import json, numpy as np
reps = json.load(open(f"{OUT}/merge_summary.json"))
for r in reps:
    if r["accepted"]:
        seg = np.load(f"{OUT}/{r['case']}/merged_{r['sample']:02d}.npz")["seg"]   # (z, y, x) 0/1/2
        sac = seg == 2                          # the synthetic aneurysm
        vessel = seg > 0                        # artery + aneurysm as one structure
```

`merge_summary.json` also gives `sac_volume_mm3` and shape descriptors per sample, if you
want to balance sizes.

## 6. Reference runs on your 10 test graphs

Both runs used `--n_sites 6 --seed 0` (60 samples each), on
`image-generation/debugging/graph1d_aneug_test`. Outputs, with renders, are in the original
AneuG folder: `runtime_infer/ostium/<run>/`.

| | `abl_mlp_crop_norot` ep 1000 (use) | `graph_mix` ep 1000 |
|---|---|---|
| run folder | `abl_mlp_crop_norot_ep1000` | `abl_mlp_crop_rot15_selfrim_graph_mix_ep1000` |
| accepted | **53 / 60 (88%)** | 44 / 60 (73%) |
| built but rejected | 7 (not_dome_like 5, dome_into_other_vessel 2, dome_inside_lumen 2) | 15 (not_dome_like 14, dome_inside_lumen 10) |
| not built | 0 | 1 (detached_sac) |
| sac volume, built samples, median / 90th pct (mm³) | 21.6 / 58.2 | 9.1 / 37.9 |
| label-2 voxels outside the core | 0 | 0 |

A sample can have several rejection reasons. For scale: the real aneurysms behind the model
have a median dome volume of about 53 mm³. Generated sacs are smaller, and the core clips large
ones.

A second seed of the `norot` model is in `abl_mlp_crop_norot_ep1000_seed1`. Prompted against
model-chosen sites (120 samples each, same model) are compared in `site_mode_comparison.md`.

## 7. Before using the masks for training

1. Check some `merged_XX.png` and `patch_overview.png` files by eye, and show a selection to
   the project lead. Their judgement decides; numbers alone do not.
2. Confirm that label 2 lies only inside the core box, and touches label 1 (the merge
   enforces both; check your extraction keeps them).
3. Record the checkpoint path, seed and options with every generated sample.
