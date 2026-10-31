# Handoff: synthetic aneurysm masks from AneuG

For the agent working in the image-generation repository
(`/media/yaplab2/HDD Storage/wenhao/image-generation`). AneuG provides a tool for this: give it
a healthy vessel patch, and it returns the patch's label map with a generated aneurysm attached
to the artery, ready to condition the image model. You choose:
- the model (checkpoint);
- where the masks are written;
- how the aneurysm site is chosen (by the model, at your prompt, or at random);
- for a morphology-conditioned model, the dome's volume and aspect ratio.

This follows `HANDOFF_inference_1d_graph.md`, which defined the graph format. The patch folders
you already export (`debugging/graph1d_aneug_test/<case>_<group>/`) are the right input, as
they are.

## 1. What it does

```
patch folder: graph.npz (1D vessel graph, mm), patch_seg.npy (artery = 1), meta.json (frame, core)
  -> site on the graph, inside the core          (model score, your prompt, or random)
  -> ostium ring -> aneurysm type -> GHD complex mesh -> morphology sensor: dome
  -> dome attached to the artery wall, voxelised on the patch grid (inference/merge.py)
  -> checks; failed samples are dropped and another one is tried
mask: (z, y, x) uint8, 0 background, 1 artery, 2 generated aneurysm (inside the core only)
```

## 2. Setup (once)

**Code: your clone of AneuG, branch `vessel-conditioned`.** Update it before generating:

```bash
cd "/media/yaplab2/HDD Storage/wenhao/image-generation/AneuG"
git fetch origin && git checkout vessel-conditioned && git pull --ff-only origin vessel-conditioned
```

**Weights: one link to the original repo.** Checkpoints are not in git. The tool reads three
of them, all under the original repo's `runtime_train/`:
- the ostium model you choose (§4);
- the frozen VAE decoder;
- the frozen morphology sensor.

Link the folder once, in the clone's root (git ignores the link):

```bash
ln -sfn "/media/yaplab2/HDD Storage/wenhao/AneuG/runtime_train" runtime_train
```

Read only: never write, move or delete anything through this link; training runs write there.

**Python:** `/home/yaplab2/miniconda3/envs/new/bin/python`, the env the AneuG models run in.
Run from the clone's root.

## 3. Use

### Command line

```bash
cd "/media/yaplab2/HDD Storage/wenhao/image-generation/AneuG"
PY=/home/yaplab2/miniconda3/envs/new/bin/python

# 4 masks per patch, sites chosen by the model; every patch folder under the given folder
$PY scripts/ostium/make_synthetic_masks.py \
    --ckpt runtime_train/ostium_model/abl_mlp_crop_norot/epoch_1000.pth \
    --patches "/media/yaplab2/HDD Storage/wenhao/image-generation/debugging/graph1d_aneug_test" \
    --out "<where you want the masks>" --n 4 --seed 0 [--preview]

# one patch, aneurysm at a given point (voxel index z,y,x in patch_seg, or --prompt_mm x,y,z)
$PY scripts/ostium/make_synthetic_masks.py --ckpt ... --patches <one patch folder> --out ... \
    --n 1 --site prompt --prompt_vox 48,52,40
```

| option | meaning |
|---|---|
| `--ckpt` | ostium-model checkpoint (required; §4) |
| `--patches` | patch folders, or folders containing patch folders |
| `--out` | output folder (required); created if needed |
| `--n` | accepted masks wanted per patch. Failed samples are retried, up to `--max_attempts` (default 4 × n). |
| `--site model` | site drawn from the model's site score over the graph points in the core (default); `--temperature` flattens (> 1) or sharpens (< 1) it |
| `--site prompt` | site at the graph point nearest `--prompt_vox z,y,x` or `--prompt_mm x,y,z`. With `--prompt_sigma MM` > 0 it is drawn near the prompt (σ along the vessel), weighted by the model's score. One patch per call, and the prompt must be in the core. |
| `--site random` | uniform along the vessel in the core, away from the vessel ends |
| `--volume_mm3`, `--aspect_ratio` | requested dome: volume in mm³; aspect ratio = dome height / **mean neck radius** (2× the clinical height / neck diameter). Only for a morphology-conditioned checkpoint; any other checkpoint stops with an error. With a conditioned checkpoint, whichever is not given is drawn from the real aneurysms. |
| `--seed` | random seed (sites, crops); part of the file names |
| `--preview` | also a PNG per mask: voxels and surface around the aneurysm; real sac orange for aneurysm patches |
| `--device` | `cpu` (default; the GPUs are busy with training). For a GPU: `CUDA_VISIBLE_DEVICES=<0, 1 or 3>` and `--device cuda`. **Never GPU 2: it crashes the workstation.** |

Output in `--out`:
- `<patch>_<seed>_<k>.npz`: key `seg`, (z, y, x) uint8 with 0 background, 1 artery, 2 the
  generated aneurysm. Same grid as the patch's `patch_seg.npy`. If the patch had a real
  aneurysm, it is removed first, so label 2 is only the generated one.
- `<patch>_<seed>_<k>.png`: with `--preview`.
- `masks.json`: one record per mask, updated by later runs into the same folder. Fields:
  - `patch`, `case`, `origin_xyz`, `spacing_xyz`, `patch_start_zyx_in_scan`, `core_start_zyx`,
    `core_size`;
  - `site`, `site_mm`, `atype`, `sac_voxels`, `sac_volume_mm3`, `merge_flags`;
  - `morph_target` and `dome_morph` (requested and generated volume and aspect ratio;
    conditioned models only);
  - `ckpt`, `epoch`, `seed`.

The binary condition mask is `seg > 0`; the aneurysm label is `seg == 2`.
`rim_inside_lumen` in `merge_flags` is a warning only: the neck was moved out onto the wall.

### Python

```python
import sys; sys.path.insert(0, "/media/yaplab2/HDD Storage/wenhao/image-generation/AneuG")
from inference.synthetic_masks import AneurysmMaskGenerator, load_patch

gen = AneurysmMaskGenerator("runtime_train/ostium_model/abl_mlp_crop_norot/epoch_1000.pth", device="cpu")
patch = load_patch(".../graph1d_aneug_test/1KPkS2Uocy_healthy")   # graph, seg, origin, spacing, core
masks = gen.generate(**patch, n=4, site="model", seed=0)
masks = gen.generate(**patch, n=1, site="prompt", prompt_vox=(48, 52, 40))
seg = masks[0]["seg"]                                                # (z, y, x) uint8 0/1/2
```

You can also pass your own arrays instead of a folder:
- `graph`: a dict with `pos`, `radius`, `edge_index`, or a path to `graph.npz`;
- `seg`: the (z, y, x) labels;
- `origin` and `spacing`: (x, y, z);
- `core`: `(start_zyx, size)`, or `None` for the whole patch.

Each returned mask is a dict with the fields of `masks.json` plus `seg` and the merge `report`.
`gen.attempts` lists every sample tried in the last call, with why it was dropped. Load the
model once and call `generate` for every patch.

## 4. Which checkpoint

**Default:** `runtime_train/ostium_model/abl_mlp_crop_norot/epoch_1000.pth`. On your 10 test
graphs this model was the best measured end to end. With 6 sites per patch, seed 0 (60
samples):

| | `abl_mlp_crop_norot` ep 1000 | `abl_mlp_crop_rot15_selfrim_graph_mix` ep 1000 |
|---|---|---|
| samples accepted | **53 / 60 (88%)** | 44 / 60 (73%) |
| sac volume, median / 90th pct (mm³) | 21.6 / 58.2 | 9.1 / 37.9 |

Use the second one only if the project lead asks for it.

Still training, not to be used yet:
- `abl_mlp_crop_rot15_selfrim_graph_mix_w2`;
- `abl_mlp_crop_rot15_selfrim_graph_mix_morph`, the morphology-conditioned model
  (`--volume_mm3` / `--aspect_ratio`).

You will be told when one is ready. The tool works with any of them unchanged: only `--ckpt`
changes. Record the checkpoint with every mask (`masks.json` does) and do not mix checkpoints
silently.

## 5. Notes

- Generated sacs are smaller than the real ones; the real median dome volume is about 53 mm³.
  The core also clips large sacs.
- Every mask was checked:
  - the sac is attached to the artery;
  - it is not mostly inside the parent vessel;
  - it does not run into another vessel;
  - it has a dome-like shape.

  Still, look at some `--preview` renders, and show a selection to the project lead before
  training on the masks. Their judgement decides.
- The frame is the one in your `DATASET_CARD.md`: mm = `origin + (x, y, z) * spacing`, no
  direction matrix, array axes (z, y, x).
- Write masks only into your own folders. Do not edit AneuG code in the clone: it is
  overwritten by the next pull. Report what needs changing.
- The geometry dataset `/media/yaplab2/wd8tb/wenhao/angioflow/data/geometry/` is read-only.
  Patient data stays local, with pseudonymised case IDs only.
