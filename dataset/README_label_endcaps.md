# Manual endcap labeling — `label_endcaps.py`

Interactive PyVista tool for hand-brushing the true vessel-opening ("cap")
region on a mesh, per branch. Written on a headless workstation and never
run interactively there — this doc is meant to get it running on a machine
with a real display (e.g. your Windows box).

## Why this exists

`dataset/preprocess_endcaps.py` derives cap patches automatically: every
vertex within 1mm graph (edge-length) distance of a ray-crossing endpoint.
That's a reasonable proxy but a fixed radius doesn't adapt to vessel
calibre — it over-covers thin vessels and under-covers wide ones. This tool
lets a human brush the real region directly, for:

- **Real cases** — refining/replacing the automatic label, with the real
  centerline and the automatic guess shown as reference.
- **Synthetic cases** — which have no automatic label at all (no real
  centerline exists for a freshly generated shape), using the frozen GHD-VAE
  to sample a mesh and the fixed canonical opening indices as an approximate
  branch-identity guide.

## Setup

```bash
conda activate new          # or your equivalent env
pip install pyvista         # needs a real display / working GL context —
                             # will NOT work over plain SSH without X forwarding
                             # or a virtual framebuffer
```

Needs the repo's `dataset/processed/` (real cases), `dataset/processed_endcaps/`
(automatic reference labels), and — for synthetic mode only —
`dataset/canonical/` plus a trained GHD-VAE checkpoint. All three are
gitignored, so copy them over from this machine separately; they won't come
with `git clone`/`git pull`.

Default synthetic-mode checkpoint path (edit `GHD_VAE_CKPT` near the top of
the script if yours lives elsewhere):
```
tr_checkpoints/v2_1/stage1/ghd_vae_h512_z16_kl0.5/epoch_05000.pth
```

## Running it

```bash
# Real cases — labels every case in dataset/processed/ not already in the output dir
python dataset/label_endcaps.py --mode real

# Just a few specific cases
python dataset/label_endcaps.py --mode real --case CASE_ID_1 --case CASE_ID_2

# Synthetic cases — samples 20 fresh shapes from the GHD-VAE
python dataset/label_endcaps.py --mode synthetic --n-synthetic 20 --seed 0
```

All paths default to the repo layout but are overridable:
`--real-dir`, `--endcaps-dir`, `--output-dir`, `--device` (synthetic mode
only — which device to run the GHD-VAE on).

## Controls

One branch is labeled per window (2 windows for type 1/2 sidewall cases, 3
for type 0 bifurcation cases — closes and reopens automatically between
branches):

| action | key/mouse |
|---|---|
| select faces | drag a rectangle over the mesh |
| add more faces to the selection | drag again (accumulates) |
| clear this branch's selection | `r` |
| confirm this branch and move on | `c` |
| skip this branch/case | close the window without pressing `c` |

Skipping doesn't save anything for that case — rerun later to retry it (a
case's output is only written once **all** of its branches are confirmed).
Already-labeled cases (present in `--output-dir`) are skipped automatically
on the next run, so it's safe to stop and resume across sessions.

## What you'll see while brushing

- **Real cases**: the real centerline for that branch (black/blue/red by
  branch index, matching every other sanity image in this project), the
  automatic endpoint + tangent (yellow arrow), and the automatic patch
  (faint dots) — so you're correcting a guess, not starting from nothing.
- **Synthetic cases**: faint dots at the fixed canonical opening indices for
  that branch slot, labeled as approximate. These come from the same
  fixed-index assumption that motivated building a *learned* endcap
  predictor in the first place — treat them only as "this is roughly branch
  0 / branch 1 / branch 2," not as ground truth.
- Your current selection is always highlighted **orange**.

## What gets saved

One `.npy` per case in `--output-dir` (default
`dataset/processed_endcaps_manual/`), schema-compatible with
`dataset/endcap_dataset.py`'s `EndcapDataset`:

```python
{
    "case": str,
    "aneurysm_type": int,
    "phi": np.ndarray[float32],           # GHD coefficients
    "endpoints": np.ndarray[3, 3] float32,  # per-branch, padded to max_branches
    "tangents":  np.ndarray[3, 3] float32,
    "branch_mask": np.ndarray[3] bool,
    "in_patch": np.ndarray[3, N] bool,      # N = vertex count for that aneurysm_type
}
```

`endpoint` = centroid of the brushed patch's vertices. `tangent` = SVD-based
outward surface normal of the patch (same technique used elsewhere in this
codebase, e.g. `MultiCanonicalGHDReconstruct.reconstruct_fused_mesh`) — the
cap is roughly a disk cutting across the vessel tube, so the disk's normal
approximates the vessel's own axial direction.

This is a **separate output directory** from `dataset/processed_endcaps/`
(the automatic labels) — nothing gets silently overwritten. Point
`EndcapDataset` at whichever directory (or a merge of both) you want to
train against.

## Known rough edge — flag if you hit it

`_on_pick` in `brush_one_branch` reads picked face indices from
`picked.cell_data["vtkOriginalCellIds"]`. This is the standard PyVista/VTK
key for `enable_cell_picking(..., through=False)`, but it's the one call in
this script most likely to need a version-specific tweak. If nothing
highlights when you drag-select, check the terminal — a warning prints
`picked.array_names` (the actual keys available); swap the key in
`_on_pick` (near the top of `label_endcaps.py`) accordingly.
