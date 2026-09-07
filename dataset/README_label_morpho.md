# Manual endcap labeling — `label_morpho.py`

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

- **Real cases** — refining the cap region only, with the real centerline
  and the automatic guess shown as reference. Endpoint and tangent are NOT
  touched: they already come from ray-crossing along the real patient
  centerline (`preprocess_endcaps.py`), which is better-grounded than the
  centroid/normal of a hand-dragged patch. Brushing exists to fix the
  fixed-radius cap-region proxy, not to replace ground-truth geometry.
- **Synthetic cases** — which have no automatic label at all (no real
  centerline exists for a freshly generated shape), using the frozen GHD-VAE
  to sample a mesh and the fixed canonical opening indices as an approximate
  branch-identity guide. Here the brush IS the only source, so endpoint and
  tangent are derived from it too (see "What gets saved").

## Setup

```bash
conda activate new          # or your equivalent env
pip install pyvista         # needs a real display / working GL context —
                             # will NOT work over plain SSH without X forwarding
                             # or a virtual framebuffer
```

Needs the repo's `runtime/dataset/processed/` (real cases),
`runtime_dataset/AneuG_morpho/` (automatic reference labels), and — for
synthetic mode only — `dataset/canonical/` (small, checked into git, comes
with `git clone`/`git pull` as normal) plus a trained GHD-VAE checkpoint under
`runtime/tr_checkpoints/`.

Everything under `runtime/` is gitignored and lives in one folder specifically
so it can be synced with a single rsync — copy (or rsync) that whole folder
over from this machine; it won't come with `git clone`/`git pull`:
```bash
rsync -avz --progress <this-machine>:"AneuG/runtime/" ./runtime/
```
You don't need the entire folder — for this script you only need
`runtime/dataset/processed/`, `runtime_dataset/AneuG_morpho/`, and (synthetic
mode only) the one GHD-VAE checkpoint below.

Default synthetic-mode checkpoint path (edit `GHD_VAE_CKPT` near the top of
the script if yours lives elsewhere):
```
runtime/tr_checkpoints/v2_1/stage1/ghd_vae_h512_z16_kl0.5/epoch_05000.pth
```

## Running it

```bash
# Real cases — labels every case in runtime/dataset/processed/ not already manually labeled
python dataset/label_morpho.py --mode real

# Just a few specific cases
python dataset/label_morpho.py --mode real --case CASE_ID_1 --case CASE_ID_2

# Synthetic cases — samples 20 fresh shapes from the GHD-VAE
python dataset/label_morpho.py --mode synthetic --n-synthetic 20 --seed 0
```

All paths default to the repo layout but are overridable:
`--real-dir`, `--endcaps-dir` (real mode), `--synthetic-dir` (synthetic mode),
`--device` (synthetic mode only — which device to run the GHD-VAE on).

## Controls

One branch is labeled per window (2 windows for type 1/2 sidewall cases, 3
for type 0 bifurcation cases — closes and reopens automatically between
branches):

| action | key/mouse |
|---|---|
| select faces | **left-click and drag** a box over the mesh (not a single click) |
| add more faces to the selection | drag another box elsewhere (accumulates, doesn't replace) |
| clear this branch's selection | `z` |
| confirm this branch and move on | `c` |
| skip this branch/case | close the window without pressing `c` |

Not `r` for clearing — PyVista's box-select interactor style already binds
`r` itself (it toggles between camera-rotate and box-select mode), so `z` is
used instead to avoid the two colliding. The window also shows PyVista's own
built-in picking hint on screen, which is the authoritative description for
whatever version you have installed if this differs slightly.

Skipping doesn't save anything for that case — rerun later to retry it (a
case's record is only updated once **all** of its branches are confirmed).
Already-labeled cases (their record already has `manual_in_patch`) are
skipped automatically on the next run, so it's safe to stop and resume
across sessions.

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

No separate output folder. For a **real** case, brushing writes back into the
SAME `runtime_dataset/AneuG_morpho/<case>.npy` record that
`preprocess_endcaps.py` already produced, adding only `manual_in_patch` —
the automatic `endpoints` / `tangents` / `branch_mask` are left exactly as
they were:

```python
{
    # ...existing automatic fields, UNCHANGED: endpoints, tangents, branch_mask, in_patch...
    "manual_in_patch": np.ndarray[3, N] bool,   # N = vertex count for that aneurysm_type
}
```

(Rare fallback: if a real case somehow has no automatic record at all — every
case should, since `preprocess_endcaps.py` covers the whole real dataset —
there's nothing to defer to, so that one case gets a full `manual_endpoints`
/ `manual_tangents` / `manual_branch_mask` / `manual_in_patch` set derived
from the brush instead, as the only available source.)

A **synthetic** case has no automatic record and no real case in
`runtime/dataset/processed/` to attach to — a genuinely different population,
not a refinement of an existing one — so it's saved into its OWN folder,
`runtime_dataset/AneuG_morpho_synthetic/synthetic_seed{seed}_{i}.npy`,
using the plain (non-`manual_`-prefixed) field names directly. Here the brush
IS the endpoint/tangent source (there's no ray-crossing centerline to defer
to), so all four fields come from it. Keeping synthetic cases out of
`AneuG_morpho/` also means a real `preprocess_endcaps.py` rerun can't
accidentally interact with them.

`endpoint` = centroid of the brushed patch's vertices. `tangent` = SVD-based
outward surface normal of the patch (same technique used elsewhere in this
codebase, e.g. `MultiCanonicalGHDReconstruct.reconstruct_fused_mesh`) — the
cap is roughly a disk cutting across the vessel tube, so the disk's normal
approximates the vessel's own axial direction. This derivation is used for
synthetic cases' endpoint/tangent (their only source) and for every case's
`in_patch` — never for a real case's endpoint/tangent.

`dataset/morpho_dataset.py`'s `MorphoDataset` prefers `manual_in_patch` over
automatic `in_patch` when present, but always uses the automatic
endpoints/tangents/branch_mask when they exist, and accepts a list of roots —
so a labeled real case is picked up for training automatically, and combining
real + synthetic is one line:
```python
MorphoDataset([
    ROOT / "runtime_dataset" / "AneuG_morpho",
    ROOT / "runtime_dataset" / "AneuG_morpho_synthetic",
])
```

## Known rough edge — flag if you hit it

`_on_pick` in `brush_one_branch` reads picked face indices from
`picked.cell_data["vtkOriginalCellIds"]`. This is the standard PyVista/VTK
key for `enable_cell_picking(..., through=False)`, but it's the one call in
this script most likely to need a version-specific tweak. If nothing
highlights when you drag-select, check the terminal — a warning prints
`picked.array_names` (the actual keys available); swap the key in
`_on_pick` (near the top of `label_morpho.py`) accordingly.
