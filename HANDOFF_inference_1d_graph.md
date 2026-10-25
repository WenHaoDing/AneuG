# Handoff: 1D vessel graph from an artery mask (inference stage)

For the agent working on the image-generation project
(`/media/yaplab2/HDD Storage/wenhao/image-generation`). It describes a task that must be done
**before** the generative models are built: turn a 3D artery mask into the same 1D vessel
graph that the models will be trained on.

## 1. Why this is needed

The plan for aneurysm synthesis (details in the AneuG repo,
`TASK_vessel_conditioned_aneurysm.md`):

```
artery mask (no aneurysm) -> 1D vessel graph -> [model 1] ostium -> [model 2] aneurysm sac
                                                      (both conditioned on the 1D graph)
-> voxelise the sac, union with the artery mask -> image model inpaints the synthetic scan
```

The graph is a skeleton of the vessel tree: centerline points every 0.1 mm, each with a
radius, plus the connectivity. It is the condition for both generative models.

**The problem.** The training graphs come from curated surface meshes of real aneurysm
cases, with manually checked endpoints. At inference we only have a **3D artery mask** (GT
or nnU-Net), so the graph must be extracted automatically. If that graph differs
systematically from the training graphs, the models will not generalise. Examples: other
endpoints, another crop extent, another radius definition, a different smoothness.

**The task.** Build `mask -> 1D graph` for inference and **show** that its graphs match the
training graphs on the same cases (§6).

## 2. What already exists (AneuG repo)

Repo: `/media/yaplab2/HDD Storage/wenhao/AneuG`.

**`geometry/extract_skeleton.py`.** `extract_skeleton(mesh, endpoints, source_id=0,
zone_pts=None, smooth_mm=1.0, smooth_pos_mm=0.5)` takes a surface mesh plus endpoints and
returns the graph. Run it in an env with VMTK:
`/home/yaplab2/miniconda3/envs/vmtk310/bin/python`. Steps, which the inference path must
reproduce:
1. **VMTK centerlines** (`vmtkCenterlines`) from `endpoints[source_id]` to every other
   endpoint, resampled to 0.1 mm and smoothed (100 iterations × 0.1). These are the same
   settings as the AneuSeg pipeline that made the training data.
2. **Radius** = VMTK's `MaximumInscribedSphereRadius`: the inscribed sphere, not an
   equivalent-area radius.
3. *(training only)* With `zone_pts` (the removed sac), the radius across the aneurysm
   region is replaced by a C1 Hermite interpolation. At inference `zone_pts=None`.
4. **Graph** from `vmtkBranchExtractor`'s own topology (line and tract order, no distance
   tolerance). Bifurcation pieces are kept, small gaps between VMTK lines are filled, and
   degenerate specks are dropped.
5. **Smoothing over the graph:** positions σ = 0.5 mm (vessel ends fixed), radius
   σ = 1 mm.
6. The CLI writes `skeleton_graph.npy` (per-edge arrays) and **`skeleton_graph_points.npz`**
   (the model input):

   | key | shape | meaning |
   |---|---|---|
   | `pos` | (N, 3) | points in mm, 0.1 mm apart |
   | `radius` | (N,) | condition radius (smoothed) |
   | `edge_index` | (2, E) | adjacency, both directions |
   | `branch` | (N,) | branch id 1..B (breadth-first from the source end, so the root is 1) |
   | `branch_id` | (N,) | same, but 0 in the aneurysm region (all = `branch` at inference) |
   | `branch_onehot` | (N, B+1) | one-hot of `branch_id` |
   | `aneurysm`, `is_junction`, `is_end`, `extracted` | (N,) | masks |
   | `meta` | | point and branch counts, root point, smoothing scales, units, frame |

   Load it with `from geometry import load_point_graph`.

Speed: about 1 s for a 10k-face mesh (VMTK centerlines 0.7 s).

Examples on 9 ImperialNHS cases: `AneuG/runtime_skeleton/<case>/`.
- `skeleton_graph_points.npz`: the graph.
- `graph_1d.png`: the graph coloured by radius, with close-ups of every bifurcation.
- `input/`: the case's `label.nrrd`, meshes and `endpoints_manual.npy`.

**AneuSeg** (`/media/yaplab2/HDD Storage/wenhao/AneuSeg`) made the training meshes, and
two of its parts are useful here:
- `IAgents/tools/reconstruction.py :: forward_marching_cubes`. **This is how the training
  surfaces were made:** marching cubes of the mask at **level 0.7** (slightly inside),
  `spacing` from the image, vertices placed at `origin + ijk * spacing` (**no direction
  matrix**), then **25 Laplacian iterations** (relaxation 0.1).
- `IAgents/tools/centerline_extraction.py`:
  - `extract_endpoints`: `skeletonize_3d` of a mask, leaves of the 26-connected skeleton,
    with branches shorter than 2 voxels pruned. This is how the automatic endpoints were
    proposed.
  - `extract_boundary_endpoints`: centroids of the open boundary rings of a clipped mesh;
    the source is the ring with the largest radius.

## 3. How the training graphs are made (what inference must match)

Per real aneurysm case (`vessel_edit/skeleton_pipeline.py`):
1. **Surface:** `reconstruction4clipping.ply`, a local mesh region-grown **from the sac**
   into the arteries (about 5k voxels), marching cubes as above. It covers the sac and the
   parent vessels up to about the nearest bifurcations.
2. **Endpoints:** `endpoints_manual.npy`, proposed by skeleton leaves and **checked by
   hand**. The source is endpoint 0.
3. **Patch:** the sac (label 2) plus a 1 mm collar are removed and the vessel is patched,
   so VMTK's line is not pulled into the sac.
4. `extract_skeleton(patched mesh, endpoints, zone_pts=sac)`.

## 4. Inference options

**A. Mask → surface → automatic endpoints → VMTK → `extract_skeleton` (recommended).**
1. Choose the crop: a connected vessel region (§5.1).
2. Marching cubes exactly as AneuSeg (`level 0.7`, `origin + ijk * spacing`, 25 Laplacian
   iterations), keeping the largest component.
3. Automatic endpoints:
   - **(i)** For a box crop: open the mesh where it is cut by the crop box and use the ring
     centroids (`extract_boundary_endpoints`; source = largest ring).
   - **(ii)** For a closed mesh: skeleton leaves of the mask (`extract_endpoints`), with
     pruning.
   - Either way the source must be chosen consistently, because branch ids are numbered
     from it. Use the largest-radius end, which is usually the inlet.
4. `extract_skeleton(mesh, endpoints)` with `zone_pts=None`.

This reuses the training code end to end, so the only new part is the endpoint picking.

**B. Thinning only** (`skeletonize` of the mask, radius from the Euclidean distance
transform). Simpler and has no VMTK dependency, but:
- the radius is quantised to the voxel scale (0.35–0.7 mm voxels against 0.5–2 mm radii);
- skeletons have spurs and staircase steps;
- the inscribed-sphere radius of VMTK is not the same quantity.

It would need its own smoothing and pruning, and it would differ from the training graphs.
Use thinning for **endpoint detection** (A-ii), not as the graph itself, unless §6 shows it
matches.

## 5. Generalisation risks to check

1. **Crop extent.** Training meshes are grown from the sac, so the aneurysm always sits
   near the middle of the graph. At inference there is no sac; the crop will be different.
   The ostium model could learn "ostium = middle of the graph". **Fix it on both sides with
   one shared crop rule**, for example the vessel tree inside the image model's patch (96³
   core + context) or within R mm of a point. The training graphs will then be regenerated
   with that rule (§7).
2. **Endpoints.** Manual versus automatic. Different endpoints give different branches and a
   different source, so different branch ids. Spurious short leaves must be pruned the same
   way on both sides.
3. **Mask source.** Training masks are manual labels (Dataset014). Inference uses GT or
   nnU-Net, whose masks differ in thickness and smoothness. Compare radius statistics.
4. **Frame.** The training meshes and graphs use `origin + ijk * spacing` with **no
   direction matrix**, not the scanner frame with its direction cosines. Whatever frame
   inference uses, it must be the same as training, or converted, and recorded in `meta`.
5. **Radius definition and smoothing.** Keep VMTK's inscribed-sphere radius, and the same
   σ (positions 0.5 mm, radius 1 mm).
6. **The sac in training data.** Training graphs come from vessels that had a sac (removed
   and patched). Inference vessels never had one. Interpolation removes the radius
   inflation, but local geometry near the neck may still differ slightly.

## 6. Validation (before anything is used for training)

On the ImperialNHS cases (start with the 9 in `AneuG/runtime_skeleton/`; `label.nrrd` is
in each `input/`):
1. Run the inference path on the **artery mask with the sac voxels removed**:
   `label == 1` only. Note: marching cubes of label 1 alone closes over the neck where the
   expert drew it, which is a reasonable "healthy" surface for a skeleton.
2. Compare with the training graph of the same case:
   - **Topology:** number of branches and bifurcations, matched junction positions (mm).
   - **Geometry:** symmetric mean and max point-to-curve distance (mm).
   - **Radius:** mean absolute difference along matched points, overall and near the
     aneurysm region.
   - **Endpoints:** automatic against manual (count, distance).
3. Save side-by-side renders per case (graph coloured by radius, as in `graph_1d.png`).
   The project lead checks them by eye.
4. Then try GT versus nnU-Net masks on the same scans, to size the mask-source shift.

## 7. Recommendation

**Use the same extractor for training and inference.** Once the mask-based path (A) is
validated, regenerate the training graphs from the masks with it, instead of from the
curated meshes and manual endpoints:
- training: `label == 1` mask → path A → `extract_skeleton(..., zone_pts=sac)`, so the
  radius is interpolated over the aneurysm;
- inference: artery mask → path A → `extract_skeleton(...)`.

Train and test graphs then differ only by the sac. This may also make the mesh patching
step unnecessary for training. That needs checking: VMTK on the label-1 surface, which is
closed over the drawn neck, may not be pulled into the sac at all.

## 8. Deliverable

A function and CLI, `mask_to_graph(mask, spacing, origin, [crop], [sac_mask]) ->
point graph`. It should write the same `skeleton_graph_points.npz` format (§2.6), plus:
- a short report of the §6 comparison (numbers and renders);
- the crop rule chosen, so the AneuG side can regenerate the training graphs with it.

## 9. Rules

- The geometry dataset `/media/yaplab2/wd8tb/wenhao/angioflow/data/geometry/` is
  **read-only**. Copy cases into your own working folder.
- Put renders in each case's output folder, for the project lead to check by eye.
