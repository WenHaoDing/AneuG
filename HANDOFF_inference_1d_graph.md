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

   Load it with `from geometry import load_point_graph`; write it with `save_point_graph`.

Speed: about 1 s for a 10k-face mesh (VMTK centerlines 0.7 s).

**Training graphs, 522 cases:** `AneuG/runtime_dataset/assembled/<dataset>/<case>/vessel_graph/`
(`dataset` = `ImperialNHS` or `AneuX`), written by `dataset/preprocess_vessel_graph.py`.
- `vessel_graph.npz`: the table above, plus training-only targets (`rim`, `site_point`,
  `geo_dist_from_site`, `ostium_*`). Inference does not produce these.
- `skeleton_sanity.png`: the graph coloured by radius, with close-ups of the bifurcations.
- `patch.png`: the patched vessel.
- `meta.json`: includes `geometry_dir`, the case's source folder in the geometry dataset
  (`label.nrrd`, meshes, `endpoints_manual.npy`; read-only, see §9).

### 2.1 What the ostium model actually reads (the contract)

Of the keys above, the model uses only three. The others can be written for inspection, but
must not be needed:

| key | requirement |
|---|---|
| `pos` (N, 3) float | **mm**, in the frame `origin + ijk * spacing` (no direction matrix; §5.4). Points about **0.1 mm apart** along each centerline, as `extract_skeleton` resamples them. |
| `radius` (N,) float | **mm**, VMTK's maximum inscribed sphere radius, smoothed over the graph (σ = 1 mm). |
| `edge_index` (2, E) int | undirected: every edge stored in both directions; one connected component, no cycles, no self-loops. |

The model derives everything else itself from these three:
- node degree: junctions are degree ≥ 3, ends are degree 1;
- sign-free tangents from the edges;
- the signed tangent at a site, oriented away from the first degree-1 point;
- the coarse levels (farthest-point sampling at about 1 mm and 4 mm) and their positional
  encodings;
- the centring (positions are centred on their mean, so absolute translation does not matter).

Not used by the model: `branch`, `branch_id`, `branch_onehot`, `aneurysm`, `extracted`, `meta`.

**Output side, for reference:** the model returns, in the same mm frame as `pos`:
- a site on the graph;
- an ostium ring (64 points);
- a placed GHD complex mesh;
- the sensor's dome patch.

The dome patch is what gets voxelised and merged (union) into the patch mask. So the graph's
frame must map back to the image patch's voxels exactly: keep `origin` and `spacing` with
the graph.

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

Per real aneurysm case (`dataset/preprocess_vessel_graph.py`, both datasets):
1. **Surface:** `reconstruction.ply`, the case's full artery mesh. It is the mesh the manual
   endpoints were picked on. AneuX meshes with a mean edge below 0.25 mm are decimated
   first.
2. **Endpoints:** `endpoints_manual.npy`, proposed by skeleton leaves and **checked by
   hand**. The source is endpoint 0. Any endpoint that would seed on the sac is dropped.
3. **Patch:** the sac (taken from the case's placed GHD fit, its dome faces) is removed and the
   vessel is patched, so VMTK's line is not pulled into the sac.
4. `extract_skeleton(patched mesh, endpoints, zone_pts=sac)`. The radius is interpolated
   across the aneurysm region, then everything is smoothed.

**Training-time crops.** The model never sees these whole graphs as they are. Each training
sample is randomly trimmed off-centre along the graph; 20% of samples keep the whole graph.
Every crop keeps all points within 6 mm of the ostium (and within 15 mm of the site along the
graph). Over 600 crops:
- the vessel kept has a median of 52 mm of centerline (5th–95th percentile: 26–130 mm);
- the nearest cut end sits a median of 13.5 mm from the site (5th percentile: 7 mm).

Crops then get a random rotation, a random tangent sign and a 0.3 mm site jitter. Whether the
rotation is kept is being tested now (§5.4).

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

1. **Crop extent.** The training whole graphs are centred on the sac. The model is therefore
   trained on random off-centre crops (§3), so it should not rely on "ostium = middle of the
   graph". At inference, the graph is whatever vessel lies in the image model's patch.
   - It should stay within the training crop range: at least about 25 mm of centerline, and
     typically 40–130 mm.
   - Any place where the generated aneurysm could go should be at least about 7 mm (along the
     vessel) from the patch boundary, where the vessel is cut.
   - A cut end at the patch boundary is fine: training crops have cut ends too.
   - Report the centerline length of each inference graph, so this can be checked.
2. **Endpoints.** Manual versus automatic. Different endpoints give different branches and a
   different source, so different branch ids. Spurious short leaves must be pruned the same
   way on both sides.
3. **Mask source.** Training masks are manual labels (Dataset014). Inference uses GT or
   nnU-Net, whose masks differ in thickness and smoothness. Compare radius statistics.
4. **Frame (now critical).** The training meshes and graphs use `origin + ijk * spacing` with
   **no direction matrix**, not the scanner frame with its direction cosines. The axis order
   is i, j, k as in the arrays, as in AneuSeg's marching cubes. Inference must use exactly
   the same convention, including axis order and sign, and record it in `meta`.
   - An ablation running now suggests the ostium model learns much better **without random
     rotation**. In that case it relies on the orientation of the graph, so a swapped or
     flipped axis at inference would silently break it.
   - Before relying on this, check one case end to end: the graph you produce from its
     `label.nrrd` should overlay the training graph in `vessel_graph.npz`, point by point.
5. **Radius definition and smoothing.** Keep VMTK's inscribed-sphere radius, and the same
   σ (positions 0.5 mm, radius 1 mm).
6. **The sac in training data.** Training graphs come from vessels that had a sac (removed
   and patched). Inference vessels never had one. Interpolation removes the radius
   inflation, but local geometry near the neck may still differ slightly.

## 6. Validation (before anything is used for training)

On the ImperialNHS cases (start with about 10 of those in
`AneuG/runtime_dataset/assembled/ImperialNHS/`; each case's `vessel_graph/meta.json` gives
its `geometry_dir`, where its `label.nrrd` is):
1. Run the inference path on the **artery mask with the sac voxels removed**:
   `label == 1` only. Note: marching cubes of label 1 alone closes over the neck where the
   expert drew it, which is a reasonable "healthy" surface for a skeleton.
2. Compare with the training graph of the same case:
   - **Topology:** number of branches and bifurcations, matched junction positions (mm).
   - **Geometry:** symmetric mean and max point-to-curve distance (mm).
   - **Radius:** mean absolute difference along matched points, overall and near the
     aneurysm region.
   - **Endpoints:** automatic against manual (count, distance).
3. Save side-by-side renders per case (graph coloured by radius, as in each case's
   `vessel_graph/skeleton_sanity.png`).
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
point graph`. It should write the point-graph `.npz` with `save_point_graph` (§2.6): at least
`pos`, `radius` and `edge_index` as in §2.1, plus `meta` recording `origin`, `spacing`, the
frame convention and the patch box, so the dome can be voxelised back into the patch. Also:
- a short report of the §6 comparison (numbers and renders);
- the centerline length of each graph and the crop rule used (§5.1).

## 9. Rules

- The geometry dataset `/media/yaplab2/wd8tb/wenhao/angioflow/data/geometry/` is
  **read-only**. Copy cases into your own working folder.
- Put renders in each case's output folder, for the project lead to check by eye.
