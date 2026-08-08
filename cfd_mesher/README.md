# CFD Mesher

Turns an aneurysm complex plus its clipped centreline into meshes a flow solver can run.

Two stages:

1. **Surface** — propagate the vessel branches outward along the clipped centreline, fuse
   them into the complex, smooth the seams, and write a surface that is closed except at
   the inlet and outlets. *(implemented here)*
2. **Volume** — mesh that surface into tetrahedra with vmtk, tagging wall / inlet / outlet
   boundaries for the solver. *(to be moved in)*

```bash
python -m cfd_mesher.generate_cfd_surface_meshes                 # batch, no prompts
python -m cfd_mesher.generate_cfd_surface_meshes --no-automatic  # tune per case
python -m cfd_mesher.generate_cfd_surface_meshes \
    --post_dir /path/to/PostTr --ghd_dir /path/to/fitting_results \
    --save_root /path/to/out --mesh_source ghd --overwrite
```

## Inputs

Per case, `--post_dir/<case>/` supplies the centreline and fusion bookkeeping:

| file | role |
|---|---|
| `clipped_centerline.npy` | centreline the branches are propagated along |
| `forward_fusion_info.npz` | where each branch attaches to the complex |
| `branch_ranking.npy` | branch ordering (which is inlet, which are outlets) |

The complex surface itself comes from one of two sources, chosen with `--mesh_source`:

- **`ghd`** (default) — `--ghd_dir/<case>/ghd_fitted_uncapped_world.obj`, the GHD-fitted
  mesh. Smooth and topologically regular, since it is a deformed template.
- **`clipped`** — `--post_dir/<case>/clipped_reconstruction.ply`, straight from the
  segmentation. Follows the image more closely but carries its noise.

## Outputs

Written to `--save_root/<case>/`, one folder per case named by the case:

| file | role |
|---|---|
| `ghd_smoothed_reconstruction.obj` | **the deliverable** — seams and openings smoothed |
| `ghd_merged_reconstruction.obj` | complex fused with branches, before smoothing |
| `ghd_reconstructed.obj` | propagated complex before fusion |
| `ghd_forward_fusion_info.npz` | fusion bookkeeping for downstream stages |
| `ghd_fusion_params.json` | the parameters this case actually used |

Cases whose smoothed mesh already exists are skipped unless `--overwrite` is passed, so an
interrupted batch resumes cheaply. Failures are collected in
`surface_mesh_failures.json` rather than stopping the run.

## Options worth knowing

`--extrude_outlets` (on by default) extends the outlets straight, giving the solver a
developed-flow length so the outflow boundary condition is not imposed on a still-turning
vessel. `--extrude_inlet` does the same at the inlet, off by default.

`--no-automatic` restores the interactive loop: after each case you can inspect the mesh
and re-run the propagation with adjusted parameters (`flaw_opening_min_size`, `init_step`,
`max_cl_length`, `ring_downsample_ratio`, `smooth_n_rings`, `smooth_n_iter`).

## Requirements

**pyvista must be new enough to write `.obj`.** `vessel_reconstruct` saves the merged and
smoothed surfaces through `pyvista.PolyData.save`, which only gained `.obj` support in
later releases — 0.47 works, 0.42 raises `Invalid file extension`. The script probes this
at startup and refuses with a clear message rather than failing identically on every case
partway through. During development this ran under the `vmtk_autogen` environment.

```bash
conda activate vmtk_autogen
python -m cfd_mesher.generate_cfd_surface_meshes --help
```

Otherwise: numpy, pyvista and networkx. The volume stage will add vmtk, vtk and open3d.

## Status

Surface stage is complete and verified end to end — `--mesh_source clipped` regenerates
meshes matching the existing pipeline output.

The volume stage is not moved in yet. It will bring `vmtk_cfdmesher`, `vmtk_vtu2msh` and
`mesh_utils`, plus the vmtk/vtk/open3d dependencies.

### Changes from the original

Rewritten from `aneux_forward_mesh_fusion.py`, which was AneuX-specific and hardcoded its
paths. Beyond the argparse interface:

- **Fixed a fatal bug.** The original referenced `ghd_dir` at two places without ever
  assigning it, so it raised `NameError` on the first case. It is now `--ghd_dir`.
- **Parameters are saved next to the mesh they produced.** The original wrote
  `ghd_fusion_params.json` into the *GHD fitting* directory rather than the output
  directory, leaving outputs undocumented and mixing state into an input folder.
- **Both mesh sources in one script.** The AneuX and ImperialNHS variants differed only in
  which surface they propagated from; `--mesh_source` covers both.
- Failures no longer abort the batch, and case selection/exclusion is available from the
  command line.

## Layout

```
generate_cfd_surface_meshes.py   surface stage (entry point)
vessel_reconstruct.py            branch propagation and fusion
patching.py                      opening planarisation, seam smoothing
config.py                        all paths, overridable by CFDMESH_* env vars
```
