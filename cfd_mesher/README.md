# CFD Mesher

Turns an aneurysm complex plus its clipped centreline into meshes a flow solver can run.

Two stages:

1. **Surface** — propagate the vessel branches outward along the clipped centreline, fuse
   them into the complex, smooth the seams, and write a surface that is closed except at
   the inlet and outlets.
2. **Volume** — mesh that surface into tetrahedra with vmtk, tagging wall / inlet / outlet
   boundaries for the solver, and exporting the bookkeeping a Fluent run needs.

```bash
python -m cfd_mesher.generate_cfd_surface_meshes                 # batch, no prompts
python -m cfd_mesher.generate_cfd_surface_meshes --no-automatic  # tune per case
python -m cfd_mesher.generate_cfd_surface_meshes \
    --post_dir /path/to/PostTr --ghd_dir /path/to/fitting_results \
    --save_root /path/to/out --mesh_source ghd --overwrite

python -m cfd_mesher.generate_cfd_volume_meshes                  # reads the surface stage's output
python -m cfd_mesher.generate_cfd_volume_meshes \
    --surface_root /path/to/out --overwrite
```

## Surface stage

### Inputs

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

### Outputs

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

### Options worth knowing

`--extrude_outlets` (on by default) extends the outlets straight, giving the solver a
developed-flow length so the outflow boundary condition is not imposed on a still-turning
vessel. `--extrude_inlet` does the same at the inlet, off by default.

`--no-automatic` restores the interactive loop: after each case you can inspect the mesh
and re-run the propagation with adjusted parameters (`flaw_opening_min_size`, `init_step`,
`max_cl_length`, `ring_downsample_ratio`, `smooth_n_rings`, `smooth_n_iter`).

## Volume stage

### Inputs

Per case, `--surface_root/<case>/` — the surface stage's own output folder, read directly:

| file | role |
|---|---|
| `ghd_smoothed_reconstruction.obj` | the closed-except-openings surface to tetrahedralize |
| `ghd_forward_fusion_info.npz` | `opening_centroids` (extension) and `cpcd_glo` branch endpoints (relabeling) |

### Outputs

Written to `--save_root/<case>/`, which defaults to `--surface_root/<case>/` itself (in
place, since the inputs are already sitting there):

| file | role |
|---|---|
| `mesh.msh` | **the deliverable** — Fluent-format volume mesh |
| `mesh.vtu` | the same mesh, tagged wall=3 / inlet=4 / outlet=5[,6] |
| `ghd_surface.vtp` | the input `.obj` converted to `.vtp` |
| `ghd_surface_extended.vtp` | only written when `--extend_inlet`/`--extend_outlets` is set |
| `inlet_centroids.csv` | every inlet node's coordinates, for a Fluent velocity-profile UDF |
| `flowsplit_ratio.txt` | two-outlet cases only: normalized per-outlet flow-split ratio |

Cases whose `mesh.vtu` and `mesh.msh` already both exist are skipped unless `--overwrite`
is passed. Relabeling and the flow-split/inlet-CSV exports always (re)run when a case
isn't fully skipped, since they're cheap relative to tetrahedralization itself. Failures
are collected in `volume_mesh_failures.json`.

### Options worth knowing

`--extend_inlet`/`--extend_outlets` (both off by default, matching how this pipeline is
actually run) add a developed-flow length via `vtkvmtkPolyDataFlowExtensionsFilter`
*after* fusion/smoothing, at the open boundaries themselves — a different mechanism from
the surface stage's `--extrude_outlets`, which extends the centreline *before* fusion.
Both exist because they were tuned independently; using both at once has not been
validated.

`--base_edge`/`--ref_volume`/`--ref_area`/`--vol_exponent`/`--area_exponent`/`--max_edge`
control `volume_mesh.compute_adaptive_edge` — element size scales with whichever of the
shape's enclosed volume or surface area implies the coarser edge, so a small-volume but
elongated/thin shape still gets caught. The defaults are the values this pipeline was
actually tuned and run with, not that function's own (gentler) fallback defaults.

## Requirements

**Surface stage: pyvista must be new enough to write `.obj`.** `vessel_reconstruct` saves
the merged and smoothed surfaces through `pyvista.PolyData.save`, which only gained
`.obj` support in later releases — 0.47 works, 0.42 raises `Invalid file extension`. The
script probes this at startup and refuses with a clear message rather than failing
identically on every case partway through.

**Volume stage: the vmtk CLI must be on PATH**, not just the `vmtk` Python package —
`vmtkmeshgenerator` is shelled out to (see `vmtk_backend.cfdmesher_custom`). The script
probes this at startup too.

Both stages were developed and run under the `vmtk_autogen` conda environment:

```bash
conda activate vmtk_autogen
python -m cfd_mesher.generate_cfd_surface_meshes --help
python -m cfd_mesher.generate_cfd_volume_meshes --help
```

Surface stage needs numpy, pyvista and networkx. Volume stage additionally needs vmtk,
vtk, pandas and the vmtk CLI tools.

## Status

Both stages are moved in. Surface stage is verified end to end — `--mesh_source clipped`
regenerates meshes matching the existing pipeline output. Volume stage has not yet been
run end to end against real cases in this repo (only import/CLI-level checks so far);
treat it as ported-but-unverified until a batch has actually been run through vmtk.

### Changes from the original

**Surface stage**, rewritten from `aneux_forward_mesh_fusion.py`, which was AneuX-specific
and hardcoded its paths. Beyond the argparse interface:

- **Fixed a fatal bug.** The original referenced `ghd_dir` at two places without ever
  assigning it, so it raised `NameError` on the first case. It is now `--ghd_dir`.
- **Parameters are saved next to the mesh they produced.** The original wrote
  `ghd_fusion_params.json` into the *GHD fitting* directory rather than the output
  directory, leaving outputs undocumented and mixing state into an input folder.
- **Both mesh sources in one script.** The AneuX and ImperialNHS variants differed only in
  which surface they propagated from; `--mesh_source` covers both.
- Failures no longer abort the batch, and case selection/exclusion is available from the
  command line.

**Volume stage**, ported from `AneuSeg/cfd_meshing/get_mesh_aneux_tune_elements_v3.py`
(plus its `vmtk_cfdmesher.py`/`vmtk_vtu2msh.py` dependencies) into an argparse batch
driver matching the surface stage's conventions, rather than one hardcoded `__main__`
block with an interactive-history `readline` setup meant for a single workstation. Beyond
that:

- **Dropped dead code**: the centroid-based `relabel_inlet_outlets`/
  `classify_mesh_surface_entities_by_npz` (the original's `__main__` only ever called the
  cpcd-endpoint-based `relabel_inlet_outlets_v2`, which is more robust — see its
  docstring), `_match_centroid` (referenced nothing), and the `cfdmesher_single`/
  `cfdmesher_multiple`/`write_msh_multi` variants the driving script never called.
  `mesh_utils.planarize_openings`, which the original imported for a disabled cleanup
  step, already exists in this package as `patching.planarize_openings`.
- **`os.system` failures now raise.** The original's shelled-out `vmtkmeshgenerator` call
  never checked the exit code, so a vmtk crash silently produced a missing/empty `.vtu`
  and failed later with a confusing error on the next read. `cfdmesher_custom` now raises
  `RuntimeError` immediately on a non-zero exit code.
- **One `--overwrite` flag** instead of three independent per-step force-flags
  (`force_relabel_boundaries`, `force_write_flowsplit`, `force_scan_inlet_nodes`), matching
  the surface stage's convention; relabel/flow-split still always re-run when a case isn't
  skipped outright, since they're cheap.
- Case selection/exclusion/limit and a `volume_mesh_failures.json` failure log, matching
  the surface stage.

## Layout

```
generate_cfd_surface_meshes.py   surface stage (entry point)
vessel_reconstruct.py            branch propagation and fusion
patching.py                      opening planarisation, seam smoothing
generate_cfd_volume_meshes.py    volume stage (entry point)
volume_mesh.py                   opening extension, relabeling, adaptive edge, flow-split
vmtk_backend.py                  thin wrappers around the vmtk CLI and vmtkscripts mesh I/O
config.py                        all paths, overridable by CFDMESH_* env vars
```
