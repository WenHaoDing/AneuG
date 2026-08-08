# Mesh Regularizer

Normalizes the surface roughness of MRI-derived vessel meshes to a physiological
reference, so that shapes reconstructed from different acquisitions become comparable
without flattening the anatomy that makes them worth studying.

```python
from mesh_regularizer import MeshRegularizer

reg = MeshRegularizer()                      # loads the bundled reference model
mesh, report = reg.forward("case.obj")       # regularized mesh + full audit trail
```

```bash
python -m mesh_regularizer.mesh_regularizer scan --n_jobs 16      # build a reference model
python -m mesh_regularizer.mesh_regularizer run                   # regularize a folder
python -m mesh_regularizer.ablation run                           # sweep configurations
python -m mesh_regularizer.measure_downstream_edge_length         # downstream resolution
```

---

## The problem

Meshes from MR angiography carry voxel-scale noise that is not anatomy. At their native
resolution (~0.4–0.5 mm edges) they are also too coarse to represent sub-millimetre
structure at all, so smoothing alone cannot fix them — they must first be resampled.

Measured across 301 cases, **every one sits 2.4 to 9.2 SD above the reference** in
fine-scale roughness. The gap is systematic, a property of the acquisition pipeline
rather than a handful of bad reconstructions.

## How roughness is measured

At vertex *v* and scale *r*, take the patch of nearby vertices, build a local frame by
PCA, and least-squares fit a quadric `z = ax² + bxy + cy² + dx + ey + f`. Roughness is the
RMS residual divided by *r*, which makes it dimensionless.

**Fitting a quadric is what makes this work.** A cylinder, sphere or saddle is quadratic
to second order, so ordinary vessel anatomy of any calibre is absorbed by the fit and
leaves almost no residual — only genuine bumps survive. An analytic cylinder of R=1.36 mm
measures **1e-5** here against **~3e-3** for real reference tissue, a margin of ~300×.

This suppresses calibre dependence strongly rather than eliminating it: a tube written as
a height field has higher-order terms a quadric cannot represent. Measures that do *not*
discount curvature were tested and rejected — mean curvature, normal dispersion and plain
smoothing residual all report a calibre-dependent pedestal (~r/2R for a cylinder) that
swamps the noise signal and makes vessels of different width incomparable.

### Choice of scales

The floor is set by the fit, not by ring structure: a 1-ring patch holds ~7 points for 6
coefficients, leaving one degree of freedom and a meaningless residual. Two rings give 19
points, which at the 0.132 mm reference edge length is **~0.3 mm**.

The ceiling is set by calibre. At an effective vessel radius of 1.36 mm, patches beyond
~0.8 mm begin wrapping around the tube and stop being single-valued height fields. Patch
folding measured as p99 planarity: 0.68 at 0.8 mm, **0.95 at 1.0 mm, 0.98 at 1.2 mm**.

Scales play two roles. **Target scales** are driven toward the reference — sub-0.5 mm
content is reconstruction signature in both datasets, so normalizing it costs no biology.
The **guard scale** is a floor, never a target: 0.8 mm structure includes blebs and
daughter sacs, so the criterion only checks the shape does not fall *below* the reference
band there.

> The guard is evidence against over-smoothing, **not proof that focal pathology
> survives**. A smooth bleb is well described by a local quadric and contributes little
> residual either way. It is evaluated on the upper quantiles only (q ≥ 0.70), where
> small-area features live, because averaging over the whole grid dilutes a focal drop to
> nothing. Visual inspection of before/after geometry remains the definitive check.

## How the reference is built

Each reference mesh yields an area-weighted quantile curve of per-vertex roughness. The
model is the **average of those per-mesh curves** (the 1-D Wasserstein barycenter), *not*
the pooled per-vertex values — pooling mixes within-shape and between-shape variation and
describes a distribution no real shape has.

Statistics are kept in **log space**, because roughness is positive and right-skewed. In
linear space the across-case SD at the guard scale is comparable to the mean, so the band
dips below zero and a z of −1 corresponds to roughness near zero, making the guard almost
impossible to trip. In log space one SD is a consistent multiplicative factor at every
scale (1.29 at 0.3 mm rising to 2.21 at 2.0 mm).

The across-mesh spread supplies the tolerance, so stopping is a **depth statement** — "as
typical as reference shapes are to each other" — rather than a p-value, which with
thousands of sampled vertices would be driven by the sample size rather than the effect.

## Control

Smoothing runs in rounds; after each, roughness is re-measured and compared to the band.

Control steers on the **signed** deviation. Smoothing only ever reduces roughness, so a
controller watching |z| would keep smoothing a shape that is already too smooth and drive
it further from the target. Candidates overshooting below the band are rejected and the
step halved instead.

`tolerance` is the aggressiveness knob: 1.0 stops within one SD of typical, 0.0 targets
the reference mean, negative values deliberately go smoother. **Which scales are targets
is not an aggressiveness knob** — the finest scale starts furthest from the reference and
binds the loop, so adding coarser targets already inside tolerance changes nothing. Three
scale configurations were verified to produce byte-identical meshes at tolerance 1.0.

## Anti-shrinkage

Curvature flow moves a tube of radius R inward at ~t/R, so drift scales as **1/R** and
thin branches shrink hardest. Uncorrected, a sub-0.5 mm branch loses **37% of its radius**.

A single global offset cannot fix a spatially varying field — forcing the mean drift to
zero robs thin vessels to inflate thick ones (−30% on thin, +0.3% on thick). Instead the
normal displacement is split by spatial frequency: bulk shrinkage is low-frequency and
varies with calibre over millimetres, while the denoising to keep is high-frequency at the
target scales. Diffusing the displacement to scale σ isolates the bulk component;
subtracting only that cancels shrinkage while leaving the smoothing intact, and because it
is per-vertex, thin vessels get more pushback automatically.

The correction is **iterated** (default 3 passes) because diffusion is not idempotent:
removing `Heat(dn)` still leaves `Heat(dn) − Heat²(dn)`. Iterating sharpens the filter's
rolloff without moving its cutoff, which beats simply lowering σ — that would move the
cutoff into the denoising band.

| compensation | <0.5 mm | 0.5–0.7 | 1.6+ mm |
|---|---|---|---|
| none | −37.0% | −7.4% | −0.5% |
| global offset | −30.5% | −4.6% | +0.25% |
| local σ=1.5, 1 pass | −4.4% | −2.2% | +0.11% |
| **local σ=1.5, 3 passes** | **−1.6%** | **−0.9%** | +0.06% |

Cost is ~0.1 in z(0.3) — the smoothing is essentially unaffected. `compensate_sigma`
defaults to `"auto"` = `max(1.5, 2 × coarsest target)`, keeping the correction clear of
the target scales it would otherwise undo.

## Export resolution

Regularization runs at the reference edge length (0.132 mm) because that is what makes
z-scores comparable. Downstream models were trained on a different tessellation, so the
exported mesh gets one final remesh to **0.125 mm** (`export_edge`, measured from 100
downstream training cases; set to 0 to disable). Applied after metrics are taken.

Note the downstream data is in **metres**, not millimetres, and its resolution is not
tight — a spike at 0.107–0.114 mm with a tail to 0.207.

## Known limitations

- **pymeshlab overshoots** the requested edge length by a stable +4.2% at every value,
  unaffected by iteration count. `remesh(..., calibrate=True)` divides this out for the
  export; the internal remesh deliberately does not calibrate, so existing reference
  models stay comparable. Targets are therefore ~4% coarser than the nominal 0.132 mm.
- **Patch construction is approximate.** Candidates grow over the mesh graph by a bounded
  hop count (which prevents jumping the lumen to the far wall), then are cut by *Euclidean*
  radius — so patches approximate rather than exactly reproduce geodesic balls.
- **The guard weakens as its scale rises**, on two independent axes: the reference band
  widens (1 SD factor 1.40 at 0.8 mm → 2.21 at 2.0 mm) and patch folding grows. Treat
  guard scales above ~1.0 mm as exploratory.
- Already-smooth inputs sink **further below the guard band** than noisy ones (0.66 vs
  0.04 at tolerance −2.0): clean shapes have no fine noise to absorb the smoothing, so it
  reaches real geometry sooner.
- **Not yet validated against ground truth.** The outstanding test is corrupting a
  reference mesh with known synthetic noise and checking it lands back on the original.

## Layout

```
mesh_regularizer.py                 core module — measurement, smoothing, control
ablation.py                         sweep harness: rank / run / rebuild / summarize
measure_downstream_edge_length.py   reads export resolution off downstream training data
config.py                           all paths, overridable by MESHREG_* env vars
reference_models/                   5 cached models (315 reference cases each)
```

Cached models are committed because rebuilding one costs ~15 min and needs the reference
dataset present. Filenames encode their scale roles, e.g.
`reference_roughness__t0.3-0.5__g0.8.npz` = targets 0.3/0.5 mm, guard 0.8 mm.

Depends only on numpy, trimesh, pyvista, scipy and matplotlib — no intra-repo imports.
`pymeshlab` (or `vmtk`) is needed for remeshing; `torch` only for the downstream
measurement script.
