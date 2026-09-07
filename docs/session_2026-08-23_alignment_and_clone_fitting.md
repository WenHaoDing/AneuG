# Session handoff — alignment investigation + GHD-coefficient fitting cloning

**Date:** 2026-08-23 (session spans 2026-08-21 through 2026-08-23)
**Handoff reason:** user switching machines; picking this up in a fresh agent session.

This document summarizes a long, multi-phase working session on the AneuG
alignment + GHD-coefficient-fitting pipeline. It's written for an agent with
no memory of the conversation to pick up context quickly. Read top to bottom
once; after that, use it as a reference/index.

## 1. The core problem that started all of this

Some fitted cases (tracked in `runtime/manifest.csv`, `review_status=redo`)
show a **false dome stub**: a branch fails to bend far enough to reach its
real target location, and instead of the branch stretching, a piece of the
dome bulges outward to (partially, badly) explain the chamfer loss. This
happens because chamfer/occupancy losses alone give no *directional* signal
telling the optimizer "bend this specific branch this specific way" — any
locally-nearby surface can absorb the loss, including the wrong one.

**Underlying hypothesis investigated all session:** the alignment/warm-start
stage matters a lot. If the initial pose (translation/rotation/scale, and
ideally rough branch-bending) is already close to correct, Stage-2 GHD
fitting (`ghd/fitting/ghd_fit.py`) converges to the right answer; if it
starts far off, gradient descent can wander into the false-dome-stub local
optimum.

## 2. Chronological summary of what was tried

### 2a. TPS (thin-plate-spline) registration — tried, mostly abandoned
`ghd/registration/tps_register.py`. Landmark-based warp (branch endpoints +
neck centroid) from canonical to a target. **Problem: TPS's global affine +
unbounded RBF kernel inflates the WHOLE shape (~1.4x volume) to satisfy
stretched branch landmarks**, including the unconstrained dome region.
Landmark trimming (`landmark_start_frac`) and TPS regularization
(`reg` parameter, ridge-style) were both tried and made things comparably
bad or worse (regularization especially: branches merged into a blob at
high `reg`). This whole avenue was superseded by ARAP.

### 2b. ARAP (As-Rigid-As-Possible) registration — the winning warm-start method
`ghd/registration/arap_register.py`. Uses `libigl`'s ARAP solver
(`igl.arap_precomputation`/`igl.arap_solve`). Iterated through several
handle strategies:

- **Point-snap handles** (independent nearest-vertex per centerline point):
  produced local twisting/pinching — a centerline point has no notion of
  which side of the tube's circular cross-section a nearby vertex sits on,
  so adjacent handles snap to front/back-mismatched vertices.
- **Rigid RING handles (WINNER)**: each branch's opening ring
  (`canonical_topology.npy`'s boundary loop) moves as ONE rigid body —
  rotate by the minimal/no-twist rotation aligning canonical→target tangent
  direction, translate centroid to target tip. No dome anchor (tried
  anchoring `dome_vertex_indices` to fix TPS-style inflation — it worked for
  volume [~0.82 ratio] but creased the mesh at the dome/branch boundary;
  dropped in favor of mesh health). **Result on test case: volume ratio
  1.04, watertight, no self-intersection, clean.**
- **`use_ring_normal` refinement**: the ring's rotation was originally
  derived from a coarse 2-point centerline-tangent estimate, which sometimes
  left the ring's cap facing a visibly wrong direction (verified: real clip
  planes aren't always perpendicular to the local centerline). Fixed by
  using each ring's own true outward normal instead (canonical's precomputed
  `cross_vector` vs. the target's own real opening ring, matched via
  `landmarks.npz`'s `opening_ring_idx_k`). **But full ring-normal correction
  (`ring_normal_blend=1.0`) introduced NEW self-intersection** (100
  intersecting face pairs vs. baseline ~22) because it can require a bigger
  reorientation than the tube surface behind the ring can smoothly absorb.
  **`ring_normal_blend=0.5` (SLERP between tangent-based and normal-based
  rotation) was the sweet spot** — direction mostly corrected, no new
  self-intersection.
- **Dense "rail" handles** (`ghd/registration/arap_register_railed.py`):
  tried building front/back point-correspondence rails along the whole
  branch length (using a consistent plane-normal-based front/back
  classification derived from initial branch tangents + SVD + cyclic
  cross-product sign-fix — this math is sound and reusable). **Result: worse
  than ring handles** (52 degenerate faces, volume ratio 0.78, visible
  crumpling) — independent point handles, even side-consistent ones, lack
  the coordination a true rigid body has. **This script is a documented
  negative result, kept as-is, not used.**

Self-intersection detection utility: `ghd/registration/self_intersection.py`
— `find_self_intersections(mesh)` using `trimesh.collision.CollisionManager`
with each face as its own named object (`in_collision_internal`), filtering
out adjacent-face pairs (share a vertex) as false positives. **Requires
`python-fcl`, now installed in the `new` conda env.** Per explicit user
request this is NOT wired into any script's normal output — call it
manually when needed.

### 2c. Two-stage fitting using ARAP as warm start — validated pipeline
`ghd/registration/fit_with_arap_init.py`. Stage A: gradient-based GHD fit
(phi/pose) toward the ARAP-warped mesh via exact per-vertex MSE (branch-
proximity-weighted) + relaxed regularizers. Stage B: continues the same
phi/pose toward the REAL target via ghd_fit.py's standard loss battery.
Key validated hyperparameters:
- Stage B `--lr 3e-3` (NOT the ghd_fit.py default 1e-3, which was too slow,
  and NOT 1e-2, which caused visible mesh collapse once the rigid-loss
  weight decayed — confirmed via direct render inspection, not just
  numbers).
- `--use-ring-normal --ring-normal-blend 0.5` for the ARAP orientation fix.
- Stage A now has a **self-intersection checkpoint/rollback mechanism**:
  every `log_every` iterations (after `stage_a_intersection_check_min_iter`,
  default 500), checks the live mesh for self-intersection and remembers the
  latest clean checkpoint; if the final iteration isn't clean, rolls back to
  that checkpoint instead of trusting whatever the last iteration landed on.
- `fit_with_arap_init.py` also supports `--alignment-mode {chamfer,skeleton}`
  (see 2d) and `--lambda-opening-chamfer` (weak experimental opening-cap
  loss, tested at 0.125 — inconclusive/not pursued further per user
  instruction "with opening never helps").

**Batch validated**: all 27 `review_status=redo` ImperialNHS cases run
through this pipeline successfully (`runtime/arap_init_redo_batch/`), zero
failures, `--lr 3e-3 --use-ring-normal --ring-normal-blend 0.5`.
**⚠️ These results have NOT yet been reviewed/eyeballed by the user for
quality — only confirmed to run without crashing.**

### 2d. Skeleton alignment — an alternative, simpler alignment strategy
`ghd/fitting/skeleton_alignment.py` (deliberately NOT merged into
`ghd/fitting/alignment.py` per explicit instruction). Ported the *idea*
from the old pipeline (`AneuSeg/prep_case_for_ghd.py`, both its bifurcation
Procrustes-to-template branch and its sidewall hand-built-frame branch):
align using ONLY a handful of skeleton landmarks (branch endpoints +
dome/neck centroid), completely ignoring the mesh surface — no gradient-
descent refinement at all. This is literally `alignment.py`'s own
`umeyama_similarity` closed-form init, just used as the FINAL answer instead
of a gradient-descent starting point.

**Findings (small sample, real signal but not exhaustive):**
- On an already-good control case: skeleton alignment did ~2x WORSE
  (chamfer 0.00847 vs. 0.004612) and showed a visible branch-misalignment
  artifact — expected, since standard alignment already works well there.
- On a genuine `redo` case (`i69sYpgXky_aneurysm1`): skeleton alignment +
  **plain default fit config** (no opening-chamfer extras) BEAT both the
  original standard-pipeline result AND the ARAP-init result:
  - Original (standard): 0.002939
  - ARAP-init (ring, blend=0.5): 0.001118
  - **Skeleton align + default fit: 0.00088** (visually clean too)

**This was NOT followed up with a full 27-case batch before the session
paused** — that's a natural next step (see §5).

Wired into `run_case.py` too via `--alignment-mode {chamfer,skeleton}` (same
flag convention as `fit_with_arap_init.py`).

`scripts/fit/manage.py` was updated to add an `aneurysm_type` column to
`runtime/manifest.csv` (populated from `endpoints_manual.npy` during
`scan`), so aneurysm type can be queried from the manifest instead of
re-reading geometry files. **⚠️ `runtime/manifest.csv` was deleted by the
user mid-session (intentionally) and has not been regenerated — running
`manage.py scan` again will rebuild it, now with the new column.**

### 2e. Canonical directory rename (sidewall)
User decided to revert to an older sidewall canonical basis. Renamed on
disk: old current `Sidewall/` → `Sidewall_v2/`, old `Sidewall_v1/` →
`Sidewall/`. Because every script in the codebase hardcodes the directory
name `"Sidewall"` (not a config value), this rename alone makes every
script pick up the reverted canonical with **zero code changes needed** —
confirmed this is why an initial edit to
`models/multi_canonical_ghd_reconstruct.py` (pointing type 1/2 at
`"Sidewall_v1"`) was reverted back to plain `"Sidewall"`.

**⚠️ Caveat, not yet resolved**: `dataset/canonical/Sidewall/` (the
reverted-to directory) is missing `canonical_topology.npy` — only has
`canonical_picks.npy`, `centerline.vtp`, `eigenvectors.npy`,
`landmarks.npz`, `mesh.obj`. User said they need to rerun
`dataset/canonical/record_topology.py` for it. **Any alignment-dependent
code path for sidewall cases (branch_endpoints, openings, etc.) will
currently fail or behave unexpectedly until that's done.** Bifurcated-only
testing was used throughout the rest of the session specifically to avoid
this landmine.

### 2f. "Fitting cloning" — reusing an old project iteration's fitted results
New idea, separate workstream from the alignment investigation above. There
is an OLD, already-fitted dataset at
`/media/yaplab2/HDD Storage/almaha/Aneu_GHD/Fitting_Results_Final/{ImperialNHS,AnueX}/<case_name>/`
(old GHD-VAE project iteration). Its `phi` can't be reused directly — wrong
eigenbasis/scale (different canonical mesh resolution). But its
`ghd_fitted.obj` is real, high-fidelity 3D geometry (old `chamfer_best` was
often ~1e-5), so it can be used as a **warm-start target** for re-deriving
phi in the CURRENT canonical basis.

**New script**: `ghd/fitting/clone_fit.py` — two-stage:
- **Stage A**: warm-start phi/pose toward the OLD fitted mesh, via the
  standard loss battery PLUS a node-to-node nearest-neighbour MSE
  (`pytorch3d.ops.knn_points`, recomputed every iteration — old and current
  canonicals have different topology/vertex counts, confirmed, so there's no
  shared index to exploit). **Rigid weight kept HIGH and CONSTANT
  (`lambda_rigid_stage_a`, default 2.0, tested 1.0/1.5/2.0) — per explicit
  instruction: this stage should nudge phi into a good neighbourhood, NOT
  aggressively warp the mesh.** `lr_stage_a` had to be lowered from the
  `fit_with_arap_init.py`-derived default 1e-2 down to **1e-3** — 1e-2
  caused the already-close starting point to overshoot badly (chamfer went
  0.000117 → 0.14 by iter 200 despite high rigid weight).
- **Stage B**: standard Stage-1 alignment (`fit_alignment`, exactly like
  `run_case.py`) against the case's REAL current geometry, then continues
  the SAME phi/w/log_s/t toward the real target using `ghd_fit.py`'s
  default loss weights/schedule (duplicated loop, not a call to `ghd_fit()`
  itself, matching this session's established "don't touch shared
  production files" pattern).

**🐛 MAJOR BUG FOUND AND FIXED**: the old `ghd_fitted.obj` is saved in the
OLD pipeline's own NORMALIZED coordinate space (divided by its own
`s_can`), NOT physical mm. Confirmed empirically: raw old-mesh volume was
0.056 mm³ (absurd for an aneurysm) vs. the real target's 86.7 mm³ — a
~1536x volume ratio. **Fix**: read `old_case_dir/metrics.json`'s `s_can`
field and multiply the old mesh's vertices by it before using it as a
target. After the fix, old-mesh volume (182 mm³) and the real target's
volume (86.7 mm³) are in the same plausible ballpark (real anatomical/
pipeline-convention gap, not a units bug). **This fix is already applied in
the current `clone_fit.py`** — anyone re-running old code without it will
get badly broken results (Stage B starting from a mesh ~11x too small
linearly, chamfer≈0.38-0.40, volume/occupancy loss saturated near their
ceiling).

**Lambda-rigid-stage-a sweep** (10 bifurcated cases, POST-FIX):

| case | λ=1.0 | λ=1.5 |
|---|---|---|
| 0ZafRQZ51s_aneurysm1 | 0.003891 | 0.000815 |
| 0ZafRQZ51s_aneurysm2 | 0.015942 | 0.010796 |
| 184D0jB3dS_aneurysm1 | 0.000237 | 0.000239 |
| 2uw5ITsPNj_aneurysm1 | 0.000391 | 0.000430 |
| 3AZ6NRxph7_aneurysm1 | 0.000215 | 0.000205 |
| 4lNuZOEFjC_aneurysm1 | 0.000933 | 0.000776 |
| 4N00Dar9XN_aneurysm1 | 0.000229 | 0.000240 |
| 5M9lbzxsQn_aneurysm1 | 0.000219 | 0.000219 |
| 5ywqV955hJ_aneurysm1 | 0.001227 | 0.001286 |
| 6GpXbvnIHl_aneurysm1 | 0.000331 | 0.000301 |

λ=2.0 result exists for `184D0jB3dS_aneurysm1` only (0.00030, from before
the sweep script existed) — roughly comparable to λ=1.0/1.5 on that case.
**Overall: λ=1.0 vs 1.5 are close on most cases; λ=1.5 notably better on
the two `0ZafRQZ51s_*` cases (both worse-than-average outliers at either
setting).** No batch run has been done for λ=2.0 across all 10 for a fully
fair 3-way comparison. **These results have NOT yet been eyeballed
visually** — only chamfer numbers compared so far; the earlier visual check
(single case, λ=2.0, before the sweep) looked clean.

Output layout: `runtime/fitting/cloned_lambda_sweep/lambda_{1.0,1.5}/ImperialNHS/<case_name>/`
(and the original single test at `runtime/fitting/cloned/ImperialNHS/184D0jB3dS_aneurysm1/`).
Each case dir has `clone_target.obj` (rescaled old mesh), `final_aligned.obj`
(real Stage-1 target), `ghd_fitted.obj` (final result), `sanity/`,
`sanity_stage_a/`, `rigid_checkpoints/`, `metrics.json`, loss curves/history.

## 3. User's parting note (session-ending message)

> "I think the alignment problem still persists."

**This was not further diagnosed before the handoff** — it's ambiguous
which of several things it refers to:
1. The ORIGINAL false-dome-stub problem (§1) — possible this is judged not
   fully solved even by the ARAP-init pipeline, pending the user's visual
   review of the 27-case redo batch (§2c) which hasn't happened yet.
2. Something noticed in the clone-fitting lambda sweep results (§2f) — e.g.
   the two `0ZafRQZ51s_*` outlier cases, or a visual issue not yet reported
   in this chat.
3. Something about skeleton alignment (§2d) not yet run at scale.

**First thing to do in the new session: ask the user directly what
"the alignment problem" refers to now** — don't assume; the investigation
branched in several directions (ARAP warm-start vs. skeleton alignment vs.
clone-fitting) and "alignment" could mean any of them at this point.

## 4. Key files (new/modified this session)

| File | Status | Purpose |
|---|---|---|
| `ghd/registration/tps_register.py` | existing, extended | TPS registration prototype (mostly superseded) |
| `ghd/registration/arap_register.py` | new | ARAP ring-handle registration — **the winning warm-start method** |
| `ghd/registration/arap_register_railed.py` | new | Dense rail-handle ARAP variant — **negative result, kept for reference, not used** |
| `ghd/registration/self_intersection.py` | new | Standalone self-intersection checker (`find_self_intersections`, `report_self_intersections`) — NOT auto-wired into any pipeline script |
| `ghd/registration/fit_with_tps_init.py` | new | Two-stage fit using TPS warm start (superseded by ARAP version) |
| `ghd/registration/fit_with_arap_init.py` | new | **Two-stage fit using ARAP warm start — validated pipeline**, `--alignment-mode`, self-intersection rollback |
| `ghd/fitting/skeleton_alignment.py` | new | Closed-form-only alignment (no gradient refinement) — alternative to `fit_alignment` |
| `ghd/fitting/clone_fit.py` | new | Two-stage "fitting cloning" from old project results — s_can bug fixed |
| `ghd/fitting/run_case.py` | modified | Added `--alignment-mode {chamfer,skeleton}` |
| `scripts/fit/manage.py` | modified | Added `aneurysm_type` manifest column |
| `models/multi_canonical_ghd_reconstruct.py` | unchanged (reverted) | Sidewall canonical selection — now relies on the on-disk rename, not code |
| `dataset/canonical/Sidewall_v2/`, `Sidewall/` | renamed on disk by user | Sidewall canonical swap — **Sidewall/ needs `record_topology.py` rerun** |

## 5. Suggested next steps

1. **Ask the user what "the alignment problem" refers to** before doing
   anything else — see §3.
2. If it's the ARAP-init batch: get the user to eyeball
   `runtime/arap_init_redo_batch/ImperialNHS/<case>/sanity/sanity_final.png`
   for a sample of the 27 redo cases — this was never done.
3. If it's skeleton alignment: run the validated `skeleton align + default
   fit` combo across all 27 redo cases (only 1 case tested so far) and
   compare chamfer + visuals against both the original and ARAP-init
   results.
4. If it's clone-fitting: eyeball the lambda-sweep sanity renders
   (`runtime/fitting/cloned_lambda_sweep/lambda_*/ImperialNHS/*/sanity/sanity_final.png`),
   especially the `0ZafRQZ51s_*` outlier cases.
5. Once `dataset/canonical/Sidewall/record_topology.py` is rerun, sidewall
   cases become testable again for all of the above.
6. `runtime/manifest.csv` needs a fresh `manage.py scan` to be regenerated
   (it was deleted mid-session).
