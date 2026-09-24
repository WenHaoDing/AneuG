"""Dome-normalized roughness: measure every shape at scales relative to its own sac.

A VARIANT of mesh_regularizer, built on top of it. Nothing in mesh_regularizer.py is
modified: the mm-based model and MeshRegularizer keep working exactly as before, and this
module adds a second way to use them.

THE PROBLEM. Roughness scales are fixed in millimetres, so 0.3 mm means something
different on a 3 mm sac than on a 14 mm one: the same bump is gross on the first and
slight on the second. The reference sacs span 3.4-11.7 mm (p05-p95 over the 315 AneuX
cases, median 6.15 mm), so "0.3 mm" is not a fixed fraction of anything anatomical.

THE IDEA. Scale each shape by

    alpha = REFERENCE_SAC_MM / sac_span(shape)

so its sac matches the reference median, measure and smooth there, then scale back. A
0.3 mm scale is then the same FRACTION of the sac on every shape (0.049 of the sac span),
and one tolerance means the same thing on a small aneurysm and a large one. Shapes are
compared by relative smoothness, not absolute bump size.

THE ANCHOR is the sac's longest span (largest PCA extent), not a volume-equivalent
diameter: AneuX dome_sac.ply is an OPEN surface (0 of 315 are watertight) so its volume is
meaningless, while a PCA span is well defined on an open mesh and on a voxel mask alike.
That is what makes the reference side (dome_sac.ply) and the ImperialNHS side
(dome_size.npy, written by get_dome_ImperialNHS.py from the label-2 mask) comparable.

TWO CLAMPS, in real millimetres, because both ends of the scale range break otherwise:

  * fine end -- on a big sac alpha is small, so a normalized 0.3 mm scale can fall below
    what the source mesh resolves. A patch narrower than ~2.3 edge lengths holds 7 points
    for a 6-coefficient quadric and its residual is meaningless (a smooth analytic tube
    measures 4e-11 there, against 1.6e-05 at 0.3 mm).
  * coarse end -- on a small sac a normalized 1.2 mm scale can exceed the vessel calibre,
    where a patch wraps around the tube and stops being a single-valued height field.

So a case measures only those model scales whose REAL radius lands inside
[min_real_scale_mm, max_real_scale_mm]; the rest are dropped and the report says which
were used. A case can therefore carry four targets rather than five, which is honest --
it says the data cannot support the fifth -- but z-scores are then not drawn from an
identical scale set across cases, and anything aggregating them must read `target_scales`.

BUILDING THE REFERENCE. The band has to be measured in the same normalized space:

    python -m mesh_regularizer.dome_normalized scan --n_jobs 8

normalizes and remeshes each AneuX reference case in memory and writes only the model,
reference_models/reference_roughness__domenorm6.15__t....npz -- the normalized meshes are
never saved. Only against that model are the z-scores meaningful; scoring a normalized
shape against the mm-based model is an approximation that holds only near the median sac.

    python -m mesh_regularizer.dome_normalized assess <surface> --sac_mm 13.8
    python -m mesh_regularizer.dome_normalized run <surface> --sac_mm 13.8 --out out.obj
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json

import numpy as np

from .config import MODEL_DIR
from .mesh_regularizer import (DEFAULT_GUARD_SCALE, DEFAULT_TARGET_SCALES, MeshRegularizer,
                               QUANTILES, ReferenceModel, config_stem, load_mesh, log_stats,
                               multiscale_roughness, remesh, vertex_areas, weighted_quantile)

# Median sac span over the 315 AneuX reference cases (dome_sac.ply, largest PCA extent).
# Every shape is scaled so its sac matches this, which keeps normalized lengths close to
# familiar millimetre values: at the median sac the variant reduces to the mm-based model.
REFERENCE_SAC_MM = 6.15

REFERENCE_ROOT = "/media/yaplab2/wd8tb/wenhao/angioflow/data/geometry/AneuX"
REFERENCE_MESH = "merged_reconstruction_remeshed.obj"
REFERENCE_SAC = "dome_sac.ply"
DOME_SIZE_NPY = "dome_size.npy"          # written by get_dome_ImperialNHS.py

# The edge the normalized copies are remeshed to, in dome units. 0.132 mm is what the
# mm-based reference set happens to sit at, so the two models stay comparable.
NORMALIZED_EDGE = 0.132

# Defaults for the clamps above: 2.3 * NORMALIZED_EDGE at the fine end (the quadric floor),
# and vessel calibre at the coarse end.
MIN_REAL_SCALE_MM = 0.30
MAX_REAL_SCALE_MM = 1.60

# The grid the reference is SCANNED over, in dome units: geometric, ratio ~1.23, from the
# quadric floor (2.3 * NORMALIZED_EDGE) up to where a patch stops being a height field on
# the parent vessel. Denser than the millimetre model's five scales on purpose: which
# scales are driven is chosen per case at inference, and under normalization each case
# uses a different real-millimetre window, so a sparse grid can leave a large sac with
# only two usable targets. Scanning is the expensive step; re-picking targets is free.
SCAN_SCALES = (0.30, 0.38, 0.47, 0.58, 0.72, 0.90, 1.10, 1.35, 1.65)
SCAN_GUARD_SCALE = 2.00

# What a run drives by default: the grid's closest equivalents to the millimetre model's
# 0.3/0.5/0.8/1.0/1.2, so results stay comparable with the millimetre pipeline.
DEFAULT_INFERENCE_SCALES = (0.30, 0.47, 0.72, 0.90, 1.10)

# Where smoothing aims: a quarter SD above the dome-normalized reference mean. Picked by
# eye over a ladder run on a big-sac case (1KPkS2Uocy, 13.8 mm sac): +1.0 converges in 12
# rounds and 0.39 mm of movement, +0.5 in 22 rounds and 0.52 mm, and 0.0 stalls after 43
# rounds having pushed the finest scale to -1.45 SD, smoother than typical tissue. A lower
# target also decides WHICH cases are touched at all, not just how hard: at +1.0 a
# mid-range case measuring +0.8 SD is left alone entirely. Under dome normalization this
# is a far stronger instruction than the same number in millimetres, because a big sac's
# bumps now score against the sac's own size.
DEFAULT_TOLERANCE = 0.25

# Snapping levels. Instead of one target for every shape, a shape is smoothed to the next
# level BELOW where it already measures: a case at -0.22 SD goes to -0.25, one at +0.10
# goes to 0.0, and anything above +0.25 goes to +0.25. Every shape therefore improves by
# at least one notch rather than passing untouched because it was already inside the band.
# The lowest level is -1.0, which is also where the default overshoot floor sits: a shape
# already below it is left alone, since smoothing it further would take it further from
# physiological roughness rather than closer.
SNAP_LEVELS = (-1.0, -0.5, -0.25, 0.0, 0.25)


def snap_target(score, levels=SNAP_LEVELS):
    """The level a shape measuring `score` should be smoothed to, or None to leave it.

    The highest level strictly below `score`. None when the shape already sits at or
    below the lowest level.
    """
    below = [float(l) for l in levels if float(l) < float(score)]
    return max(below) if below else None

# How hard a run is allowed to push before a backstop stops it. A preset is a named set of
# stopping conditions, not a target: `tolerance` still decides where smoothing aims.
#
#   default   the original's backstops. A rough case tends to stop on one of them rather
#             than on the target -- 57 of 145 ImperialNHS cases did.
#   relaxed   for shapes rough enough that the backstops fire before the target is met:
#             the overshoot floor is dropped far below anything expected, the round cap
#             raised, the over-smoothing guard switched off, and the movement budget
#             widened. A case then stops because it reached the target, or because
#             smoothing genuinely stopped making progress.
#
# max_deviation_real_mm is in REAL millimetres and converted per case (the mesh is
# measured in dome units, where the same tissue distance is alpha times larger).
STOPPING_PRESETS = {
    "default": dict(overshoot_limit=-1.0, max_rounds=30, guard_tolerance=1.0,
                    max_deviation_real_mm=0.4),
    "relaxed": dict(overshoot_limit=-3.0, max_rounds=50, guard_tolerance=1e6,
                    max_deviation_real_mm=1.0),
}


# ======================================================================================
# the anchor: how big is this aneurysm
# ======================================================================================
def pca_span(points):
    """Longest extent along the point set's own principal axes, in input units."""
    P = np.asarray(points, dtype=float)
    C = P - P.mean(0)
    _, _, vt = np.linalg.svd(C, full_matrices=False)
    proj = C @ vt.T
    return float(np.max(proj.max(0) - proj.min(0)))


def sac_span_from_mesh(path):
    """Sac span from a dome mesh, open or closed (AneuX dome_sac.ply)."""
    import trimesh
    return pca_span(trimesh.load(path, process=False).vertices)


def sac_span_from_npy(path):
    """Sac span from dome_size.npy, as written by get_dome_ImperialNHS.py."""
    d = np.load(path, allow_pickle=True).item()
    return float(d["max_extent_mm"])


def resolve_sac_span(case_dir):
    """Whichever of the two the case carries. Raises if it carries neither."""
    npy = os.path.join(case_dir, DOME_SIZE_NPY)
    if os.path.exists(npy):
        return sac_span_from_npy(npy)
    ply = os.path.join(case_dir, REFERENCE_SAC)
    if os.path.exists(ply):
        return sac_span_from_mesh(ply)
    raise FileNotFoundError("no %s or %s in %s" % (DOME_SIZE_NPY, REFERENCE_SAC, case_dir))


def scale_factor(sac_mm, reference_sac_mm=REFERENCE_SAC_MM):
    if not np.isfinite(sac_mm) or sac_mm <= 0:
        raise ValueError("sac span must be positive and finite, got %r" % (sac_mm,))
    return float(reference_sac_mm) / float(sac_mm)


def scaled_copy(mesh, alpha):
    """A copy of `mesh` scaled about the origin. The input is never modified."""
    import trimesh
    return trimesh.Trimesh(np.asarray(mesh.vertices, dtype=float) * alpha,
                           np.asarray(mesh.faces), process=False)


def subset_model(model, scales):
    """A copy of `model` carrying only `scales`.

    Every smoothing round measures roughness at EVERY scale the model holds
    (MeshRegularizer._curves iterates model.scales), so a densely scanned model makes each
    round proportionally more expensive -- and the cost is dominated by the coarsest
    scale, whose patches hold thousands of points. Measured on a 10.5k-vertex case: one
    pass over five scales took 80 s, of which the 2.0 guard was the bulk. Restricting the
    model to the scales a run actually drives is what keeps the dense scan free at
    inference: scan once over many scales, measure only the few that matter.
    """
    keep = tuple(sorted(float(s) for s in scales if float(s) in set(model.scales)))
    missing = [s for s in scales if float(s) not in set(model.scales)]
    if missing:
        raise ValueError("scales %s are not in the model" % missing)
    meta = dict(model.meta)
    meta["target_scales"] = [s for s in meta.get("target_scales", keep) if s in keep]
    g = meta.get("guard_scale")
    meta["guard_scale"] = g if (g is not None and float(g) in keep) else None
    meta["subset_of"] = [float(x) for x in model.scales]
    return ReferenceModel(keep, model.quantiles,
                          {s: model.mean[s] for s in keep}, {s: model.sd[s] for s in keep},
                          model.target_edge, model.n_cases, model.cases,
                          {s: model.per_case[s] for s in keep if s in model.per_case}, meta)


# ======================================================================================
# the regularizer variant
# ======================================================================================
class DomeNormalizedRegularizer:
    """MeshRegularizer applied in dome units, with the scale clamps applied per case.

    Every method takes the shape's sac span in millimetres, and reports in BOTH spaces:
    scores and scales in dome units, displacements converted back to real millimetres.
    """

    def __init__(self, model=None, reference_sac_mm=REFERENCE_SAC_MM,
                 min_real_scale_mm=MIN_REAL_SCALE_MM, max_real_scale_mm=MAX_REAL_SCALE_MM,
                 target_scales=None, guard_scale=SCAN_GUARD_SCALE,
                 tolerance=DEFAULT_TOLERANCE, stopping="default",
                 export_edge_real_mm=None, **regularizer_kwargs):
        self.reference_sac_mm = float(reference_sac_mm)
        self.min_real_scale_mm = min_real_scale_mm
        self.max_real_scale_mm = max_real_scale_mm
        # None means "every scanned scale the clamps allow, per case": under
        # normalization each shape's real-millimetre window sits somewhere different on
        # the grid, so fixing one list of targets for all cases wastes the dense scan and
        # can leave a large sac with two usable scales.
        self.target_scales = (None if target_scales is None
                              else tuple(float(s) for s in target_scales))
        self.guard_scale = None if guard_scale is None else float(guard_scale)
        # The model file is named for the scales it was SCANNED over, which is a superset
        # of the scales any one run drives; deriving the path from the targets would look
        # for a file that was never written.
        self.model = model if model is not None else ReferenceModel.load(
            default_model_path(SCAN_SCALES, SCAN_GUARD_SCALE, self.reference_sac_mm))
        missing = [s for s in (self.target_scales or ()) + (
            (self.guard_scale,) if self.guard_scale else ())
            if s not in set(self.model.scales)]
        if missing:
            raise ValueError(
                "scales %s are not in the model, which carries %s. Pick targets from the "
                "scanned grid, or rescan with the scales you want."
                % (missing, list(self.model.scales)))
        self.tolerance = float(tolerance)
        # Remeshing happens in dome units, so a fixed normalized edge is a DIFFERENT real
        # edge per case: at alpha 0.445 a 0.132 normalized edge is 0.30 mm of tissue, and
        # a big sac came back at 9.9k vertices instead of 122k. Exporting at a real edge
        # keeps the tessellation the same for every case, which is what anything
        # downstream (volume meshing above all) expects.
        self.export_edge_real_mm = export_edge_real_mm
        if isinstance(stopping, str):
            if stopping not in STOPPING_PRESETS:
                raise ValueError("unknown stopping preset %r; have %s"
                                 % (stopping, sorted(STOPPING_PRESETS)))
            self.stopping_name, self.stopping = stopping, dict(STOPPING_PRESETS[stopping])
        else:
            self.stopping_name, self.stopping = "custom", dict(stopping)
        self.regularizer_kwargs = regularizer_kwargs

    @property
    def candidate_scales(self):
        """Scales a run may drive: the explicit targets, else every scanned scale but the
        guard. The clamps then cut this down per case."""
        if self.target_scales is not None:
            return self.target_scales
        return tuple(s for s in self.model.scales if s != self.guard_scale)

    # -- per-case setup ----------------------------------------------------------------
    def usable_scales(self, alpha):
        """Model target scales whose REAL radius survives both clamps, with what was cut."""
        keep, dropped = [], {}
        for s in self.candidate_scales:
            real = s / alpha
            if self.min_real_scale_mm is not None and real < self.min_real_scale_mm:
                dropped[s] = "%.3f mm real: below the %.2f mm resolution floor" % (
                    real, self.min_real_scale_mm)
            elif self.max_real_scale_mm is not None and real > self.max_real_scale_mm:
                dropped[s] = "%.3f mm real: above the %.2f mm calibre ceiling" % (
                    real, self.max_real_scale_mm)
            else:
                keep.append(s)
        if not keep:
            raise ValueError("every target scale was clamped away at alpha=%.3f; the sac "
                             "size or the clamps are wrong" % alpha)
        return tuple(keep), dropped

    def _regularizer(self, scales, alpha=1.0):
        stop = dict(self.stopping)
        real_mm = stop.pop("max_deviation_real_mm", None)
        if real_mm is not None:
            stop["max_deviation"] = real_mm * alpha       # real mm -> dome units
        stop.update(self.regularizer_kwargs)              # explicit kwargs win
        if self.export_edge_real_mm:
            stop["export_edge"] = self.export_edge_real_mm * alpha
        # Measure only what this run uses. The guard scale is the most expensive one to
        # measure, so it is carried only when it can actually stop the run; a preset that
        # disables the guard (a huge guard_tolerance) drops it entirely.
        guard_live = (self.guard_scale is not None
                      and float(stop.get("guard_tolerance", 1.0)) < 1e3)
        wanted = tuple(scales) + ((self.guard_scale,) if guard_live else ())
        return MeshRegularizer(model=subset_model(self.model, wanted),
                               target_scales=scales,
                               guard_scale=self.guard_scale if guard_live else None,
                               tolerance=self.tolerance, **stop)

    def _prepare(self, mesh, sac_mm):
        alpha = scale_factor(sac_mm, self.reference_sac_mm)
        scales, dropped = self.usable_scales(alpha)
        m = load_mesh(mesh) if isinstance(mesh, str) else mesh
        return alpha, scales, dropped, scaled_copy(m, alpha)

    # -- the two operations ------------------------------------------------------------
    def assess(self, mesh, sac_mm, n_seeds=4000):
        """Measure without modifying, in dome units. Adds the real-mm scale mapping."""
        alpha, scales, dropped, work = self._prepare(mesh, sac_mm)
        out = self._regularizer(scales, alpha).assess(work, n_seeds=n_seeds)
        out.update(sac_span_mm=float(sac_mm), alpha=alpha, stopping=self.stopping_name,
                   reference_sac_mm=self.reference_sac_mm,
                   target_scales_used=list(scales),
                   target_scales_dropped=dropped,
                   scales_in_real_mm={s: s / alpha for s in scales})
        return out

    def forward(self, mesh, sac_mm, freeze_boundary=True):
        """Smooth in dome units; the returned mesh is back in real millimetres."""
        alpha, scales, dropped, work = self._prepare(mesh, sac_mm)
        out, report = self._regularizer(scales, alpha).forward(work,
                                                               freeze_boundary=freeze_boundary)
        report.update(sac_span_mm=float(sac_mm), alpha=alpha, stopping=self.stopping_name,
                      reference_sac_mm=self.reference_sac_mm,
                      target_scales_used=list(scales),
                      target_scales_dropped=dropped,
                      scales_in_real_mm={s: s / alpha for s in scales})
        # displacements were measured in dome units; real tissue moved 1/alpha as far
        for key in ("surface_deviation_vs_original_max_mm",
                    "surface_deviation_vs_original_mean_mm",
                    "smoothing_displacement_mm", "total_sigma_mm", "edge_out"):
            if isinstance(report.get(key), (int, float)):
                report[key + "_real"] = report[key] / alpha
        return scaled_copy(out, 1.0 / alpha), report


# ======================================================================================
# building the normalized reference model
# ======================================================================================
def default_model_path(target_scales=SCAN_SCALES, guard_scale=SCAN_GUARD_SCALE,
                       reference_sac_mm=REFERENCE_SAC_MM, out_dir=MODEL_DIR):
    """Where a model scanned over these scales lives. Defaults to the scanned grid, not
    the inference targets: one scan serves many target choices."""
    stem = config_stem(target_scales, guard_scale if guard_scale is not None else 0)
    stem = stem.replace("reference_roughness__",
                        "reference_roughness__domenorm%g__" % reference_sac_mm)
    return os.path.join(out_dir, stem + ".npz")


def _scan_one_normalized(job):
    """Measure one reference case in dome units. Nothing is written; the normalized copy
    lives only in this worker's memory.

    Mirrors mesh_regularizer._scan_one, with two steps in front: scale so the sac matches
    the reference size, then remesh to the common normalized edge. The remesh is not
    cosmetic -- scaling multiplies edge lengths along with everything else, and roughness
    is only comparable across cases measured at the same tessellation.
    """
    case, root, reference_sac_mm, edge, scales, n_seeds, seed = job
    sac = sac_span_from_mesh(os.path.join(root, case, REFERENCE_SAC))
    alpha = scale_factor(sac, reference_sac_mm)
    mesh = remesh(scaled_copy(load_mesh(os.path.join(root, case, REFERENCE_MESH)), alpha),
                  edge, "auto")
    n = len(mesh.vertices)
    idx = (np.arange(n) if n_seeds is None or n_seeds >= n
           else np.random.default_rng(seed).choice(n, size=n_seeds, replace=False))
    vals = multiscale_roughness(mesh, scales, idx)
    w = vertex_areas(mesh)[idx]
    curves = {float(s): weighted_quantile(vals[float(s)], w) for s in scales}
    return curves, float(mesh.edges_unique_length.mean()), sac, alpha


def build_reference_model(root=REFERENCE_ROOT, reference_sac_mm=REFERENCE_SAC_MM,
                          edge=NORMALIZED_EDGE, target_scales=DEFAULT_TARGET_SCALES,
                          guard_scale=DEFAULT_GUARD_SCALE, n_seeds=4000, n_jobs=4,
                          cases=None, max_cases=None, save_path=None, verbose=True):
    """Scan the reference set in dome units and save the model.

    Only the model is written, as for the millimetre models in reference_models/: the
    normalized meshes exist per case inside a worker and are discarded.
    """
    import multiprocessing as mp

    target_scales = tuple(sorted(float(s) for s in target_scales))
    guard_scale = None if guard_scale is None else float(guard_scale)
    scales = tuple(sorted(set(target_scales) | ({guard_scale} if guard_scale else set())))

    cases = cases or sorted(c for c in os.listdir(root)
                            if os.path.exists(os.path.join(root, c, REFERENCE_MESH))
                            and os.path.exists(os.path.join(root, c, REFERENCE_SAC)))
    if max_cases:
        cases = cases[:max_cases]
    jobs = [(c, root, reference_sac_mm, edge, scales, n_seeds, i)
            for i, c in enumerate(cases)]
    if verbose:
        print("scanning %d reference case(s) in dome units (sac -> %.2f mm, edge %.3f)"
              % (len(jobs), reference_sac_mm, edge), flush=True)

    rows = {float(s): [] for s in scales}
    used, edges, sacs, alphas, failures = [], [], [], [], []

    def collect(case, result):
        curves, edge_mean, sac, alpha = result
        for s in scales:
            rows[float(s)].append(curves[float(s)])
        used.append(case)
        edges.append(edge_mean)
        sacs.append(sac)
        alphas.append(alpha)
        if verbose and len(used) % 25 == 0:
            print("  scanned %d / %d ..." % (len(used), len(jobs)), flush=True)

    if n_jobs and n_jobs > 1:
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(var, "1")
        with mp.get_context("spawn").Pool(n_jobs) as pool:
            for job, res in zip(jobs, pool.imap(_scan_one_normalized, jobs, chunksize=1)):
                try:
                    collect(job[0], res)
                except Exception as exc:                        # pragma: no cover
                    failures.append({"case": job[0], "error": repr(exc)})
    else:
        for job in jobs:
            try:
                collect(job[0], _scan_one_normalized(job))
            except Exception as exc:
                failures.append({"case": job[0], "error": repr(exc)})
                if verbose:
                    print("  [skip] %s: %s" % (job[0], exc), flush=True)
    if len(used) < 2:
        raise RuntimeError("need at least 2 reference cases, got %d (%d failure(s))"
                           % (len(used), len(failures)))

    # Every case was remeshed to the same normalized edge, so the millimetre model's
    # edge-consistency filter has nothing left to reject; the spread is recorded instead.
    edges = np.asarray(edges)
    per_case = {float(s): np.asarray(rows[float(s)]) for s in scales}
    mean, sd = {}, {}
    for s in per_case:
        mean[s], sd[s] = log_stats(per_case[s])
    sacs = np.asarray(sacs)
    meta = {
        "stat_space": "log", "normalization": "dome",
        "reference_sac_mm": float(reference_sac_mm), "normalized_edge": float(edge),
        "target_scales": list(target_scales), "guard_scale": guard_scale,
        "root": root, "filename": REFERENCE_MESH, "sac_file": REFERENCE_SAC,
        "n_seeds": n_seeds, "scales": [float(s) for s in scales],
        "edge_median": float(np.median(edges)),
        "edge_p05": float(np.percentile(edges, 5)),
        "edge_p95": float(np.percentile(edges, 95)),
        "sac_span_median_mm": float(np.median(sacs)),
        "sac_span_p05_mm": float(np.percentile(sacs, 5)),
        "sac_span_p95_mm": float(np.percentile(sacs, 95)),
        "alpha_min": float(np.min(alphas)), "alpha_max": float(np.max(alphas)),
        "per_case_sac_mm": {c: float(v) for c, v in zip(used, sacs)},
        "failures": failures,
        "scanned_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(),
    }
    model = ReferenceModel(scales, QUANTILES, mean, sd, float(np.median(edges)), len(used),
                           used, per_case, meta)
    path = save_path or default_model_path(target_scales, guard_scale, reference_sac_mm)
    model.save(path)
    if verbose:
        print("scanned %d case(s), %d failure(s) | sac span median %.2f mm (p05 %.2f, p95 %.2f)"
              % (len(used), len(failures), np.median(sacs), *np.percentile(sacs, [5, 95])),
              flush=True)
        print("dome-normalized model saved: %s" % path, flush=True)
    return model, path


# ======================================================================================
# CLI
# ======================================================================================
def _sac_from_args(args):
    if args.sac_mm:
        return float(args.sac_mm)
    return resolve_sac_span(args.case_dir or os.path.dirname(os.path.abspath(args.surface)))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference_sac_mm", type=float, default=REFERENCE_SAC_MM)
    p.add_argument("--target_scales", type=float, nargs="+", default=None,
                   help="scales to drive, in dome units; default is every scanned scale "
                        "the per-case clamps allow")
    p.add_argument("--guard_scale", type=float, default=SCAN_GUARD_SCALE)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("scan", help="build the dome-normalized reference model")
    s.add_argument("--reference_root", default=REFERENCE_ROOT)
    s.add_argument("--edge", type=float, default=NORMALIZED_EDGE)
    s.add_argument("--save_path", default=None)
    s.add_argument("--n_seeds", type=int, default=4000)
    s.add_argument("--scan_scales", type=float, nargs="+", default=list(SCAN_SCALES),
                   help="scales measured into the model, in dome units")
    s.add_argument("--scan_guard_scale", type=float, default=SCAN_GUARD_SCALE)
    s.add_argument("--n_jobs", type=int, default=4)
    s.add_argument("--max_cases", type=int, default=None)

    for name, helptext in (("assess", "measure a surface without changing it"),
                           ("run", "smooth a surface and write the result")):
        q = sub.add_parser(name, help=helptext)
        q.add_argument("surface")
        q.add_argument("--sac_mm", type=float, default=None,
                       help="sac span; by default read from the surface's own folder")
        q.add_argument("--case_dir", default=None, help="where to look for the sac size")
        q.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
        q.add_argument("--export_edge_real_mm", type=float, default=None,
                       help="remesh the exported surface to this REAL edge length")
        q.add_argument("--stopping", default="default", choices=sorted(STOPPING_PRESETS),
                       help="how hard to push before a backstop stops the run")
        q.add_argument("--min_real_scale_mm", type=float, default=MIN_REAL_SCALE_MM)
        q.add_argument("--max_real_scale_mm", type=float, default=MAX_REAL_SCALE_MM)
        q.add_argument("--model", default=None)
        if name == "run":
            q.add_argument("--out", required=True)
            q.add_argument("--report", default=None)

    args = p.parse_args()

    if args.command == "scan":
        build_reference_model(root=args.reference_root,
                              reference_sac_mm=args.reference_sac_mm, edge=args.edge,
                              target_scales=tuple(args.scan_scales),
                              guard_scale=args.scan_guard_scale, n_seeds=args.n_seeds,
                              n_jobs=args.n_jobs, max_cases=args.max_cases,
                              save_path=args.save_path)
        return

    model = ReferenceModel.load(args.model) if args.model else None
    reg = DomeNormalizedRegularizer(
        model=model, reference_sac_mm=args.reference_sac_mm,
        min_real_scale_mm=args.min_real_scale_mm, max_real_scale_mm=args.max_real_scale_mm,
        target_scales=tuple(args.target_scales) if args.target_scales else None,
        guard_scale=args.guard_scale,
        tolerance=args.tolerance, stopping=args.stopping,
        export_edge_real_mm=args.export_edge_real_mm, export_edge=None,
        time_budget_s=None, n_seeds=4000, verbose=False)
    sac = _sac_from_args(args)

    if args.command == "assess":
        out = reg.assess(args.surface, sac)
        print("sac %.2f mm, alpha %.3f | scales used %s (real mm: %s)"
              % (sac, out["alpha"], out["target_scales_used"],
                 {k: round(v, 2) for k, v in out["scales_in_real_mm"].items()}))
        if out["target_scales_dropped"]:
            print("dropped: %s" % out["target_scales_dropped"])
        print("%s | worst target %+.2f SD | per scale %s"
              % (out["verdict"], out["worst_target_excess_sd"],
                 {k: round(v, 2) for k, v in out["signed_excess_sd"].items()}))
        return

    import trimesh
    mesh, report = reg.forward(args.surface, sac)
    trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.faces),
                    process=False).export(args.out)
    report.pop("history", None)
    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2, default=float)
    print("sac %.2f mm, alpha %.3f | %s after %d rounds | moved max %.3f mm of real tissue"
          % (sac, report["alpha"], report["status"], report["rounds"],
             report["surface_deviation_vs_original_max_mm_real"]))
    print("after %s" % {k: round(v, 2) for k, v in report["signed_excess_after"].items()})
    print("written: %s" % args.out)


if __name__ == "__main__":
    main()
