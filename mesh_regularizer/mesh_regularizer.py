"""
Normalize the surface roughness of MRI-derived vessel meshes to a physiological reference.

Shapes reconstructed from MR angiography carry voxel-scale noise that does not reflect
anatomy. This module measures roughness in a way that is largely blind to vessel calibre,
learns what "physiological" looks like from a reference dataset, and smooths an uploaded
shape until it matches -- while refusing to smooth so far that real anatomy is destroyed.

Roughness measure
-----------------
At a vertex v and scale r, take the patch of nearby vertices, build a local frame by PCA,
and least-squares fit a quadric z = ax^2+bxy+cy^2+dx+ey+f. Roughness is the RMS residual
divided by r, which makes it dimensionless.

Fitting a quadric is what makes this work. A cylinder, sphere or saddle is quadratic to
second order, so ordinary vessel anatomy of any calibre is almost entirely absorbed by the
fit and leaves little residual -- only genuine bumps survive. This suppresses calibre
dependence strongly rather than eliminating it: a tube of radius R written as a height
field carries higher-order terms (x^4/8R^3 and beyond) that a quadric cannot represent.
Empirically the leak is small, because least squares absorbs most of the quartic into the
fitted coefficients: an analytic cylinder of R=1.36 mm measures 1e-5 here, against ~3e-3
for real reference tissue, a margin of roughly 300x. Residual calibre sensitivity also
depends on patch radius, frame estimation, sampling density and triangle anisotropy, all
of which are present in the reference measurement too.

Measures that do *not* discount curvature were tested and rejected: mean curvature, normal
dispersion and plain smoothing residual all report a calibre-dependent pedestal (a smooth
cylinder of radius R gives ~r/2R), which swamps the noise signal and makes vessels of
different width incomparable.

Patch construction: candidates are grown over the mesh graph by a bounded hop count, then
cut to radius r by Euclidean distance. The graph stage is what prevents a patch from
jumping across the lumen to the opposite wall; the radius cut is Euclidean, not accumulated
edge-path length, so patches approximate rather than exactly reproduce geodesic balls. Hop
budgets are derived from a low percentile of edge length so that regions of unusually short
edges are not truncated, and coverage is checked and reported.

Re-tessellating a reference mesh through the same remesher used on targets shifts its
measured roughness by under 1%, so the statistic reflects surface geometry rather than
meshing signature, and reference meshes are measured as loaded.

Scales
------
The floor is set by the fit, not by ring structure: a 1-ring patch holds ~7 points for 6
coefficients, leaving one degree of freedom and a meaningless residual. Two rings give 19
points, which at the reference edge length of 0.132 mm is ~0.3 mm. The ceiling is set by
calibre -- at an effective vessel radius of 1.36 mm a patch beyond ~0.8 mm wraps around the
tube and stops being a single-valued height field.

The scales play two roles. Target scales are driven toward the reference: sub-0.5 mm
content is reconstruction signature in both datasets, so normalizing it away costs no
biology. The guard scale is a floor, never a target: 0.8 mm structure includes blebs and
daughter sacs, so the criterion only checks that the shape does not fall *below* the
reference band there.

The guard is evidence against over-smoothing, not proof that focal pathology survives. A
smooth bleb is well represented by a local quadric and so contributes little residual, and
small-area features are easily diluted in an aggregate. The guard is therefore evaluated on
the upper quantiles only, where small-area features live, and the definitive check remains
visual inspection of before/after geometry.

Matching
--------
Each mesh yields a distribution of per-vertex roughness at each scale, summarized as an
area-weighted quantile curve. The reference model is the *average of the per-mesh quantile
curves* (the 1-D Wasserstein barycenter), not the pooled per-vertex values -- pooling would
mix within-shape and between-shape variation and describe a distribution no real shape has.
The across-mesh spread of each quantile supplies the tolerance, so the stopping rule is a
depth statement ("as typical as reference shapes are to each other") rather than a p-value,
which would otherwise be driven by how many vertices were sampled.

Control is steered on the *signed* deviation. Smoothing only ever reduces roughness, so a
controller watching |z| would keep smoothing a shape that is already too smooth and drive
it further from the target; candidates that overshoot below the band are rejected and the
step is halved instead.
"""

import os
import json
import time
import argparse
import datetime

import numpy as np
import trimesh
import pyvista as pv
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, splu
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


from .config import (REFERENCE_ROOT, REFERENCE_FILENAME,  # noqa: F401
                     INPUT_ROOT, INPUT_FILENAME,
                     MODEL_DIR as DEFAULT_MODEL_DIR)

# Which scales are driven toward the reference and which one only guards against
# over-smoothing is a modelling choice, so both are arguments. They are recorded inside
# each cached model, and the cache filename carries them, so several configurations can
# coexist and a run always knows which roles its model was built for.
DEFAULT_TARGET_SCALES = (0.3, 0.5)
DEFAULT_GUARD_SCALE = 0.8
GUARD_QUANTILE_MIN = 0.70     # focal features live in the upper tail; see module docstring
QUANTILES = np.round(np.arange(0.05, 0.96, 0.05), 2)
REPORT_QUANTILES = (0.25, 0.50, 0.90)


# Mean wall-mesh edge length of the downstream training data (angioflowv2_merged), measured
# over 100 cases by measure_downstream_edge_length.py. Exported meshes are remeshed to this
# so they match what the downstream models were trained on.
DOWNSTREAM_EDGE_MM = 0.125

# pymeshlab's isotropic remesher overshoots the requested edge length by a stable +4.2%,
# independent of the value asked for (0.100 -> 0.1041, 0.132 -> 0.1375) and unaffected by
# iteration count. Because it is a constant multiplier it divides straight out, so an
# export that must land on a specific resolution asks for edge / this factor.
PYMESHLAB_EDGE_BIAS = 1.042


def config_stem(target_scales, guard_scale, prefix="reference_roughness"):
    """Filename stem encoding the scale roles, e.g. reference_roughness__t0.3-0.5__g0.8."""
    t = "-".join(f"{float(s):g}" for s in sorted(float(x) for x in target_scales))
    return f"{prefix}__t{t}__g{float(guard_scale):g}"


def model_path(target_scales, guard_scale, out_dir=DEFAULT_MODEL_DIR):
    return os.path.join(out_dir, config_stem(target_scales, guard_scale) + ".npz")

# Plausible bounding-box diagonal for an intracranial vessel complex in millimetres.
# Outside this the input is probably in metres or voxels and the whole model is invalid.
PLAUSIBLE_DIAGONAL_MM = (2.0, 500.0)

REMESHERS = ("auto", "pymeshlab", "vmtk", "none")
SMOOTHERS = ("mcf", "taubin", "humphrey")

_DEFAULT_CACHE = model_path(DEFAULT_TARGET_SCALES, DEFAULT_GUARD_SCALE)


# ======================================================================================
# mesh helpers
# ======================================================================================

def load_mesh(path):
    m = trimesh.load(path, process=True)
    return m.dump(concatenate=True) if isinstance(m, trimesh.Scene) else m


def to_pyvista(mesh):
    faces = np.hstack([np.full((len(mesh.faces), 1), 3), mesh.faces]).ravel()
    return pv.PolyData(np.asarray(mesh.vertices), faces)


def boundary_vertices(mesh):
    """Vertices on an open boundary -- inlet/outlet rings and clip planes."""
    groups = trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)
    if len(groups) == 0:
        return np.zeros(0, dtype=np.int64)
    return np.unique(mesh.edges_sorted[groups])


def vertex_areas(mesh):
    """Barycentric area per vertex; the weight used for every quantile in this module."""
    a = np.zeros(len(mesh.vertices))
    np.add.at(a, np.asarray(mesh.faces).ravel(), np.repeat(mesh.area_faces, 3) / 3.0)
    return a


def triangle_quality(mesh):
    """4*sqrt(3)*area / sum(edge^2) per face; 1.0 is equilateral."""
    tri = mesh.vertices[mesh.faces]
    a = np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1)
    b = np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1)
    c = np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1)
    s = (a + b + c) / 2
    area = np.sqrt(np.maximum(s * (s - a) * (s - b) * (s - c), 1e-20))
    return 4 * np.sqrt(3) * area / np.maximum(a**2 + b**2 + c**2, 1e-20)


def inspect_mesh(mesh):
    """Cheap sanity report. Units are not recoverable from a mesh file, so the best we can
    do is flag a bounding box that is implausible for millimetres."""
    diag = float(np.linalg.norm(mesh.bounding_box.extents))
    el = mesh.edges_unique_length
    info = {
        "n_vertices": int(len(mesh.vertices)), "n_faces": int(len(mesh.faces)),
        "bbox_diagonal": diag, "watertight": bool(mesh.is_watertight),
        "n_components": int(len(mesh.split(only_watertight=False))),
        "n_degenerate_faces": int((mesh.area_faces <= 1e-14).sum()),
        "edge_mean": float(el.mean()), "edge_cv": float(el.std() / max(el.mean(), 1e-12)),
        "n_boundary_vertices": int(len(boundary_vertices(mesh))),
    }
    warnings = []
    if not (PLAUSIBLE_DIAGONAL_MM[0] <= diag <= PLAUSIBLE_DIAGONAL_MM[1]):
        warnings.append(f"bounding-box diagonal {diag:.3f} is outside the plausible "
                        f"millimetre range {PLAUSIBLE_DIAGONAL_MM}; check units")
    if info["n_components"] > 1:
        warnings.append(f"{info['n_components']} disconnected components")
    if info["n_degenerate_faces"]:
        warnings.append(f"{info['n_degenerate_faces']} degenerate faces")
    info["warnings"] = warnings
    return info


def cotangent_laplacian(mesh):
    """Cotangent Laplacian L (negative semi-definite) and barycentric mass matrix M."""
    V, F = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    n = len(V)
    i0, i1, i2 = F[:, 0], F[:, 1], F[:, 2]
    v0, v1, v2 = V[i0], V[i1], V[i2]
    e0, e1, e2 = v2 - v1, v0 - v2, v1 - v0

    def cot(u, w):
        cross = np.linalg.norm(np.cross(u, w), axis=1)
        return np.einsum("ij,ij->i", u, w) / np.maximum(cross, 1e-12)

    c0, c1, c2 = cot(-e1, e2), cot(-e2, e0), cot(-e0, e1)
    I = np.concatenate([i1, i2, i2, i0, i0, i1])
    J = np.concatenate([i2, i1, i0, i2, i1, i0])
    W = np.concatenate([c0, c0, c1, c1, c2, c2]) * 0.5
    L = sp.coo_matrix((W, (I, J)), shape=(n, n)).tocsr()
    L = L - sp.diags(np.asarray(L.sum(axis=1)).ravel())

    face_area = 0.5 * np.linalg.norm(np.cross(-e2, e1), axis=1)
    m = np.zeros(n)
    np.add.at(m, i0, face_area / 3)
    np.add.at(m, i1, face_area / 3)
    np.add.at(m, i2, face_area / 3)
    return L, sp.diags(np.maximum(m, 1e-12))


# ======================================================================================
# multi-scale roughness
# ======================================================================================

def ring_sets(mesh, scales, hop_percentile=10.0, max_hops=40):
    """Graph-reachability matrix per scale.

    The hop budget is derived from a low percentile of edge length rather than the mean, so
    that patches are not truncated where edges happen to be short. Growing candidates over
    mesh edges is what stops a patch from reaching the far wall of the lumen.
    """
    el = mesh.edges_unique_length
    step = max(float(np.percentile(el, hop_percentile)), 1e-9)
    n = len(mesh.vertices)
    e = mesh.edges_unique
    rows = np.concatenate([e[:, 0], e[:, 1]])
    cols = np.concatenate([e[:, 1], e[:, 0]])
    A = sp.csr_matrix((np.ones(len(rows), bool), (rows, cols)), shape=(n, n))
    A = (A + sp.eye(n, dtype=bool, format="csr")).astype(bool)
    need = {float(s): min(int(np.ceil(s / step)) + 1, max_hops) for s in scales}
    out, R, k = {}, A.copy(), 1
    for s in sorted(float(x) for x in scales):
        while k < need[s]:
            R = (R @ A).astype(bool)
            k += 1
        out[s] = R.copy()
    return out


def roughness_at_scale(mesh, r, R, seeds, chunk=3000, return_coverage=False):
    """Per-seed quadric-fit residual at scale r, normalized by r (dimensionless).

    Also reports the fraction of seeds whose candidate set did not extend beyond r, i.e.
    patches that may have been clipped by the hop budget rather than by the radius.
    """
    V = np.asarray(mesh.vertices)
    indptr, indices = R.indptr, R.indices
    counts_all = np.diff(indptr)
    rho = np.zeros(len(seeds))
    truncated = np.zeros(len(seeds), bool)
    for a in range(0, len(seeds), chunk):
        sd = seeds[a:a + chunk]
        starts, counts = indptr[sd], counts_all[sd]
        maxc = int(counts.max())
        cols = starts[:, None] + np.arange(maxc)[None, :]
        valid = np.arange(maxc)[None, :] < counts[:, None]
        idx = indices[np.clip(cols, 0, len(indices) - 1)]
        P = V[idx]
        d = np.linalg.norm(P - V[sd][:, None, :], axis=2)
        w = (valid & (d <= r)).astype(float)
        nw = np.maximum(w.sum(1), 1)
        truncated[a:a + chunk] = np.where(valid, d, -np.inf).max(1) <= r

        mu = (w[:, :, None] * P).sum(1) / nw[:, None]
        Q = (P - mu[:, None, :]) * w[:, :, None]
        C = np.einsum("smi,smj->sij", Q, Q) / nw[:, None, None]
        _, evecs = np.linalg.eigh(C)                      # ascending eigenvalues
        nrm, t1, t2 = evecs[:, :, 0], evecs[:, :, 1], evecs[:, :, 2]

        D = P - mu[:, None, :]
        x = np.einsum("smj,sj->sm", D, t1)
        y = np.einsum("smj,sj->sm", D, t2)
        z = np.einsum("smj,sj->sm", D, nrm)
        M = np.stack([x * x, x * y, y * y, x, y, np.ones_like(x)], axis=2)
        Mw = M * w[:, :, None]
        AtA = np.einsum("smi,smj->sij", Mw, M) + np.eye(6)[None] * 1e-12
        Atb = np.einsum("smi,sm->si", Mw, z)
        # NumPy >=2.0 no longer treats an (S,M)-shaped b as a batch of right-hand-side
        # vectors against (S,M,M) a; give it an explicit trailing axis and squeeze back.
        c = np.linalg.solve(AtA, Atb[..., None])[..., 0]
        res = z - np.einsum("smi,si->sm", M, c)
        rho[a:a + chunk] = np.sqrt((w * res**2).sum(1) / nw) / r
    if return_coverage:
        return rho, float(truncated.mean())
    return rho


def multiscale_roughness(mesh, scales, seeds=None, rings=None, coverage=None):
    """{scale: per-seed roughness}. seeds=None uses every vertex (the inference default)."""
    if seeds is None:
        seeds = np.arange(len(mesh.vertices))
    rings = ring_sets(mesh, scales) if rings is None else rings
    out = {}
    for s in scales:
        s = float(s)
        if coverage is None:
            out[s] = roughness_at_scale(mesh, s, rings[s], seeds)
        else:
            out[s], coverage[s] = roughness_at_scale(mesh, s, rings[s], seeds,
                                                     return_coverage=True)
    return out


def weighted_quantile(values, weights, quantiles=QUANTILES):
    """Area-weighted quantiles.

    Weighting by area rather than by vertex keeps the statistic consistent between the
    scan (which subsamples seeds) and inference (which uses every vertex), and stops
    densely tessellated regions from dominating.
    """
    values = np.asarray(values, float)
    weights = np.asarray(weights, float)
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cw = np.cumsum(w)
    if cw[-1] <= 0:
        return np.percentile(values, np.asarray(quantiles) * 100)
    cw = (cw - 0.5 * w) / cw[-1]
    return np.interp(np.asarray(quantiles), cw, v)


# ======================================================================================
# smoothing backends
# ======================================================================================

def _uniform_laplacian(mesh):
    n = len(mesh.vertices)
    e = mesh.edges_unique
    I = np.concatenate([e[:, 0], e[:, 1]])
    J = np.concatenate([e[:, 1], e[:, 0]])
    A = sp.coo_matrix((np.ones(len(I)), (I, J)), shape=(n, n)).tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel()
    return sp.diags(1.0 / np.maximum(deg, 1)) @ A - sp.identity(n, format="csr")


def smooth_mcf(mesh, frozen, sigma):
    """Implicit curvature flow for a diffusion time t = sigma^2 / 2.

    The step carries units of length^2, so `sigma` is the physical smoothing length: the
    filter attenuates surface detail below sigma and leaves structure well above it alone.
    That explicit scale control is why mcf is the default -- Taubin and HC have no
    calibrated relationship between iteration count and the size of what they remove.
    """
    L, M = cotangent_laplacian(mesh)
    V = np.asarray(mesh.vertices).copy()
    keep = V[frozen].copy()
    A = (M - (sigma ** 2 / 2.0) * L).tolil()
    for i in frozen:
        A.rows[i], A.data[i] = [i], [1.0]
    rhs = M @ V
    rhs[frozen] = keep
    out = spsolve(A.tocsc(), rhs)
    out[frozen] = keep
    return out


def _explicit_iters(mesh, sigma):
    h = float(mesh.edges_unique_length.mean())
    return max(1, int(round((sigma / h) ** 2)))


def smooth_taubin(mesh, frozen, sigma, lamb=0.5, mu=-0.53):
    """Taubin lambda|mu. `sigma` maps to an iteration count only approximately."""
    L = _uniform_laplacian(mesh)
    V = np.asarray(mesh.vertices).copy()
    keep = V[frozen].copy()
    for _ in range(_explicit_iters(mesh, sigma)):
        V += lamb * (L @ V)
        V += mu * (L @ V)
        V[frozen] = keep
    return V


def smooth_humphrey(mesh, frozen, sigma, alpha=0.1, beta=0.6):
    """HC-Laplacian (Vollmer et al.) with explicit shrinkage push-back."""
    L = _uniform_laplacian(mesh)
    O = np.asarray(mesh.vertices).copy()
    V, keep = O.copy(), O[frozen].copy()
    ident = sp.identity(len(V), format="csr")
    for _ in range(_explicit_iters(mesh, sigma)):
        Qv = V.copy()
        V = V + (L @ V)
        B = V - (alpha * O + (1 - alpha) * Qv)
        V -= beta * B + (1 - beta) * ((L + ident) @ B - B)
        V[frozen] = keep
    return V


_SMOOTH_FN = {"mcf": smooth_mcf, "taubin": smooth_taubin, "humphrey": smooth_humphrey}


def compensate_shrinkage(mesh, V_new, frozen, sigma=1.5, passes=3):
    """Undo the inward drift a curvature-driven filter produces, locally.

    Curvature flow moves a tube of radius R inward at ~t/R, so the drift scales as 1/R and
    a thin daughter branch shrinks several times faster than the parent vessel. A single
    global offset cannot correct a spatially varying field: forcing the *mean* drift to
    zero robs the thin vessels to inflate the thick ones. Measured over 8 rounds, a global
    correction still left sub-0.6 mm vessels ~8% narrower while expanding 2.2 mm+ regions
    by ~0.4%.

    Instead, split the normal displacement by spatial frequency. Bulk shrinkage is
    low-frequency -- it varies with calibre, over millimetres -- while the denoising we
    want to keep is high-frequency, at the target scales of 0.8 mm and below. Diffusing the
    normal displacement to a scale of `sigma` isolates the bulk component, and subtracting
    only that cancels the shrinkage while leaving the smoothing intact. Because the
    correction is per-vertex, thin vessels automatically get more pushback than thick ones.

    `sigma` must sit above the target scales (or it would undo the smoothing itself) and
    below the length over which calibre changes. Pass sigma=None for the old global
    behaviour.
    """
    V_old = np.asarray(mesh.vertices)
    n = np.asarray(mesh.vertex_normals)
    free = np.ones(len(V_old), bool)
    free[frozen] = False
    if free.sum() == 0:
        return V_new

    dn = np.einsum("ij,ij->i", V_new - V_old, n)
    V_new = V_new.copy()
    if sigma is None:
        a = vertex_areas(mesh)[free]
        drift = float(np.average(dn[free], weights=np.maximum(a, 1e-12)))
        V_new[free] -= drift * n[free]
        return V_new

    # One subtraction leaves residual bulk drift behind, because diffusion is not
    # idempotent: removing Heat(dn) still leaves Heat(dn) - Heat(Heat(dn)) of
    # low-frequency content. Iterating the same correction mops that up and works far
    # better than widening sigma, which would eat into the denoising band instead. On a
    # 0.5 mm branch, one pass leaves -4.4% shrinkage and three passes -1.4%, at a cost of
    # 0.1 in z(0.3). The operator is unchanged between passes, so factorize once.
    L, M = cotangent_laplacian(mesh)
    solve = splu((M - (sigma ** 2 / 2.0) * L).tocsc()).solve
    total = np.zeros(len(V_old))
    for _ in range(max(1, int(passes))):
        bulk = solve(M @ (dn - total))
        total += bulk
    V_new[free] -= total[free][:, None] * n[free]
    return V_new


# ======================================================================================
# remeshing backends
# ======================================================================================

def remesh_pymeshlab(mesh, target_edge, iterations=8):
    import pymeshlab
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces)))
    value = getattr(pymeshlab, "PureValue", None) or pymeshlab.AbsoluteValue
    ms.meshing_isotropic_explicit_remeshing(targetlen=value(target_edge),
                                            iterations=iterations)
    out = ms.current_mesh()
    return trimesh.Trimesh(out.vertex_matrix(), out.face_matrix(), process=False)


def remesh_vmtk(mesh, target_edge, iterations=8):
    """Best triangle quality, but undershoots the requested edge by ~15% and is ~40x
    slower than pymeshlab."""
    from vmtk import vmtkscripts
    r = vmtkscripts.vmtkSurfaceRemeshing()
    r.Surface = to_pyvista(mesh)
    r.ElementSizeMode = "edgelength"
    r.TargetEdgeLength = target_edge
    r.NumberOfIterations = iterations
    r.LogOn = 0
    r.Execute()
    out = pv.wrap(r.Surface).triangulate()
    return trimesh.Trimesh(np.asarray(out.points), out.faces.reshape(-1, 4)[:, 1:],
                           process=False)


def remesh(mesh, target_edge, backend="auto", iterations=8, calibrate=False):
    """Isotropically remesh to target_edge.

    calibrate=True divides out the remesher's known bias so the achieved edge
    length matches the request, which matters when the output has to hit a
    resolution a downstream consumer expects. Left off for the internal remesh so
    that roughness stays comparable with previously computed reference models.
    """
    if calibrate:
        target_edge = target_edge / PYMESHLAB_EDGE_BIAS
    if backend == "none":
        return mesh
    order = ["pymeshlab", "vmtk"] if backend == "auto" else [backend]
    errors = []
    for name in order:
        try:
            fn = remesh_pymeshlab if name == "pymeshlab" else remesh_vmtk
            out = fn(mesh, target_edge, iterations)
            out.remove_unreferenced_vertices()
            return out
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    raise RuntimeError("no remeshing backend available -> " + " | ".join(errors))


def surface_deviation(reference_mesh, points, sample=20000):
    """Point-to-surface distance from `points` to `reference_mesh`, in mm.

    Reported against the *original upload* so the figure covers remeshing as well as
    smoothing. Falls back to nearest-sampled-point if the exact query is unavailable.
    """
    try:
        _, dist, _ = trimesh.proximity.closest_point(reference_mesh, points)
        return float(np.max(dist)), float(np.mean(dist))
    except Exception:
        from scipy.spatial import cKDTree
        pts, _ = trimesh.sample.sample_surface(reference_mesh, sample)
        d = cKDTree(pts).query(points)[0]
        return float(np.max(d)), float(np.mean(d))


# ======================================================================================
# reference model
# ======================================================================================

ROUGHNESS_FLOOR = 1e-12          # guards log() against perfectly planar patches


def log_stats(curves):
    """Geometric mean and multiplicative spread of a stack of quantile curves.

    Statistics are kept in log space because roughness is positive and strongly
    right-skewed -- at the guard scale the across-case SD is comparable to the mean, so a
    linear mean +/- SD band dips below zero and a z of -1 would correspond to roughness
    near zero. That makes a linear guard almost impossible to trip: it would only fire
    after the surface had already been flattened. In log space the band is multiplicative,
    stays positive, and one SD means a consistent *factor* at every scale.
    """
    lg = np.log(np.maximum(np.asarray(curves, float), ROUGHNESS_FLOOR))
    return lg.mean(0), lg.std(0, ddof=1)


def _scan_one(job):
    """Measure one reference mesh. Module-level so it can be sent to a worker process."""
    path, scales, n_seeds, seed = job
    m = load_mesh(path)
    n = len(m.vertices)
    idx = (np.arange(n) if n_seeds is None or n_seeds >= n
           else np.random.default_rng(seed).choice(n, size=n_seeds, replace=False))
    vals = multiscale_roughness(m, scales, idx)
    w = vertex_areas(m)[idx]
    return ({float(s): weighted_quantile(vals[float(s)], w) for s in scales},
            float(m.edges_unique_length.mean()))


class ReferenceModel:
    """Geometric mean and multiplicative spread of the per-scale roughness quantile
    curves over a reference set. `mean` and `sd` are in log space; see log_stats."""

    def __init__(self, scales, quantiles, mean, sd, target_edge, n_cases, cases=None,
                 per_case=None, meta=None):
        self.scales = tuple(float(s) for s in scales)
        self.quantiles = np.asarray(quantiles, float)
        self.mean = {float(s): np.asarray(mean[float(s)], float) for s in self.scales}
        self.sd = {float(s): np.asarray(sd[float(s)], float) for s in self.scales}
        self.target_edge = float(target_edge)
        self.n_cases = int(n_cases)
        self.cases = list(cases or [])
        self.per_case = per_case or {}
        self.meta = dict(meta or {})

    # -- construction ------------------------------------------------------------------

    @classmethod
    def scan(cls, root=REFERENCE_ROOT, filename=REFERENCE_FILENAME,
             target_scales=DEFAULT_TARGET_SCALES, guard_scale=DEFAULT_GUARD_SCALE,
             n_seeds=4000, max_cases=None, seed=0, edge_tolerance=0.15, n_jobs=1,
             verbose=True):
        """Measure every reference mesh once. This is the expensive step; the result is
        cached so that trying a different smoother never requires rescanning.

        `target_scales` are the scales the regularizer will drive toward this reference and
        `guard_scale` the one it only checks for over-smoothing. Both are measured, and the
        roles are stored in the model so a later run cannot mistake one config for another.
        """
        target_scales = tuple(sorted(float(s) for s in target_scales))
        guard_scale = None if guard_scale is None else float(guard_scale)
        scales = tuple(sorted(set(target_scales) |
                              ({guard_scale} if guard_scale is not None else set())))
        for s in scales:
            if s <= 0:
                raise ValueError(f"scales must be positive, got {s}")

        cases = sorted(c for c in os.listdir(root) if os.path.isdir(os.path.join(root, c)))
        paths = [(c, os.path.join(root, c, filename)) for c in cases]
        paths = [(c, p) for c, p in paths if os.path.exists(p)]
        if max_cases:
            paths = paths[:max_cases]
        jobs = [(p, scales, n_seeds, seed + i) for i, (_, p) in enumerate(paths)]

        rows, edges, used, failures = {float(s): [] for s in scales}, [], [], []

        def collect(case, result):
            curves, edge = result
            for s in scales:
                rows[float(s)].append(curves[float(s)])
            edges.append(edge)
            used.append(case)
            if verbose and len(used) % 25 == 0:
                print(f"  scanned {len(used)} / {len(paths)} ...", flush=True)

        if n_jobs and n_jobs > 1:
            # Scanning is embarrassingly parallel and dominated by the per-patch fits, so
            # a pool cuts a multi-scale scan from an hour to a few minutes. Each worker is
            # pinned to one BLAS thread, otherwise the pool and BLAS oversubscribe cores.
            import multiprocessing as mp
            for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
                os.environ.setdefault(var, "1")
            with mp.get_context("spawn").Pool(n_jobs) as pool:
                for (case, _), res in zip(paths, pool.imap(_scan_one, jobs, chunksize=1)):
                    try:
                        collect(case, res)
                    except Exception as exc:
                        failures.append({"case": case, "error": repr(exc)})
        else:
            for (case, path), job in zip(paths, jobs):
                try:
                    collect(case, _scan_one(job))
                except Exception as exc:
                    failures.append({"case": case, "error": repr(exc)})
                    if verbose:
                        print(f"  [skip] {case}: {exc}")
        if not used:
            raise RuntimeError(f"no reference meshes found under {root}/*/{filename}")

        # Meshes tessellated very differently from the rest are not comparable, and a
        # single outlier would distort both the mean curve and its spread.
        edges = np.asarray(edges)
        med = float(np.median(edges))
        keep = np.abs(edges - med) / med <= edge_tolerance
        dropped = [used[i] for i in np.nonzero(~keep)[0]]
        if verbose and dropped:
            print(f"  dropped {len(dropped)} case(s) on edge length: {dropped[:5]}")
        used = [c for c, k in zip(used, keep) if k]
        per_case = {float(s): np.asarray(rows[float(s)])[keep] for s in scales}

        n_kept = len(used)
        if n_kept < 2:
            raise RuntimeError(
                f"need at least 2 reference cases to estimate a spread, got {n_kept}")
        mean, sd = {}, {}
        for s in per_case:
            mean[s], sd[s] = log_stats(per_case[s])
        meta = {
            "stat_space": "log",
            "target_scales": list(target_scales), "guard_scale": guard_scale,
            "root": root, "filename": filename, "n_seeds": n_seeds,
            "scales": [float(s) for s in scales], "edge_tolerance": edge_tolerance,
            "edge_median": med, "edge_p05": float(np.percentile(edges[keep], 5)),
            "edge_p95": float(np.percentile(edges[keep], 95)),
            "n_dropped_on_edge": int((~keep).sum()), "dropped_cases": dropped,
            "failures": failures,
            "scanned_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        return cls(scales, QUANTILES, mean, sd, med, n_kept, used, per_case, meta)

    # -- persistence -------------------------------------------------------------------

    def save(self, path, write_sidecar=True):
        # Everything non-numeric goes through a JSON string rather than a pickled object
        # array, so the cache can be loaded without allow_pickle.
        meta = dict(self.meta)
        meta["cases"] = self.cases
        payload = {"quantiles": self.quantiles, "scales": np.array(self.scales),
                   "target_edge": self.target_edge, "n_cases": self.n_cases,
                   "meta_json": json.dumps(meta)}
        for s in self.scales:
            payload[f"mean_{s}"] = self.mean[s]
            payload[f"sd_{s}"] = self.sd[s]
            if s in self.per_case:
                payload[f"per_case_{s}"] = self.per_case[s]
        np.savez_compressed(path, **payload)
        if write_sidecar:
            with open(os.path.splitext(path)[0] + ".json", "w") as f:
                json.dump({"summary": self.summary(), "provenance": self.meta,
                           "cases": self.cases}, f, indent=2)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        scales = tuple(float(s) for s in d["scales"])
        mean = {s: d[f"mean_{s}"] for s in scales}
        sd = {s: d[f"sd_{s}"] for s in scales}
        per_case = {s: d[f"per_case_{s}"] for s in scales if f"per_case_{s}" in d}
        meta = json.loads(str(d["meta_json"])) if "meta_json" in d else {}
        return cls(scales, d["quantiles"], mean, sd, float(d["target_edge"]),
                   int(d["n_cases"]), meta.get("cases", []), per_case, meta)

    # -- use ---------------------------------------------------------------------------

    @property
    def target_scales(self):
        """Scales this model was built to be driven toward."""
        t = self.meta.get("target_scales")
        if t:
            return tuple(float(s) for s in t if float(s) in self.scales)
        return tuple(s for s in self.scales if s in DEFAULT_TARGET_SCALES)

    @property
    def guard_scale(self):
        """Scale used only as an over-smoothing floor, or None if this model has none."""
        g = self.meta.get("guard_scale", DEFAULT_GUARD_SCALE)
        if g is None:
            return None
        return float(g) if float(g) in self.scales else None

    @property
    def config_name(self):
        return config_stem(self.target_scales, self.guard_scale
                           if self.guard_scale is not None else 0)

    def z_curve(self, scale, curve):
        """Deviation of a curve from the reference, in reference SD units (log space)."""
        s = float(scale)
        lg = np.log(np.maximum(np.asarray(curve, float), ROUGHNESS_FLOOR))
        return (lg - self.mean[s]) / np.maximum(self.sd[s], 1e-12)

    def band(self, scale, n_sd=1.0):
        """(low, high) roughness curves bounding the reference band, in linear units."""
        s = float(scale)
        return np.exp(self.mean[s] - n_sd * self.sd[s]), np.exp(self.mean[s] + n_sd * self.sd[s])

    def geometric_mean(self, scale):
        return np.exp(self.mean[float(scale)])

    def discrepancy(self, scale, curve):
        """Mean |z| over the quantile grid. Reported for interpretability; the controller
        steers on signed_excess instead."""
        return float(np.abs(self.z_curve(scale, curve)).mean())

    def signed_excess(self, scale, curve):
        """Mean signed z: positive means rougher than reference, negative smoother."""
        return float(self.z_curve(scale, curve).mean())

    def below_band(self, scale, curve, quantile_min=GUARD_QUANTILE_MIN):
        """How far the curve sits below the reference band, in SD units, measured on the
        upper quantiles only.

        Restricting to the tail matters: focal features such as blebs occupy little surface
        area, so a serious drop confined to them would be diluted to nothing by averaging
        over the whole quantile grid.
        """
        z = self.z_curve(scale, curve)
        sel = self.quantiles >= quantile_min
        if not sel.any():
            sel = np.ones_like(self.quantiles, bool)
        return float(np.maximum(-z[sel], 0).mean())

    def summary(self):
        out = {"n_cases": self.n_cases, "target_edge": self.target_edge,
               "stat_space": "log", "scales": {}}
        for s in self.scales:
            gm = self.geometric_mean(s)
            qs = {f"p{int(q * 100)}": float(np.interp(q, self.quantiles, gm))
                  for q in REPORT_QUANTILES}
            # one SD in log space is a multiplicative factor, the same at every quantile
            qs["sd_factor_at_p50"] = float(np.exp(np.interp(0.5, self.quantiles, self.sd[s])))
            qs["role"] = "guard" if s == self.guard_scale else "target"
            out["scales"][str(s)] = qs
        return out


# ======================================================================================
# plots
# ======================================================================================

def plot_reference(model, path):
    """Reference roughness per scale: quantile curves with spread, and the across-case
    distributions of median roughness and of heterogeneity."""
    scales = sorted(model.scales)
    fig, axes = plt.subplots(2, len(scales), figsize=(4.7 * len(scales), 7.6))
    axes = np.atleast_2d(axes)
    for j, s in enumerate(scales):
        gm = model.geometric_mean(s)
        lo1, hi1 = model.band(s, 1.0)
        lo2, hi2 = model.band(s, 2.0)
        ax = axes[0, j]
        if s in model.per_case:
            for row in model.per_case[s][:40]:
                ax.plot(model.quantiles, row, color="grey", lw=0.4, alpha=0.35)
        ax.plot(model.quantiles, gm, color="#1f77b4", lw=2, label="reference geometric mean")
        ax.fill_between(model.quantiles, lo1, hi1, color="#1f77b4", alpha=0.25,
                        label="+/- 1 SD across cases (log)")
        ax.fill_between(model.quantiles, lo2, hi2, color="#1f77b4", alpha=0.10,
                        label="+/- 2 SD")
        role = "guard" if s == model.guard_scale else "target"
        ax.set_title(f"scale r = {s} mm  ({role})")
        ax.set_xlabel("quantile within a shape")
        ax.set_ylabel("roughness (RMS quadric residual / r)")
        ax.set_yscale("log")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        ax = axes[1, j]
        if s in model.per_case:
            pc = model.per_case[s]
            med = np.array([np.interp(0.5, model.quantiles, r) for r in pc])
            het = np.array([np.interp(0.9, model.quantiles, r) /
                            max(np.interp(0.5, model.quantiles, r), 1e-12) for r in pc])
            ax.hist(med, bins=40, color="#2ca02c", alpha=0.85)
            ax.set_xlabel("per-shape median roughness")
            ax.set_ylabel("cases")
            ax.set_title(f"across {len(pc)} reference cases")
            ax2 = ax.twiny()
            ax2.hist(het, bins=40, color="#d62728", alpha=0.35)
            ax2.set_xlabel("heterogeneity p90/p50", color="#d62728")
            ax.grid(alpha=0.3)
    fig.suptitle(f"Reference roughness model -- {model.n_cases} cases, "
                 f"edge {model.target_edge:.4f} mm", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_convergence(history, model, path, title="", tolerance=1.0,
                     overshoot_limit=-1.0):
    """Per-shape convergence trace and the before/after quantile curves against the band."""
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5))
    rounds = [h["round"] for h in history]

    ax = axes[0]
    for s in model.target_scales:
        ax.plot(rounds, [h["signed_excess"][str(float(s))] for h in history], marker="o",
                label=f"signed excess r={s} (target)")
    if model.guard_scale is not None:
        ax.plot(rounds, [h["below_band"][str(float(model.guard_scale))] for h in history],
                marker="s", color="#d62728",
                label=f"below-band r={model.guard_scale} (guard)")
    ax.axhline(tolerance, color="k", ls="--", lw=1, label=f"tolerance = {tolerance:g} SD")
    ax.axhline(overshoot_limit, color="k", ls=":", lw=1,
               label=f"overshoot limit = {overshoot_limit:g} SD")
    ax.axhline(0.0, color="grey", lw=0.8)
    ax.set_xlabel("smoothing round")
    ax.set_ylabel("deviation from reference (SD units)")
    ax.set_title("convergence")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)
    # cumulative wall clock, so cost is visible next to progress
    secs = [h.get("elapsed_s", np.nan) for h in history]
    axt = ax.twinx()
    axt.plot(rounds, secs, color="#8c564b", lw=1.2, ls="-.", marker=".", ms=4,
             label="elapsed (s)")
    axt.set_ylabel("cumulative wall clock (s)", color="#8c564b")
    axt.tick_params(axis="y", labelcolor="#8c564b")
    axt.legend(fontsize=7, loc="lower right")

    ax = axes[1]
    ax.plot(rounds, [h["deviation_mm"] for h in history], marker="o", color="#9467bd")
    for h in history[::max(1, len(history) // 6)]:
        if h.get("elapsed_s") is not None:
            ax.annotate(f"{h['elapsed_s']:.0f}s", (h["round"], h["deviation_mm"]),
                        textcoords="offset points", xytext=(4, -9), fontsize=6,
                        color="#8c564b")
    ax.set_xlabel("smoothing round")
    ax.set_ylabel("max vertex displacement (mm)")
    ax.set_title("geometric change during smoothing")
    ax.grid(alpha=0.3)

    ax = axes[2]
    colors = {0.3: "#1f77b4", 0.5: "#2ca02c", 0.8: "#d62728"}
    for s in sorted(model.scales):
        col = colors.get(s, "grey")
        lo, hi = model.band(s, 1.0)
        ax.fill_between(model.quantiles, lo, hi, color=col, alpha=0.18)
        ax.plot(model.quantiles, model.geometric_mean(s), color=col, lw=1.6,
                label=f"reference r={s}")
        ax.plot(model.quantiles, history[0]["curves"][str(s)], color=col, ls=":", lw=1.4,
                label=f"input r={s}")
        ax.plot(model.quantiles, history[-1]["curves"][str(s)], color=col, ls="--", lw=1.8,
                label=f"output r={s}")
    ax.set_yscale("log")
    ax.set_xlabel("quantile within shape")
    ax.set_ylabel("roughness")
    ax.set_title("quantile curves vs reference")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ======================================================================================
# regularizer
# ======================================================================================

class MeshRegularizer:
    """Smooth one uploaded shape until its roughness matches the reference.

    Built for single-shape inference: nothing here needs a batch of targets, so users can
    upload one mesh at a time and each is normalized against the cached reference model.

    Parameters
    ----------
    model : ReferenceModel, optional
        Loaded from `cache_path` on first use if not supplied. Scanning is never triggered
        implicitly -- trying a different smoother reuses the cached statistics.
    smoother : {'mcf', 'humphrey', 'taubin'}
        'mcf' is the default because its step is a physical smoothing length; the other two
        have no calibrated relationship between iterations and feature size.
    sigma_step : float
        Physical smoothing length per round, in mm. Halved automatically when a step would
        overshoot below the reference band.
    tolerance : float
        Converged when the signed excess at every target scale drops to this many SD above
        the reference. This is the aggressiveness knob: 1.0 stops as soon as the shape is
        within one SD of typical, 0.0 drives it to the reference mean, and negative values
        deliberately smooth it below typical. Note that *which scales are targets* is not
        an aggressiveness knob -- the finest scale starts furthest from the reference and
        binds the loop, so adding coarser targets that are already inside tolerance changes
        nothing.
    overshoot_limit : float
        Reject a step that would push any target scale below this signed z, and halve the
        step instead. Kept independent of `tolerance` so that an aggressive negative
        tolerance is still reachable; it must sit below `tolerance`.
    guard_tolerance : float
        Stop and flag if the guard scale falls this far below the reference band.
    export_edge : float or None
        Edge length (mm) the *exported* mesh is remeshed to, after regularization and
        after the metrics are taken. Defaults to the downstream training resolution
        (DOWNSTREAM_EDGE_MM); pass None or 0 to leave the mesh at the reference resolution.
    max_deviation : float or None
        Backstop on vertex displacement during smoothing, in mm. This covers the smoothing
        stage only; the report also carries a surface-to-surface distance to the original
        upload, which includes the initial remesh.
    """

    def __init__(self, model=None, cache_path=_DEFAULT_CACHE, smoother="mcf",
                 remesher="auto", sigma_step=0.08, tolerance=1.0, guard_tolerance=1.0,
                 max_rounds=30, max_deviation=0.4, compensate=True, n_seeds=None,
                 max_halvings=4, target_scales=None, guard_scale=None,
                 overshoot_limit=-1.0, min_progress=0.01, time_budget_s=60.0,
                 min_seeds=800, overhead_reserve_s=10.0, retry_factor=2.5,
                 compensate_sigma="auto", compensate_passes=3,
                 export_edge=DOWNSTREAM_EDGE_MM, verbose=True):
        if smoother not in SMOOTHERS:
            raise ValueError(f"smoother must be one of {SMOOTHERS}")
        if remesher not in REMESHERS:
            raise ValueError(f"remesher must be one of {REMESHERS}")
        if overshoot_limit >= tolerance:
            raise ValueError(f"overshoot_limit ({overshoot_limit}) must be below "
                             f"tolerance ({tolerance}); otherwise every step is rejected")
        for name, val in [("sigma_step", sigma_step),
                          ("guard_tolerance", guard_tolerance), ("max_rounds", max_rounds),
                          ("max_halvings", max_halvings)]:
            if val is None or val <= 0:
                raise ValueError(f"{name} must be positive, got {val}")
        if max_deviation is not None and max_deviation <= 0:
            raise ValueError(f"max_deviation must be positive or None, got {max_deviation}")
        if n_seeds not in (None, "auto") and int(n_seeds) <= 0:
            raise ValueError(f"n_seeds must be positive or None, got {n_seeds}")

        self.cache_path = cache_path
        self._model = model
        self.smoother = smoother
        self.remesher = remesher
        self.sigma_step = sigma_step
        self.tolerance = tolerance
        self.overshoot_limit = overshoot_limit
        self.min_progress = min_progress
        self.guard_tolerance = guard_tolerance
        self.max_rounds = max_rounds
        self.max_deviation = max_deviation
        self.compensate = compensate
        # length scale of the bulk-drift correction; see compensate_shrinkage
        self._compensate_sigma = compensate_sigma
        self.compensate_passes = compensate_passes
        # final remesh applied to the exported mesh only; None/0 keeps the
        # reference resolution the regularization ran at
        self.export_edge = float(export_edge) if export_edge else None
        # None -> every vertex; an int -> that many; 'auto' -> fit time_budget_s
        self.n_seeds = n_seeds
        # None (or a non-positive value) disables both the wall-clock stop and the
        # seed budget, so a run goes to full fidelity for as long as it needs.
        self.time_budget_s = (None if time_budget_s is None or time_budget_s <= 0
                              else float(time_budget_s))
        self.retry_factor = retry_factor
        self.min_seeds = min_seeds
        self.overhead_reserve_s = overhead_reserve_s
        self.max_halvings = max_halvings
        # None means "use the roles recorded in the model", which is almost always right
        self._target_override = (tuple(sorted(float(s) for s in target_scales))
                                 if target_scales else None)
        self._guard_override = None if guard_scale is None else float(guard_scale)
        self.verbose = verbose

    @property
    def model(self):
        if self._model is None:
            if not os.path.exists(self.cache_path):
                raise RuntimeError(
                    f"no reference model at {self.cache_path}; run the 'scan' command first")
            self._model = ReferenceModel.load(self.cache_path)
        return self._model

    @property
    def compensate_sigma(self):
        """Length scale of the anti-shrinkage correction.

        'auto' keeps it clear of the coarsest target scale. The correction removes
        normal motion below 1/sigma, so if sigma sat near a target scale the compensator
        would undo the very smoothing that target is asking for -- a config targeting
        1.2 mm needs a wider sigma than one topping out at 0.8 mm.
        """
        s = self._compensate_sigma
        if s != "auto":
            return s
        return max(1.5, 2.0 * max(self.target_scales))

    @property
    def target_scales(self):
        """Scales driven toward the reference, as recorded in the model."""
        return self._target_override or self.model.target_scales

    @property
    def guard_scale(self):
        if self._guard_override is not None:
            return self._guard_override
        return self.model.guard_scale

    def _validate_model(self):
        if not self.target_scales:
            raise RuntimeError(
                f"reference model has scales {self.model.scales} but none match the target "
                f"scales {self.target_scales}; rescan with matching scales")
        if self.guard_scale is None and self.verbose:
            print(f"  [warn] model has no guard scale; "
                  f"over-smoothing protection is disabled")

    def _auto_seed_count(self, mesh, rings, frozen, n):
        """Pick a seed count that keeps one case inside `time_budget_s`.

        Cost is dominated by the per-patch quadric fits, which scale linearly with the
        seed count and steeply with the number and size of the scales -- measuring every
        vertex of a 24k mesh across six scales takes ~39 s, so 17 rounds of that would run
        to 11 minutes per case. Everything else is near-fixed, so the budget is spent by
        timing a small probe and solving for how many seeds fit in what remains.
        """
        if self.time_budget_s is None:
            return n
        probe = min(500, n)
        idx = np.random.default_rng(0).choice(n, size=probe, replace=False)
        t0 = time.perf_counter()
        multiscale_roughness(mesh, self.model.scales, idx, rings)
        per_seed = (time.perf_counter() - t0) / max(probe, 1)
        t0 = time.perf_counter()
        _SMOOTH_FN[self.smoother](mesh, frozen, self.sigma_step)
        per_step = time.perf_counter() - t0

        # A round is not one smooth plus one measurement. When a step overshoots the band
        # the loop halves sigma and retries, and each retry costs another smooth *and*
        # another measurement -- at negative tolerances that happens on most rounds. An
        # estimate that ignores retries underestimates cost by 2-3x, so budget for them.
        n_measure = (self.max_rounds + 1) * self.retry_factor
        available = (self.time_budget_s - self.max_rounds * per_step * self.retry_factor
                     - self.overhead_reserve_s)
        k = int(available / max(n_measure * per_seed, 1e-12))
        k = int(np.clip(k, self.min_seeds, n))
        if self.verbose:
            print(f"  [auto-seed] {k} of {n} vertices "
                  f"({per_seed * 1e3:.2f} ms/seed, {per_step:.2f} s/step, "
                  f"budget {self.time_budget_s:.0f} s)")
        return k

    def _curves(self, mesh, rings, seeds, weights):
        vals = multiscale_roughness(mesh, self.model.scales, seeds, rings)
        return {s: weighted_quantile(vals[s], weights, self.model.quantiles)
                for s in self.model.scales}

    def forward(self, mesh, freeze_boundary=True, plot_path=None, title=""):
        """Remesh, then smooth in rounds until the target scales match the reference.

        Returns (mesh, report). `status` is one of:
          converged             target scales reached the reference band
          overshoot_limited     could not reach tolerance without dropping below the band;
                                the last in-band state is returned
          guard_stopped         stopped early: the guard scale would have fallen below the
                                reference band, i.e. anatomy was at risk
          hit_max_rounds        ran out of rounds still above tolerance
          stalled_at_overshoot_limit
                                the target cannot be reached without pushing another scale
                                below overshoot_limit, so progress stopped
          stopped_on_deviation  displacement backstop tripped
          time_budget_exceeded  hit the wall-clock limit set by time_budget_s
        """
        model = self.model
        self._validate_model()
        t_start = time.perf_counter()
        original = load_mesh(mesh) if isinstance(mesh, str) else mesh
        info = inspect_mesh(original)
        for w in info["warnings"]:
            if self.verbose:
                print(f"  [warn] {w}")

        work = remesh(original, model.target_edge, self.remesher)
        # Nearest-point distance to the remeshed input, NOT per-vertex correspondence.
        # Curvature flow slides vertices tangentially along the surface, and correspondence
        # distance counts that sliding as if the surface had moved. On a noisy case it read
        # 0.332 mm of "deviation" where the surface had actually moved only 0.081 mm,
        # tripping the backstop after 4 rounds on a shape that needed ~15.
        anchor = cKDTree(np.asarray(work.vertices).copy())
        rings = ring_sets(work, model.scales)
        rng = np.random.default_rng(0)
        n = len(work.vertices)
        frozen = boundary_vertices(work) if freeze_boundary else np.zeros(0, dtype=np.int64)
        if self.n_seeds == "auto":
            k = self._auto_seed_count(work, rings, frozen, n)
        elif self.n_seeds is None:
            k = n
        else:
            k = min(int(self.n_seeds), n)
        seeds = np.arange(n) if k >= n else rng.choice(n, size=k, replace=False)
        weights = vertex_areas(work)[seeds]

        coverage = {}
        multiscale_roughness(work, model.scales, seeds, rings, coverage=coverage)
        if self.verbose and max(coverage.values()) > 0.02:
            print(f"  [warn] patch coverage truncated for "
                  f"{max(coverage.values()) * 100:.1f}% of seeds at some scale")

        gs = self.guard_scale

        def snapshot(rnd, curves, deviation, sigma):
            return {
                "round": rnd, "sigma_mm": sigma,
                "curves": {str(s): np.asarray(curves[s]).tolist() for s in model.scales},
                "signed_excess": {str(s): model.signed_excess(s, curves[s])
                                  for s in model.scales},
                "discrepancy": {str(s): model.discrepancy(s, curves[s])
                                for s in model.scales},
                "below_band": {str(s): model.below_band(s, curves[s])
                               for s in model.scales},
                "deviation_mm": deviation,
                "elapsed_s": round(time.perf_counter() - t_start, 2),
            }

        curves = self._curves(work, rings, seeds, weights)
        history = [snapshot(0, curves, 0.0, 0.0)]
        # A shape whose guard scale is already below the reference cannot be protected from
        # dropping further; flag it rather than refusing to do anything.
        guard_low_at_input = (gs is not None and
                              history[0]["below_band"][str(gs)] > self.guard_tolerance)

        def worst_excess(c):
            return max(model.signed_excess(s, c[s]) for s in self.target_scales)

        def most_negative(c):
            return min(model.signed_excess(s, c[s]) for s in self.target_scales)

        sigma = self.sigma_step
        sigma_sq_total = 0.0
        status = "hit_max_rounds"
        for rnd in range(1, self.max_rounds + 1):
            # Wall-clock stop, enforced at round boundaries: a round already in flight is
            # allowed to finish, so actual elapsed time can exceed the budget by up to one
            # round (1-3 s in practice). The seed estimator only predicts cost -- this is
            # what makes the bound hold when the prediction is wrong.
            if (self.time_budget_s is not None
                    and time.perf_counter() - t_start > self.time_budget_s):
                status = "time_budget_exceeded"
                break
            if worst_excess(curves) <= self.tolerance:
                status = "converged"
                break

            accepted = False
            for _ in range(self.max_halvings + 1):
                V = _SMOOTH_FN[self.smoother](work, frozen, sigma)
                if self.compensate:
                    V = compensate_shrinkage(work, V, frozen,
                                             sigma=self.compensate_sigma,
                                             passes=self.compensate_passes)
                cand = trimesh.Trimesh(V, work.faces, process=False)
                cand_curves = self._curves(cand, rings, seeds, weights)
                if most_negative(cand_curves) < self.overshoot_limit:
                    sigma /= 2.0                 # overshot the band: retry gentler
                    continue
                accepted = True
                break
            if not accepted:
                status = "overshoot_limited"
                break

            deviation = float(anchor.query(V)[0].max())
            if self.max_deviation is not None and deviation > self.max_deviation:
                status = "stopped_on_deviation"
                break
            if gs is not None and not guard_low_at_input:
                if model.below_band(gs, cand_curves[gs]) > self.guard_tolerance:
                    status = "guard_stopped"
                    break

            # When the target is unreachable without breaching overshoot_limit, the step
            # keeps halving and progress decays geometrically toward the limit. Detect the
            # stall rather than burning the remaining rounds on negligible movement.
            progress = worst_excess(curves) - worst_excess(cand_curves)
            work, curves = cand, cand_curves
            sigma_sq_total += sigma ** 2
            history.append(snapshot(rnd, curves, deviation, sigma))
            if progress < self.min_progress:
                status = ("converged" if worst_excess(curves) <= self.tolerance
                          else "stalled_at_overshoot_limit")
                break
        if status == "hit_max_rounds" and worst_excess(curves) <= self.tolerance:
            status = "converged"          # convergence reached on the final round

        # Regularization runs at the reference edge length, because that is what makes the
        # roughness z-scores comparable to the reference model. Downstream consumers were
        # trained on a slightly different tessellation, so the exported mesh gets one final
        # remesh to their resolution. Measured over 100 downstream training cases the wall
        # meshes average 0.125 mm against the reference 0.132 mm -- only 5% apart and well
        # inside the +/-17% spread of the training data, so this is a safety measure rather
        # than a correction for anything known to be broken.
        #
        # The reported roughness metrics deliberately stay those measured at the reference
        # resolution before this step; re-measuring here would need the ring sets rebuilt
        # and would no longer be an apples-to-apples comparison against the reference.
        if self.export_edge:
            work = remesh(work, self.export_edge, self.remesher, calibrate=True)

        max_dev, mean_dev = surface_deviation(original, np.asarray(work.vertices))
        last = history[-1]
        report = {
            "status": status,
            "converged": status == "converged",
            "guard_scale_below_reference_at_input": bool(guard_low_at_input),
            "smoother": self.smoother, "remesher": self.remesher,
            "sigma_step_mm": self.sigma_step, "tolerance_sd": self.tolerance,
            "time_budget_s": self.time_budget_s,
            "overshoot_limit_sd": self.overshoot_limit,
            "guard_tolerance_sd": self.guard_tolerance,
            "target_scales": list(self.target_scales), "guard_scale": gs,
            "rounds": last["round"],
            "total_sigma_mm": float(np.sqrt(sigma_sq_total)),
            "target_edge": model.target_edge,
            "export_edge": self.export_edge,
            "metrics_measured_at_edge": model.target_edge,
            "n_vertices_in": info["n_vertices"], "n_vertices_out": int(len(work.vertices)),
            "n_seeds_used": int(len(seeds)),
            "elapsed_s": round(time.perf_counter() - t_start, 2),
            "edge_out": float(work.edges_unique_length.mean()),
            "quality_out": float(triangle_quality(work).mean()),
            "smoothing_displacement_mm": last["deviation_mm"],
            "surface_deviation_vs_original_max_mm": max_dev,
            "surface_deviation_vs_original_mean_mm": mean_dev,
            "patch_truncation_fraction": {str(k): v for k, v in coverage.items()},
            "signed_excess_before": history[0]["signed_excess"],
            "signed_excess_after": last["signed_excess"],
            "below_band_after": last["below_band"],
            "input_inspection": info,
            "reference_cases": model.n_cases,
            "history": history,
        }
        if plot_path:
            plot_convergence(history, model, plot_path,
                             title=(title or status) +
                                   f"   [{status}, {report['elapsed_s']:.0f} s, "
                                   f"{last['round']} rounds, "
                                   f"{len(seeds)} seeds]",
                             tolerance=self.tolerance,
                             overshoot_limit=self.overshoot_limit)

        if self.verbose:
            b, a = history[0]["signed_excess"], last["signed_excess"]
            scales_txt = "  ".join(f"z({s}) {b[str(s)]:+.2f}->{a[str(s)]:+.2f}"
                                   for s in self.target_scales)
            guard_txt = f"guard({gs}) {last['below_band'][str(gs)]:.2f}  " if gs else ""
            print(f"  [{status}] rounds={last['round']} {scales_txt}  {guard_txt}"
                  f"dev(smooth) {last['deviation_mm']:.3f} mm  "
                  f"dev(vs original) {max_dev:.3f} mm  "
                  f"V {info['n_vertices']}->{len(work.vertices)}")
        return work, report

    __call__ = forward


# ======================================================================================
# CLI
# ======================================================================================

def _resolve_cache(args):
    """Explicit --cache_path wins; otherwise the path is derived from the scale roles so
    that each configuration lands in its own file inside --model_dir."""
    if args.cache_path:
        return args.cache_path
    os.makedirs(args.model_dir, exist_ok=True)
    return model_path(args.target_scales, args.guard_scale, args.model_dir)


def cmd_scan(args):
    cache = _resolve_cache(args)
    if os.path.exists(cache) and not args.overwrite:
        raise SystemExit(
            f"{cache} already exists. Scanning is expensive and the cache is meant to be "
            f"reused -- pass --overwrite to replace it.")
    os.makedirs(os.path.dirname(os.path.abspath(cache)), exist_ok=True)
    print(f"scanning {args.reference_root}/*/{args.reference_filename} "
          f"with {args.n_seeds} seeds per mesh")
    print(f"  targets {tuple(args.target_scales)} mm | guard {args.guard_scale} mm")
    model = ReferenceModel.scan(args.reference_root, args.reference_filename,
                                target_scales=args.target_scales,
                                guard_scale=args.guard_scale,
                                n_seeds=args.n_seeds, max_cases=args.max_cases,
                                n_jobs=args.n_jobs)
    model.save(cache)
    plot_path = os.path.splitext(cache)[0] + "_distribution.png"
    plot_reference(model, plot_path)
    print(json.dumps(model.summary(), indent=2))
    print(f"\nmodel    -> {cache}")
    print(f"summary  -> {os.path.splitext(cache)[0]}.json")
    print(f"plot     -> {plot_path}")
    if model.meta.get("failures"):
        print(f"{len(model.meta['failures'])} case(s) failed; see the summary json")


def cmd_run(args):
    reg = MeshRegularizer(cache_path=_resolve_cache(args), smoother=args.smoother,
                          remesher=args.remesher, sigma_step=args.sigma_step,
                          tolerance=args.tolerance, guard_tolerance=args.guard_tolerance,
                          max_rounds=args.max_rounds, max_deviation=args.max_deviation,
                          overshoot_limit=args.overshoot_limit,
                          min_progress=args.min_progress,
                          n_seeds=(None if args.n_seeds == 'all'
                                   else args.n_seeds if args.n_seeds == 'auto'
                                   else int(args.n_seeds)),
                          time_budget_s=args.time_budget,
                          compensate=not args.no_compensate,
                          compensate_sigma=(args.compensate_sigma
                                            if args.compensate_sigma == 'auto'
                                            else None if float(args.compensate_sigma) <= 0
                                            else float(args.compensate_sigma)),
                          compensate_passes=args.compensate_passes,
                          export_edge=args.export_edge)
    m = reg.model
    print(f"reference: {m.n_cases} cases, edge {m.target_edge:.4f} mm, "
          f"targets {reg.target_scales} mm, guard {reg.guard_scale} mm")

    cases = sorted(c for c in os.listdir(args.input_root)
                   if os.path.isdir(os.path.join(args.input_root, c)))
    done, outcomes = 0, []
    for case in cases:
        src = os.path.join(args.input_root, case, args.input_filename)
        dst = os.path.join(args.input_root, case, args.output_filename)
        if not os.path.exists(src) or (os.path.exists(dst) and not args.overwrite):
            continue
        print(f"[{case}]")
        stem = os.path.splitext(dst)[0]
        try:
            out, report = reg.forward(src, plot_path=stem + "_convergence.png", title=case)
        except Exception as exc:
            print(f"  failed: {exc}")
            outcomes.append({"case": case, "status": "failed", "error": repr(exc)})
            continue
        out.export(dst)
        with open(stem + "_report.json", "w") as f:
            json.dump(report, f, indent=2)
        outcomes.append({
            "case": case, "status": report["status"], "rounds": report["rounds"],
            "surface_deviation_max_mm": report["surface_deviation_vs_original_max_mm"],
        })
        done += 1
        if args.limit and done >= args.limit:
            break

    summary_path = os.path.join(args.input_root, "regularization_summary.json")
    with open(summary_path, "w") as f:
        json.dump(outcomes, f, indent=2)
    tally = {}
    for o in outcomes:
        tally[o["status"]] = tally.get(o["status"], 0) + 1
    print(f"regularized {done} cases -> {tally}")
    print(f"batch summary -> {summary_path}")


def main():
    p = argparse.ArgumentParser(
        description="Normalize vessel-surface roughness against a physiological reference.")
    p.add_argument("--cache_path", default=None,
                   help="explicit model file; by default derived from the scale roles")
    p.add_argument("--model_dir", default=DEFAULT_MODEL_DIR,
                   help="where config-stamped reference models are stored")
    p.add_argument("--target_scales", type=float, nargs="+",
                   default=list(DEFAULT_TARGET_SCALES),
                   help="scales driven toward the reference, mm")
    p.add_argument("--guard_scale", type=float, default=DEFAULT_GUARD_SCALE,
                   help="scale used only as an over-smoothing floor, mm")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("scan", help="measure the reference dataset once and cache it")
    s.add_argument("--reference_root", default=REFERENCE_ROOT)
    s.add_argument("--reference_filename", default=REFERENCE_FILENAME)
    s.add_argument("--n_seeds", type=int, default=4000)
    s.add_argument("--max_cases", type=int, default=None)
    s.add_argument("--n_jobs", type=int, default=1,
                   help="parallel worker processes for scanning")
    s.add_argument("--overwrite", action="store_true", help="replace an existing cache")
    s.set_defaults(func=cmd_scan)

    r = sub.add_parser("run", help="regularize shapes against the cached model")
    r.add_argument("--input_root", default=INPUT_ROOT)
    r.add_argument("--input_filename", default=INPUT_FILENAME)
    r.add_argument("--output_filename", default="regularized.obj")
    r.add_argument("--smoother", default="mcf", choices=SMOOTHERS)
    r.add_argument("--remesher", default="auto", choices=REMESHERS)
    r.add_argument("--sigma_step", type=float, default=0.08,
                   help="physical smoothing length per round, mm")
    r.add_argument("--tolerance", type=float, default=1.0,
                   help="stop at this signed z above the reference; lower = more "
                        "aggressive (0 = reference mean, negative = smoother than typical)")
    r.add_argument("--overshoot_limit", type=float, default=-1.0,
                   help="reject steps pushing a target below this signed z")
    r.add_argument("--min_progress", type=float, default=0.01,
                   help="stop when a round improves the worst scale by less than this")
    r.add_argument("--n_seeds", default="auto",
                   help="'auto' to fit --time_budget, 'all' for every vertex, or an int")
    r.add_argument("--time_budget", type=float, default=60.0,
                   help="wall-clock seconds per case; also sizes --n_seeds auto. "
                        "Pass 0 (or a negative value) to switch the limit off entirely "
                        "and let each case run to completion at full fidelity.")
    r.add_argument("--guard_tolerance", type=float, default=1.0)
    r.add_argument("--max_rounds", type=int, default=30)
    r.add_argument("--max_deviation", type=float, default=0.4)
    r.add_argument("--no_compensate", action="store_true",
                   help="disable shrinkage compensation")
    r.add_argument("--compensate_sigma", default="auto",
                   help="length scale (mm) of the local anti-shrinkage correction. "
                        "'auto' = max(1.5, 2x coarsest target scale); "
                        "0 falls back to a single global offset")
    r.add_argument("--compensate_passes", type=int, default=3,
                   help="iterations of the anti-shrinkage correction per round")
    r.add_argument("--export_edge", type=float, default=DOWNSTREAM_EDGE_MM,
                   help="edge length (mm) for the final remesh of the exported mesh, "
                        "matching the downstream training data; 0 disables it and leaves "
                        "the mesh at the reference resolution")
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--overwrite", action="store_true")
    r.set_defaults(func=cmd_run)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
