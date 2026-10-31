"""Shape check of a merged aneurysm sac: does it look like a dome? Metrics (scale-free, on an
isotropic binary mask) compared with real sacs (dataset/sac_shape_reference.json, made by
scripts/ostium/sac_shape.py). Only extreme shapes are flagged: each limit is a margin past the real
1st / 99th percentile (LIMITS).
  volume_mm3   too small to be a sac (slivers)
  elongation   sqrt(l1 / l3), principal-axis eigenvalues of the voxel coordinates (1 = round)
  flatness     sqrt(l2 / l3)
  thickness    inscribed-sphere diameter / length along the first principal axis (small = stripe)
  solidity     volume / convex-hull volume (small = crescent, smear)
"""
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

REFERENCE = Path(__file__).resolve().parents[1] / "dataset" / "sac_shape_reference.json"
VOX = 0.35
FACE = ndi.generate_binary_structure(3, 1)
# (metric, side, real percentile, factor): flag when past factor x that percentile of real sacs
# solidity gets a wider margin: merged sacs sit systematically lower than real ones (median 0.62 vs
# 0.73), a difference of method, not of weirdness. volume_mm3 catches slivers too small to measure.
LIMITS = [("volume_mm3", "low", 1, 0.25), ("elongation", "high", 99, 1.25), ("flatness", "high", 99, 1.25),
          ("thickness", "low", 1, 0.75), ("solidity", "low", 1, 0.6)]


def shape_metrics(m, vox=VOX):
    """Metrics of one binary piece m (isotropic voxels of size vox mm)."""
    from scipy.spatial import ConvexHull
    zyx = np.argwhere(m) * vox
    n = len(zyx)
    out = dict(volume_mm3=n * vox ** 3)
    if n < 8:
        return out
    lam = np.sort(np.linalg.eigvalsh(np.cov(zyx.T)))[::-1]
    lam = np.maximum(lam, (vox ** 2) / 12)                                 # one voxel's own spread
    ax = np.linalg.eigh(np.cov(zyx.T))[1][:, -1]
    proj = zyx @ ax
    L = float(proj.max() - proj.min()) + vox
    edt = ndi.distance_transform_edt(np.pad(m, 1)) * vox
    out.update(elongation=float(np.sqrt(lam[0] / lam[2])), flatness=float(np.sqrt(lam[1] / lam[2])),
               thickness=float(2 * edt.max() / L), length_mm=L)
    try:
        corners = (zyx[:, None] + vox * (np.array(np.meshgrid([0, 1], [0, 1], [0, 1], indexing="ij")).reshape(3, -1).T - 0.5)[None]).reshape(-1, 3)
        out["solidity"] = float(n * vox ** 3 / ConvexHull(corners).volume)
    except Exception:
        pass
    return out


def limits(ref):
    return {m: (side, ref["quantiles"][m][str(p)] * f) for m, side, p, f in LIMITS}


_LIM = None


def check(sac, vox):
    """Metrics of the largest face-connected piece of a sac mask (isotropic voxels of vox mm) and
    the metrics past their limits."""
    global _LIM
    if _LIM is None:
        _LIM = limits(json.loads(REFERENCE.read_text()))
    lab, n = ndi.label(sac, structure=FACE)
    main = lab == (np.bincount(lab.ravel())[1:].argmax() + 1) if n else sac
    r = shape_metrics(main, vox)
    past = [m for m, (side, v) in _LIM.items() if m in r and (r[m] > v if side == "high" else r[m] < v)]
    return r, past
