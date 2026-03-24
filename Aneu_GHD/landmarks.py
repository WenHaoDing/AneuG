"""
Landmark utilities for GHD aneurysm fitting.

Three entry points:
  load_landmarks(landmarks_dir, case_id)
      Load a pre-computed landmarks.npz produced by the cutting pipeline.

  normalize_lm(lm, centroid, scale)
      Map landmark positions into the same normalised space as V_n / V_tgt_n.

  compute_landmarks_from_rings(mesh, inlet_ring_nodes, outlet_ring_nodes, dome_pts)
      Compute landmarks from scratch given ring nodes and a segmented dome.
      No centerline required.
"""
from __future__ import annotations

import numpy as np


# ── Load / normalize ──────────────────────────────────────────────────────────

def load_landmarks(landmarks_dir: str, case_id: str) -> dict:
    """
    Load landmarks.npz produced by the cutting pipeline.

    Parameters
    ----------
    landmarks_dir : str   root directory containing per-case subdirectories
    case_id       : str   subdirectory name (e.g. 'GHD2/Checkpoints/Alignment/C0002')

    Returns
    -------
    dict keyed by the npz field names:
        neck, dome, cap_up, cap_down  — (3,) aligned-frame positions
        R_frame                       — (3, 3) rotation matrix
        neck_pt_world                 — (3,) world-space neck point
        r_up_mm, r_down_mm            — scalar cap radii (mm)
        z_was_flipped                 — bool
    """
    path = f"{landmarks_dir}/{case_id}/landmarks.npz"
    d    = np.load(path)
    return {k: d[k] for k in d.files}


def normalize_lm(lm: dict, centroid: np.ndarray, scale: float) -> dict:
    """
    Map landmark positions into normalised mesh space.

    Applies the same centroid-subtraction + max-norm scaling used to produce
    V_n and V_tgt_n, so landmarks are directly comparable to mesh vertices.

    Parameters
    ----------
    lm       : dict  raw landmarks from load_landmarks()
    centroid : (3,)  mesh centroid  (c_can or c_tgt)
    scale    : float mesh max-norm  (s_can or s_tgt)

    Returns
    -------
    dict with (3,) landmarks normalised; scalars and matrices unchanged.
    """
    out = {}
    for k, v in lm.items():
        arr = np.asarray(v)
        out[k] = (arr - centroid) / scale if arr.shape == (3,) else arr
    return out


# ── Compute from rings ────────────────────────────────────────────────────────

def compute_landmarks_from_rings(
    inlet_ring_nodes:  np.ndarray,
    outlet_ring_nodes: np.ndarray,
    dome_pts:          np.ndarray,
    radius_eq_tol:     float = 0.10,
) -> dict:
    """
    Compute a landmarks dict compatible with load_landmarks() / normalize_lm()
    from inputs provided by a generic cutting pipeline.

    No centerline required — dome centroid is computed directly from the
    segmented dome point cloud.

    Parameters
    ----------
    inlet_ring_nodes  : (N, 3)  ring nodes of the inlet  opening (world coords, mm)
    outlet_ring_nodes : (M, 3)  ring nodes of the outlet opening (world coords, mm)
    dome_pts          : (K, 3)  points of the segmented dome region (world coords, mm)
    radius_eq_tol     : float   radius equality tolerance for Z-flip decision (mm)

    Returns
    -------
    dict with same layout as load_landmarks(), ready for normalize_lm().
    """
    inlet_ring_nodes  = np.asarray(inlet_ring_nodes,  dtype=np.float64)
    outlet_ring_nodes = np.asarray(outlet_ring_nodes, dtype=np.float64)
    dome_pts          = np.asarray(dome_pts,          dtype=np.float64)

    # ── Cap centroids and radii ───────────────────────────────────────────────
    P_up  = inlet_ring_nodes.mean(axis=0)
    P_dn  = outlet_ring_nodes.mean(axis=0)
    r_up  = float(np.linalg.norm(inlet_ring_nodes  - P_up, axis=1).mean())
    r_dn  = float(np.linalg.norm(outlet_ring_nodes - P_dn, axis=1).mean())

    # ── Dome centroid (exact — from segmented dome region) ────────────────────
    dome_center = dome_pts.mean(axis=0)

    # ── Neck: projection of dome centroid onto the vessel axis ────────────────
    vessel_axis = P_dn - P_up
    vessel_len  = np.linalg.norm(vessel_axis) + 1e-12
    vessel_dir  = vessel_axis / vessel_len
    t           = np.clip(np.dot(dome_center - P_up, vessel_dir), 0.0, vessel_len)
    neck_pt     = P_up + t * vessel_dir

    # ── Build R_frame ─────────────────────────────────────────────────────────
    # Z = vessel direction (inlet → outlet)
    # Y = dome direction orthogonalised to Z (always points toward dome)
    # X = Y × Z
    # Y = Z × X  (reorthogonalise to ensure exact orthogonality)
    # det(R) = +1 by construction
    z = vessel_dir.copy()

    raw_y = dome_center - neck_pt
    raw_y = raw_y - np.dot(raw_y, z) * z
    if np.linalg.norm(raw_y) < 1e-6:           # degenerate fallback
        raw_y = np.array([0.0, 0.0, 1.0])
        raw_y = raw_y - np.dot(raw_y, z) * z
    y = raw_y / (np.linalg.norm(raw_y) + 1e-12)
    if np.dot(y, dome_center - neck_pt) < 0:   # always point toward dome
        y = -y

    x = np.cross(y, z);  x = x / (np.linalg.norm(x) + 1e-12)
    y = np.cross(z, x);  y = y / (np.linalg.norm(y) + 1e-12)   # reorthogonalise
    R = np.stack([x, y, z])                                      # det = +1

    # ── Z-flip: enforce larger opening → +Z ──────────────────────────────────
    if abs(r_up - r_dn) < radius_eq_tol:
        z_was_flipped = bool(np.dot(np.array([0.0, 0.0, 1.0]), z) < 0)
    else:
        z_was_flipped = bool(r_up > r_dn)      # larger opening → +Z

    if z_was_flipped:
        z = -z
        y = np.cross(z, x);  y = y / (np.linalg.norm(y) + 1e-12)
        R = np.stack([x, y, z])

    assert abs(np.linalg.det(R) - 1.0) < 1e-5, "R_frame is not SO(3)"

    # ── Pack into aligned frame (same layout as cutting pipeline npz) ─────────
    def _to_aligned(p_world: np.ndarray) -> np.ndarray:
        return (p_world - neck_pt) @ R.T

    return {
        "neck"          : np.zeros(3),
        "dome"          : _to_aligned(dome_center),
        "cap_up"        : _to_aligned(P_up),
        "cap_down"      : _to_aligned(P_dn),
        "R_frame"       : R,
        "neck_pt_world" : neck_pt,
        "r_up_mm"       : np.array(r_up),
        "r_down_mm"     : np.array(r_dn),
        "z_was_flipped" : np.array(z_was_flipped),
    }
