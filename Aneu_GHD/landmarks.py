"""
Landmark utilities for GHD aneurysm fitting.

Four entry points:
  load_landmarks(landmarks_dir, case_id)
      Load a pre-computed landmarks.npz produced by the cutting pipeline.

  normalize_lm(lm, centroid, scale)
      Map landmark positions into the same normalised space as V_n / V_tgt_n.

  compute_landmarks_from_rings(inlet_ring_nodes, outlet_ring_nodes, dome_pts)
      Sidewall: compute landmarks from 2 ring nodes + dome centroid.
      Builds a full anatomical frame (neck at origin, X=dome, Z=inlet→outlet).
      Caller must pass the upstream ring as inlet_ring_nodes (from branch_ranking).

  compute_landmarks_from_rings_bifurcation(bif_pt, inlet_ring_nodes, outlet_ring_nodes_list)
      Bifurcation: compute landmarks from the bifurcation point + 3 ring nodes.
      Translation-only (bif_pt → origin); rotation computed via Procrustes in prep_case_for_ghd.py.
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
        R_frame                       — (3, 3) rotation matrix (SO(3), det=+1)
        neck_pt_world                 — (3,) world-space neck point
        r_up_mm, r_down_mm            — scalar cap radii (mm)
        dome_valid                    — bool, True when dome centroid is from dome mesh
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
) -> dict:
    """
    Compute a landmarks dict compatible with load_landmarks() / normalize_lm()
    from inputs provided by a generic cutting pipeline.

    No centerline required — dome centroid is computed directly from the
    segmented dome point cloud.

    Z axis is defined as inlet → outlet (caller must pass the upstream ring as
    inlet_ring_nodes, identified via branch_ranking rank-0). No Z-flip is applied.

    Parameters
    ----------
    inlet_ring_nodes  : (N, 3)  ring nodes of the inlet  opening (world coords, mm)
    outlet_ring_nodes : (M, 3)  ring nodes of the outlet opening (world coords, mm)
    dome_pts          : (K, 3)  points of the segmented dome region (world coords, mm)

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
    # Matches process_sidewall_batch.py convention exactly:
    #   X = dome direction (neck-plane normal toward dome)
    #   Z = vessel direction (inlet → outlet)
    #   Y = Z × X  (completes right-hand frame)
    # det(R) = +1 by construction
    z = vessel_dir.copy()

    raw_x = dome_center - neck_pt
    raw_x = raw_x - np.dot(raw_x, z) * z       # orthogonalise to Z
    if np.linalg.norm(raw_x) < 1e-6:           # degenerate fallback
        raw_x = np.array([0.0, 0.0, 1.0])
        raw_x = raw_x - np.dot(raw_x, z) * z
    x = raw_x / (np.linalg.norm(raw_x) + 1e-12)
    if np.dot(x, dome_center - neck_pt) < 0:   # always point toward dome
        x = -x

    y = np.cross(z, x);  y = y / (np.linalg.norm(y) + 1e-12)
    R = np.stack([x, y, z])                    # det = +1

    # ── Enforce SO(3): det(R) must be exactly +1 ─────────────────────────────
    # X, Y, Z are built from cross products so det=+1 by construction.
    # This assertion catches any degenerate fallback or numerical drift.
    assert abs(np.linalg.det(R) - 1.0) < 1e-5, "R_frame is not SO(3) — frame construction error"

    # ── Pack into aligned frame (same layout as cutting pipeline npz) ─────────
    def _to_aligned(p_world: np.ndarray) -> np.ndarray:
        return (p_world - neck_pt) @ R.T

    return {
        "neck"          : np.zeros(3),
        "dome"          : _to_aligned(dome_center),
        "cap_up"        : P_up,              # world coords — correct for normalize_lm
        "cap_down"      : P_dn,              # world coords — correct for normalize_lm
        "R_frame"       : R,
        "neck_pt_world" : neck_pt,
        "r_up_mm"       : np.array(r_up),
        "r_down_mm"     : np.array(r_dn),
        "dome_valid"    : np.array(True),    # dome centroid from endpoints_manual aneurysm_centroid
    }


def compute_landmarks_from_rings_bifurcation(
    bif_pt:               np.ndarray,   # (3,) bifurcation point in world coords (mm)
                                        #      e.g. merged_centerline['nearest_bif']['centroid']
    inlet_ring_nodes:     np.ndarray,   # (N, 3) inlet  opening ring (world coords, mm)
    outlet_ring_nodes_list: list,       # [(M,3), (K,3)] outlet opening rings (world coords, mm)
) -> dict:
    """
    Compute a landmarks dict for a bifurcation aneurysm.

    Alignment convention:
      - Translate so bifurcation point is at origin (no rotation applied here)
      - Rotation is computed in prep_case_for_ghd.py via Procrustes on ring centroids

    Parameters
    ----------
    bif_pt                 : (3,)       bifurcation point in world coords
                                        use merged_centerline['nearest_bif']['centroid']
    inlet_ring_nodes       : (N, 3)     inlet opening ring nodes (world coords, mm)
    outlet_ring_nodes_list : list of (M, 3) arrays, one per outlet opening

    Returns
    -------
    dict compatible with load_landmarks() layout:
        neck_pt_world  — bifurcation point (world coords)
        r_up_mm        — inlet ring mean radius
        r_down_mm      — first outlet ring mean radius
        aneu_type      — 'bifurcation' (marker for downstream code)
        ring indices are NOT stored here — added by prep_case_for_ghd.py after NN search
    """
    bif_pt             = np.asarray(bif_pt,           dtype=np.float64)
    inlet_ring_nodes   = np.asarray(inlet_ring_nodes, dtype=np.float64)
    outlet_ring_nodes_list = [np.asarray(r, dtype=np.float64) for r in outlet_ring_nodes_list]

    # ── Cap centroids and radii ───────────────────────────────────────────────
    P_in  = inlet_ring_nodes.mean(axis=0)
    r_in  = float(np.linalg.norm(inlet_ring_nodes - P_in, axis=1).mean())

    P_out0 = outlet_ring_nodes_list[0].mean(axis=0)
    r_out0 = float(np.linalg.norm(outlet_ring_nodes_list[0] - P_out0, axis=1).mean())

    return {
        "neck"          : np.zeros(3),
        "neck_pt_world" : bif_pt,
        "r_up_mm"       : np.array(r_in),
        "r_down_mm"     : np.array(r_out0),
        "z_was_flipped" : np.array(False),
        "aneu_type"     : np.array("bifurcation"),
    }
