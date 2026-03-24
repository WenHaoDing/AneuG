"""
Two-stage alignment for GHD aneurysm fitting.

Stage 1 — anatomy_align()
    Builds anatomical coordinate frames from normalized landmarks
    (dome → +X, vessel → +Z, larger opening → +Z) and maps the
    canonical mesh into the target's normalised space.
    det(R) = +1 guaranteed by Y = Z × X construction.

Stage 2 — cpd_align()
    Rigid Coherent Point Drift registration to correct the small
    residual misalignment left after anatomical pre-alignment.
    Rotation + translation only (scale frozen at 1.0).
"""
from __future__ import annotations

import numpy as np
from probreg import cpd


# ── Frame builder ─────────────────────────────────────────────────────────────

def build_frame(
    neck_n:        np.ndarray,   # (3,) neck position in normalised space
    dome_n:        np.ndarray,   # (3,) dome position in normalised space
    cap_up_n:      np.ndarray,   # (3,) upstream  cap in normalised space
    cap_dn_n:      np.ndarray,   # (3,) downstream cap in normalised space
    z_was_flipped: bool,
) -> np.ndarray:
    """
    Build a 3×3 SO(3) anatomical frame from normalised landmarks.

    Convention
    ----------
    X = neck → dome  (primary axis, independent of vessel curvature)
    Z = vessel direction orthogonalised to X; z_was_flipped ensures +Z = larger opening
    Y = Z × X  →  det(R) = +1 by construction

    Parameters
    ----------
    neck_n, dome_n, cap_up_n, cap_dn_n : (3,) normalised landmark positions
    z_was_flipped : bool  from landmarks['z_was_flipped'] — reverses raw cap
                    direction so +Z consistently points toward the larger opening

    Returns
    -------
    R : (3, 3) rotation matrix, det = +1
    """
    x = dome_n - neck_n
    x = x / (np.linalg.norm(x) + 1e-12)

    raw_z = cap_dn_n - cap_up_n
    if z_was_flipped:
        raw_z = -raw_z          # ensure +Z = larger opening consistently
    raw_z = raw_z - np.dot(raw_z, x) * x
    z = raw_z / (np.linalg.norm(raw_z) + 1e-12)

    y = np.cross(z, x)
    y = y / (np.linalg.norm(y) + 1e-12)

    return np.stack([x, y, z])  # (3, 3), det = +1 by construction


# ── Stage 1: anatomy alignment ────────────────────────────────────────────────

def anatomy_align(
    V_n:      np.ndarray,   # (N, 3) normalised canonical vertices
    lm_can_s: dict,         # normalised canonical landmarks (from normalize_lm)
    lm_tgt_s: dict,         # normalised target    landmarks (from normalize_lm)
) -> np.ndarray:
    """
    Map canonical mesh into target normalised space using anatomical frames.

    Algorithm
    ---------
    1. Build R_can from canonical landmarks  (frame of canonical mesh)
    2. Build R_tgt from target    landmarks  (frame of target    mesh)
    3. R_relative = R_can.T @ R_tgt          (canonical → target frame)
    4. V_anat = (V_n - neck_can_n) @ R_relative + neck_tgt_n

    Parameters
    ----------
    V_n      : (N, 3) normalised canonical vertices (updated in-place copy)
    lm_can_s : normalised canonical landmarks
    lm_tgt_s : normalised target    landmarks

    Returns
    -------
    V_n_anat : (N, 3) float32  canonical vertices in target normalised space
    """
    R_can = build_frame(
        lm_can_s["neck"].astype(np.float64),
        lm_can_s["dome"].astype(np.float64),
        lm_can_s["cap_up"].astype(np.float64),
        lm_can_s["cap_down"].astype(np.float64),
        bool(lm_can_s["z_was_flipped"]),
    )
    R_tgt = build_frame(
        lm_tgt_s["neck"].astype(np.float64),
        lm_tgt_s["dome"].astype(np.float64),
        lm_tgt_s["cap_up"].astype(np.float64),
        lm_tgt_s["cap_down"].astype(np.float64),
        bool(lm_tgt_s["z_was_flipped"]),
    )

    det_can = np.linalg.det(R_can)
    det_tgt = np.linalg.det(R_tgt)
    assert abs(det_can - 1.0) < 1e-4, f"R_can not SO(3): det={det_can:.6f}"
    assert abs(det_tgt - 1.0) < 1e-4, f"R_tgt not SO(3): det={det_tgt:.6f}"

    neck_can_n = lm_can_s["neck"].astype(np.float64)
    neck_tgt_n = lm_tgt_s["neck"].astype(np.float64)
    R_relative = R_can.T @ R_tgt

    V_n_anat = (V_n.astype(np.float64) - neck_can_n) @ R_relative + neck_tgt_n
    return V_n_anat.astype(np.float32)


# ── Stage 2: CPD refinement ───────────────────────────────────────────────────

def cpd_align(
    V_base: np.ndarray,   # (N, 3) anatomy-aligned canonical (from anatomy_align)
    V_tgt:  np.ndarray,   # (K, 3) normalised target vertices
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rigid CPD registration: corrects residual misalignment after anatomy_align.

    Rotation + translation only (scale frozen at 1.0) to preserve the
    normalised-space scale established in STEP 1b.

    Parameters
    ----------
    V_base : (N, 3) source point cloud (anatomy-aligned canonical)
    V_tgt  : (K, 3) target point cloud

    Returns
    -------
    V_aligned : (N, 3) float64  CPD-aligned canonical vertices
    R_cpd     : (3, 3) float64  rotation matrix
    t_cpd     : (3,)   float64  translation vector
    """
    V_base_f64 = V_base.astype(np.float64)
    V_tgt_f64  = V_tgt.astype(np.float64)

    init_param = {
        "rot":   np.eye(3, dtype=np.float64),
        "scale": 1.0,
        "t":     np.zeros(3, dtype=np.float64),
    }
    rgd_cpd = cpd.RigidCPD(V_base_f64, tf_init_params=init_param, update_scale=False)
    tf_param, _, _ = rgd_cpd.registration(V_tgt_f64)

    R_cpd     = tf_param.rot.astype(np.float64)
    t_cpd     = tf_param.t.astype(np.float64)
    V_aligned = V_base_f64 @ R_cpd.T + t_cpd

    return V_aligned, R_cpd, t_cpd
