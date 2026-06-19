"""
CPD alignment for GHD aneurysm fitting.

cpd_align()
    Rigid Coherent Point Drift registration.
    Both meshes are already in canonical anatomical frame (dome→+X, vessel→+Z)
    from the cutting pipeline — CPD corrects residual misalignment directly.
    Rotation + translation only (scale frozen at 1.0).

"""
from __future__ import annotations

import numpy as np
from probreg import cpd


def cpd_align(
    V_base: np.ndarray,   # (N, 3) canonical vertices in normalised space
    V_tgt:  np.ndarray,   # (K, 3) normalised target vertices
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rigid CPD registration.

    Parameters
    ----------
    V_base : (N, 3) source point cloud (canonical, normalised)
    V_tgt  : (K, 3) target point cloud (normalised)

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