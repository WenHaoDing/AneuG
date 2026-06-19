"""
Ring Chamfer Loss

Matches canonical cap ring vertices (after GHD deformation) to the target
ring point cloud using bidirectional Chamfer distance.

Handles position + orientation + in-plane distribution in one loss,
replacing both LandmarkLoss (cap centroid) and RingPlanarityLoss.


Usage
-----
# Build ring_pairs — list of (can_ring_idxs, tgt_ring_pts) tuples.
# Works for any number of openings (2 for sidewall, 3 for bifurcation, etc.)

# Sidewall (2 rings):
ring_pairs = [
    (can_ring_up_idxs, V_tgt_n[lm_tgt['ring_up_idxs']]),
    (can_ring_dn_idxs, V_tgt_n[lm_tgt['ring_dn_idxs']]),
]

# Bifurcation (3 rings):
ring_pairs = [
    (can_opa['op_v_indices'][0], V_tgt_n[tgt_opa['op_v_indices'][0]]),
    (can_opa['op_v_indices'][1], V_tgt_n[tgt_opa['op_v_indices'][1]]),
    (can_opa['op_v_indices'][2], V_tgt_n[tgt_opa['op_v_indices'][2]]),
]

# Build loss:
loss_fn = RingChamferLoss(ring_pairs).to(device)

# Each iteration:
loss = loss_fn(V_rendered)
"""
import torch
import torch.nn as nn
import numpy as np
from pytorch3d.loss import chamfer_distance


class RingChamferLoss(nn.Module):
    """
    Bidirectional Chamfer distance between canonical ring vertices and
    target ring point clouds, averaged over all rings.

    Parameters
    ----------
    ring_pairs : list of (can_ring_idxs, tgt_ring_pts) tuples
        can_ring_idxs : array-like of int   canonical ring vertex indices
        tgt_ring_pts  : (n, 3) np.ndarray   target ring points (normalised space)

    Supports any number of rings — 2 for sidewall, 3 for bifurcation.
    """

    def __init__(self, ring_pairs: list) -> None:
        super().__init__()
        self.n_rings = len(ring_pairs)
        for i, (idxs, pts) in enumerate(ring_pairs):
            self.register_buffer(f"idxs_{i}", torch.as_tensor(idxs, dtype=torch.long))
            self.register_buffer(f"pts_{i}",  torch.as_tensor(pts,  dtype=torch.float32))

    def forward(self, verts: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        verts : (N, 3) deformed canonical vertices (V_rendered)

        Returns
        -------
        Scalar: mean chamfer distance across all rings.
        """
        total = verts.new_zeros(1).squeeze()
        for i in range(self.n_rings):
            src = verts[getattr(self, f"idxs_{i}")].unsqueeze(0)   # (1, k_i, 3)
            tgt = getattr(self, f"pts_{i}").unsqueeze(0)            # (1, n_i, 3)
            total = total + chamfer_distance(src, tgt)[0]
        return total / self.n_rings
