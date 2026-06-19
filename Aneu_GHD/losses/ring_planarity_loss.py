"""
Ring Planarity Loss

Forces the canonical cap ring vertices (after GHD deformation) to lie in the
plane defined by the corresponding target ring vertices.

The target ring plane (normal + centroid) is pre-computed once from the target
mesh and passed as fixed buffers — no SVD in the forward pass.

Usage
-----
# Pre-compute target ring planes (in normalised space):
ring_up_pts = V_tgt_n[lm_tgt['ring_up_idxs']]
ring_dn_pts = V_tgt_n[lm_tgt['ring_dn_idxs']]

def fit_plane(pts):
    c = pts.mean(0)
    _, _, Vt = np.linalg.svd(pts - c)
    return Vt[2], c   # (normal, centroid)

n_up, c_up = fit_plane(ring_up_pts)
n_dn, c_dn = fit_plane(ring_dn_pts)

# Build loss:
loss_fn = RingPlanarityLoss(cap_up_idxs, cap_dn_idxs,
                             n_up, c_up, n_dn, c_dn).to(device)

# Each iteration:
loss = loss_fn(V_rendered)
"""
import torch
import torch.nn as nn
import numpy as np


class RingPlanarityLoss(nn.Module):
    """
    Penalises out-of-plane deviation of canonical ring vertices from the
    target ring's best-fit plane.

    For each cap:
        d_i = dot(v_i - centroid_tgt, normal_tgt)   # signed distance
        loss = mean(d_i²)

    Parameters
    ----------
    cap_up_idxs : array-like (k_up,)   canonical upstream  ring vertex indices
    cap_dn_idxs : array-like (k_dn,)   canonical downstream ring vertex indices
    n_up : (3,)   unit normal of target upstream  ring plane
    c_up : (3,)   centroid of target upstream  ring
    n_dn : (3,)   unit normal of target downstream ring plane
    c_dn : (3,)   centroid of target downstream ring
    """

    def __init__(
        self,
        cap_up_idxs,
        cap_dn_idxs,
        n_up: np.ndarray,
        c_up: np.ndarray,
        n_dn: np.ndarray,
        c_dn: np.ndarray,
    ) -> None:
        super().__init__()
        self.register_buffer("cap_up_idxs", torch.as_tensor(cap_up_idxs, dtype=torch.long))
        self.register_buffer("cap_dn_idxs", torch.as_tensor(cap_dn_idxs, dtype=torch.long))
        self.register_buffer("n_up", torch.as_tensor(n_up, dtype=torch.float32))
        self.register_buffer("c_up", torch.as_tensor(c_up, dtype=torch.float32))
        self.register_buffer("n_dn", torch.as_tensor(n_dn, dtype=torch.float32))
        self.register_buffer("c_dn", torch.as_tensor(c_dn, dtype=torch.float32))

    def forward(self, verts: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        verts : (N, 3) deformed canonical vertices (V_rendered)

        Returns
        -------
        Scalar loss: mean squared out-of-plane distance across both rings.
        """
        d_up = ((verts[self.cap_up_idxs] - self.c_up) * self.n_up).sum(dim=1)
        d_dn = ((verts[self.cap_dn_idxs] - self.c_dn) * self.n_dn).sum(dim=1)
        return (d_up.pow(2).mean() + d_dn.pow(2).mean()) / 2.0
