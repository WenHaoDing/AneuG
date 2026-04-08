"""
Centroid Loss

L2 distance between the mean position of canonical ring vertices (after GHD
deformation) and the mean position of the corresponding target ring vertices.

This is the gentlest vessel-direction constraint:
  gradient = (mean_canonical - mean_target) / N
  — spread equally over all N ring vertices, no ring-shape distortion.

Works for any number of rings (2 for sidewall, 3 for bifurcation).

Usage
-----
# Build centroid_pairs — list of (can_ring_idxs, tgt_centroid) tuples.
# tgt_centroid = mean of target ring vertex positions = V_tgt_n[ring_idxs].mean(axis=0)

centroid_pairs = [
    (can_ring_up_idxs,  V_tgt_n[ring_up_idxs].mean(axis=0)),   # shape (3,)
    (can_ring_dn_idxs,  V_tgt_n[ring_dn_idxs].mean(axis=0)),
]

loss_fn = CentroidLoss(centroid_pairs).to(device)

# Each iteration:
loss = loss_fn(V_rendered)
"""
import torch
import torch.nn as nn


class CentroidLoss(nn.Module):
    """
    L2 centroid-to-centroid loss for vessel-direction alignment.

    Parameters
    ----------
    centroid_pairs : list of (can_ring_idxs, tgt_centroid) tuples
        can_ring_idxs : array-like of int   canonical ring vertex indices
        tgt_centroid  : (3,) array-like     mean of target ring vertex positions

    Averaged over all rings.
    """

    def __init__(self, centroid_pairs: list) -> None:
        super().__init__()
        self.n_rings = len(centroid_pairs)
        for i, (idxs, tgt_c) in enumerate(centroid_pairs):
            self.register_buffer(
                f"idxs_{i}",
                torch.as_tensor(idxs,  dtype=torch.long),
            )
            self.register_buffer(
                f"tgt_{i}",
                torch.as_tensor(tgt_c, dtype=torch.float32),
            )

    def forward(self, verts: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        verts : (N, 3) deformed canonical vertices (V_rendered)

        Returns
        -------
        Scalar: mean L2 centroid distance across all rings.
        """
        total = verts.new_zeros(1).squeeze()
        for i in range(self.n_rings):
            can_mean = verts[getattr(self, f"idxs_{i}")].mean(dim=0)
            tgt_c    = getattr(self, f"tgt_{i}")
            diff     = can_mean - tgt_c
            total    = total + (diff * diff).mean()
        return total / self.n_rings
