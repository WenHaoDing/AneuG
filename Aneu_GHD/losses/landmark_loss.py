"""
Anatomical Landmark Loss

Pulls deformed canonical vertices toward anatomical target positions:
  - dome   : single vertex nearest the dome centroid (optional — pass None to disable)
  - cap_up : k-nearest vertices to upstream  cap, penalises mean position
  - cap_dn : k-nearest vertices to downstream cap, penalises mean position

All positions must be in the same normalised mesh space as V_n / V_tgt_n.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LandmarkLoss(nn.Module):
    """
    Anatomical landmark loss for GHD fitting.

    Parameters
    ----------
    dome_idx : int or None
        Index of the canonical vertex closest to the dome landmark.
        Computed as: int(np.argmin(np.linalg.norm(V_n - lm_can_s['dome'], axis=1)))
        Pass None to use vessel-only mode (cap_up + cap_dn only).
        Automatically None when lm_tgt['dome_valid'] is False.
    cap_up_idxs : array-like of int, shape (k,)
        Indices of the k canonical vertices closest to the upstream cap.
    cap_dn_idxs : array-like of int, shape (k,)
        Indices of the k canonical vertices closest to the downstream cap.

    Usage
    -----
    # Full (dome + caps):
    loss_fn = LandmarkLoss(dome_idx, cap_up_idxs, cap_dn_idxs).to(device)
    loss = loss_fn(V_deformed, lm_tgt_dome_t, lm_tgt_cap_up_t, lm_tgt_cap_dn_t)

    # Vessel-only (caps only, e.g. when dome_valid=False in landmarks.npz):
    loss_fn = LandmarkLoss(None, cap_up_idxs, cap_dn_idxs).to(device)
    loss = loss_fn(V_deformed, None, lm_tgt_cap_up_t, lm_tgt_cap_dn_t)
    """

    def __init__(
        self,
        dome_idx:    int | None,
        cap_up_idxs: torch.Tensor | list,
        cap_dn_idxs: torch.Tensor | list,
    ) -> None:
        super().__init__()
        self.dome_idx = int(dome_idx) if dome_idx is not None else None
        self.register_buffer(
            "cap_up_idxs",
            torch.as_tensor(cap_up_idxs, dtype=torch.long),
        )
        self.register_buffer(
            "cap_dn_idxs",
            torch.as_tensor(cap_dn_idxs, dtype=torch.long),
        )

    def forward(
        self,
        verts:      torch.Tensor,           # (N, 3) deformed canonical vertices
        dome_tgt:   torch.Tensor | None,    # (3,)   target dome position, or None
        cap_up_tgt: torch.Tensor,           # (3,)   target upstream  cap centroid
        cap_dn_tgt: torch.Tensor,           # (3,)   target downstream cap centroid
    ) -> torch.Tensor:
        loss_cap_up = F.mse_loss(verts[self.cap_up_idxs].mean(dim=0), cap_up_tgt)
        loss_cap_dn = F.mse_loss(verts[self.cap_dn_idxs].mean(dim=0), cap_dn_tgt)

        if self.dome_idx is not None and dome_tgt is not None:
            loss_dome = F.mse_loss(verts[self.dome_idx], dome_tgt)
            return (loss_dome + loss_cap_up + loss_cap_dn) / 3.0

        return (loss_cap_up + loss_cap_dn) / 2.0
