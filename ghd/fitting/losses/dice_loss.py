"""
Binary Dice Loss
Ported from GHDHeart/losses/dice_loss.py
"""
import torch
import torch.nn as nn


class BinaryDiceLoss(nn.Module):
    """
    Binary Dice loss for occupancy predictions.

    Dice = 1 - (2 * |P ∩ T| + smooth) / (|P| + |T| + smooth)

    Parameters
    ----------
    smooth : float
        Laplace smoothing constant to avoid division by zero. Default 1.0.

    GHDHeart: losses/dice_loss.py::BinaryDiceLoss
    """

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        pred:   torch.Tensor,                   # (N, *) predicted occupancy
        target: torch.Tensor,                   # (N, *) ground-truth labels
        weight: torch.Tensor | None = None,     # (N,)   optional per-sample weights
    ) -> torch.Tensor:
        N = target.shape[0]
        p = pred.reshape(N, -1)
        t = target.reshape(N, -1)

        inter = (p * t).sum(dim=1)
        denom = p.sum(dim=1) + t.sum(dim=1) + self.smooth + 1e-6
        loss  = 1.0 - (2.0 * inter + self.smooth) / denom

        return (loss * weight).mean() if weight is not None else loss.mean()
