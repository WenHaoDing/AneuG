"""
Chamfer Distance Loss
Wraps pytorch3d.loss.chamfer_distance with point-only and point+normal variants.

Used in the GHD fitting loop as the primary shape-matching loss:
  loss_chamfer    — point distance only          (lambda_chamfer)
  loss_chamfer_n1 — point distance + normal term (lambda_chamfer_n1)
"""
import torch
from pytorch3d.loss import chamfer_distance


def chamfer_loss(
    src_pts: torch.Tensor,
    tgt_pts: torch.Tensor,
    src_normals: torch.Tensor | None = None,
    tgt_normals: torch.Tensor | None = None,
    point_reduction: str = "mean",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Bidirectional chamfer distance between two point clouds.

    Parameters
    ----------
    src_pts : (1, N, 3)  source point cloud (deformed canonical)
    tgt_pts : (1, M, 3)  target point cloud (fixed samples from target mesh)
    src_normals : (1, N, 3) optional source normals
    tgt_normals : (1, M, 3) optional target normals
    point_reduction : 'mean' | 'sum'

    Returns
    -------
    loss_points  : scalar — chamfer distance on positions
    loss_normals : scalar — chamfer distance on normals (0 if normals not given)
    """
    loss_pts, loss_nrm = chamfer_distance(
        src_pts,
        tgt_pts,
        x_normals=src_normals,
        y_normals=tgt_normals,
        point_reduction=point_reduction,
    )
    if loss_nrm is None:
        loss_nrm = torch.tensor(0.0, device=src_pts.device)
    return loss_pts, loss_nrm
