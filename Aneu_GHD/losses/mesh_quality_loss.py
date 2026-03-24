"""
Mesh Quality Losses
Ported from GHDHeart/losses/mesh_loss.py and GHDHeart/GHD/GHD_cardiac.py

  laplacian_loss          <- Loss_Laplacian  (mesh_laplacian_smoothing, cot)
  normal_consistency_loss <- Loss_normal_consistency (mesh_normal_consistency)
  EdgeLengthLoss          <- Mesh_loss (mesh_edge_loss with fixed ref length)
"""
import torch
import torch.nn as nn
from pytorch3d.structures import Meshes
from pytorch3d.loss import (
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
    mesh_edge_loss,
)


def laplacian_loss(mesh: Meshes, method: str = "cot") -> torch.Tensor:
    """
    Laplacian smoothness loss.

    Penalises each vertex for being far from the mean of its neighbours.
    method='cot' uses cotangent weights (geometry-aware, better at preventing
    pinching than 'uniform').

    Parameters
    ----------
    mesh   : pytorch3d Meshes
    method : 'cot' | 'uniform' | 'cotcurv'

    GHDHeart meshlossclass.py uses 'cot'; GHD_cardiac.py uses 'uniform'.
    We default to 'cot' to fix pinched zones.
    """
    return mesh_laplacian_smoothing(mesh, method=method)


def normal_consistency_loss(mesh: Meshes) -> torch.Tensor:
    """
    Penalises large dihedral angles between adjacent faces.

    Prevents the surface from flipping normals between neighbours,
    which indicates mesh self-intersection or folding.

    GHDHeart: Loss_normal_consistency via mesh_normal_consistency.
    """
    return mesh_normal_consistency(mesh)


class EdgeLengthLoss(nn.Module):
    """
    Edge length loss anchored to the source / template mesh.

    Computes the mean edge length of the source mesh once at init, then
    penalises edges of the deforming mesh for deviating from that reference.
    This prevents edges from collapsing or stretching too far.

    Parameters
    ----------
    mesh_src : pytorch3d Meshes — source / template mesh (used for reference length only)

    GHDHeart: Mesh_loss.__init__ computes self.edge_lenth from mesh_std, then
    forward calls mesh_edge_loss(meshes_scr, self.edge_lenth).
    Source: GHDHeart/losses/meshlossclass.py:40, 82
    """

    def __init__(self, mesh_src: Meshes) -> None:
        super().__init__()
        edge_idx = mesh_src.edges_packed()
        verts    = mesh_src.verts_packed()
        ref_len  = torch.norm(
            verts[edge_idx[:, 0]] - verts[edge_idx[:, 1]], dim=1
        ).mean()
        self.register_buffer("ref_length", ref_len)

    def forward(self, mesh: Meshes) -> torch.Tensor:
        return mesh_edge_loss(mesh, self.ref_length.to(mesh.device))
