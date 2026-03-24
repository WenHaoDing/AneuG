"""
DVS Mesh-Occupancy Loss
Ported from GHDHeart/ops/mesh_geometry.py  (Winding_Occupancy, dual_area_*)
             GHDHeart/GHD/GHD_cardiac.py    (Loss_occupancy block)
             GHDHeart/losses/dice_loss.py   (BinaryDiceLoss)
"""
import numpy as np
import torch
import torch.nn as nn
from pytorch3d.structures import Meshes

from .dice_loss import BinaryDiceLoss


# ── Winding-number helpers ────────────────────────────────────────────────────

def _dual_area_weights_faces(mesh: Meshes) -> torch.Tensor:
    """
    Dual area weight (F, 3) for each vertex of each triangle.
    GHDHeart: ops/mesh_geometry.py::dual_area_weights_faces
    """
    verts = mesh.verts_packed()
    faces = mesh.faces_packed()
    fc    = verts[faces]                  # (F, 3, 3)
    A     = fc[:, 1] - fc[:, 0]
    B     = fc[:, 2] - fc[:, 1]
    C     = fc[:, 0] - fc[:, 2]

    def _angle(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        cos = (-u * v).sum(-1) / (u.norm(dim=-1) * v.norm(dim=-1) + 1e-8)
        return torch.arccos(cos.clamp(-1 + 1e-6, 1 - 1e-6))

    angles = torch.stack([_angle(C, A), _angle(A, B), _angle(B, C)], dim=1)
    sin2   = torch.sin(2 * angles)
    w      = sin2 / (sin2.sum(-1, keepdim=True) + 1e-8)
    return (w[:, [2, 0, 1]] + w[:, [1, 2, 0]]) / 2    # (F, 3)


def _dual_area_vertex(mesh: Meshes) -> torch.Tensor:
    """
    Per-vertex dual area (N, 1).
    GHDHeart: ops/mesh_geometry.py::dual_area_vertex
    """
    w   = _dual_area_weights_faces(mesh)
    da  = (w * mesh.faces_areas_packed().view(-1, 1)).view(-1)
    vix = mesh.faces_packed().view(-1)
    N   = mesh.verts_packed().shape[0]
    return torch.zeros(N, device=mesh.device).scatter_add(0, vix, da).view(-1, 1)


def _electric_strength(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """
    Coulomb-like field: q (M, 3) query points, p (N, 3) sources -> (M, N, 3).
    GHDHeart: ops/mesh_geometry.py::Electric_strength
    """
    r     = p.unsqueeze(0) - q.unsqueeze(1)
    dist3 = r.norm(dim=-1, keepdim=True).pow(3) + 1e-8
    return r / dist3


def winding_occupancy(
    mesh:           Meshes,
    points:         torch.Tensor,
    max_v_per_call: int = 2000,
) -> torch.Tensor:
    """
    Differentiable winding-number occupancy in [0, 1].
    Values ≈ 1 → inside the mesh surface.
    Values ≈ 0 → outside the mesh surface.

    Parameters
    ----------
    mesh           : pytorch3d Meshes (single mesh)
    points         : (M, 3) query points
    max_v_per_call : chunk size to avoid OOM on large meshes

    GHDHeart: ops/mesh_geometry.py::Winding_Occupancy
    """
    dual_areas     = _dual_area_vertex(mesh)
    normals_areaic = mesh.verts_normals_packed() * dual_areas.view(-1, 1)
    winding        = torch.zeros(points.shape[0], device=points.device)
    for i in range(0, normals_areaic.shape[0], max_v_per_call):
        E = _electric_strength(points, mesh.verts_packed()[i:i + max_v_per_call])
        winding += torch.einsum(
            "m n c, n c -> m", E, normals_areaic[i:i + max_v_per_call]
        )
    return winding / (4 * np.pi)


# ── DVS Occupancy Loss ────────────────────────────────────────────────────────

class DVSOccupancyLoss(nn.Module):
    """
    DVS mesh-occupancy loss: winding-number field + Binary Dice.

    Randomly subsamples target_positives (inside) and target_negatives (outside)
    each forward pass, computes winding occupancy of the deformed mesh at those
    points, and penalises disagreement with ground-truth labels via Dice loss.
    Distance weighting focuses the loss on boundary-adjacent samples.

    Parameters
    ----------
    num_sample : int   number of positive samples per forward pass
    NP_ratio   : float ratio of negative to positive samples (default 1:1)

    GHDHeart: GHD/GHD_cardiac.py::fitting2target  (Loss_occupancy block)
    """

    def __init__(
        self,
        num_sample: int   = 2000,
        NP_ratio:   float = 1.0,
    ) -> None:
        super().__init__()
        self.num_sample = num_sample
        self.NP_ratio   = NP_ratio
        self._dice      = BinaryDiceLoss()

    def forward(
        self,
        mesh:             Meshes,
        target_positives: torch.Tensor,   # (P, 3) known interior points
        target_negatives: torch.Tensor,   # (N, 3) known exterior points
        dist_weights_p2n: torch.Tensor,   # (P,)   distance-based weights
        dist_weights_n2p: torch.Tensor,   # (N,)   distance-based weights
        num_sample:       int   | None = None,
        NP_ratio:         float | None = None,
    ) -> torch.Tensor:
        ns       = num_sample or self.num_sample
        npratio  = NP_ratio   or self.NP_ratio

        pos_idx  = torch.randperm(target_positives.shape[0])[:ns]
        neg_idx  = torch.randperm(target_negatives.shape[0])[:int(ns * npratio)]

        samp_pos = target_positives[pos_idx]
        samp_neg = target_negatives[neg_idx]

        w_pos = dist_weights_p2n[pos_idx];  w_pos = w_pos / w_pos.mean()
        w_neg = dist_weights_n2p[neg_idx];  w_neg = w_neg / w_neg.mean()
        weights = torch.cat([w_pos, w_neg]).view(1, -1)

        occ_pos  = torch.sigmoid((winding_occupancy(mesh, samp_pos) - 0.5) * 10).view(1, -1)
        occ_neg  = torch.sigmoid((winding_occupancy(mesh, samp_neg) - 0.5) * 10).view(1, -1)
        occ_pred = torch.cat([occ_pos, occ_neg],                   dim=-1)
        occ_gt   = torch.cat([torch.ones_like(occ_pos),
                               torch.zeros_like(occ_neg)],         dim=-1)

        return self._dice(occ_pred, occ_gt, weight=weights)
