"""
Volume Loss

Replaces the earlier scale-ceiling mechanism (a hinge on `exp(log_s_ghd)`,
the single global affine scale parameter). That was empirically shown to
fail: it only constrained the AFFINE parameter, not the mesh's actual
enclosed volume, which the non-rigid GHD deformation (`phi`) can inflate
independently and unconstrained. On one real run, the affine scale stayed
within ~5% of its ceiling throughout, while the mesh's TRUE volume (affine
+ phi combined) peaked at over 2x the target's volume mid-training -- phi
was doing most of the damage, invisibly to the old mechanism.

This penalises the LIVE fitted mesh's actual enclosed volume directly,
every iteration, computed via the divergence theorem in flux form (the
same underlying identity as trimesh's mesh volume, just differentiable):
    V = (1/3) * sum_faces (face_barycenter . face_normal) * face_area
Ported/simplified (single mesh, no batching) from
AneuG/ghd/base/mesh_geometry2.py::get_soft_volume -- verified to match
trimesh's volume exactly on a watertight mesh, with working gradients back
to vertex positions.

Requires the mesh to be closed (watertight) and consistently oriented for
the result to be a real volume -- true of the canonical/fitted mesh
throughout GHD fitting (capped, uncapping only happens as a post-process).

Loss shape -- asymmetric, constant weight (no decay)
-------------------------------------------------------
    ratio = volume / target_volume
    under = relu(1 - ratio)      -- 0 if volume >= target
    over  = relu(ratio - 1)      -- 0 if volume <= target
    loss  = under^2 + (over / (ceiling_ratio - 1))^2
`ceiling_ratio` (e.g. 1.2) sets where the "over" term reaches magnitude 1.0
-- i.e. it's calibrated so the loss ramps up sharply once the mesh exceeds
that fraction of the target's volume, while still gently pulling toward a
close match below it. The "under" term is comparatively gentle (plain
squared ratio deficit) since undershooting is the expected, harmless state
early in training (the canonical starts deliberately undersized) -- only
overshooting is the failure mode this loss exists to catch.

Deliberately CONSTANT weight throughout training, unlike the old
mechanism's decay-to-zero schedule -- that decay was implicated in letting
the mesh balloon unchecked in the second half of training, exactly when it
was observed to happen.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import trimesh
from pytorch3d.structures import Meshes


def mesh_volume_watertight(V: np.ndarray, F: np.ndarray) -> float | None:
    """
    Non-differentiable reference volume (trimesh) for a FIXED mesh -- used
    once to get the target's volume, not for the live/differentiable loss.
    Attempts trimesh.repair.fill_holes() if not already watertight. Returns
    None (rather than raising) if volume still can't be reliably computed,
    so callers can disable the feature with a warning instead of crashing.
    """
    m = trimesh.Trimesh(vertices=V, faces=F, process=False)
    if not m.is_watertight:
        trimesh.repair.fill_holes(m)
    if not m.is_watertight or m.volume <= 0:
        return None
    return float(m.volume)


def differentiable_mesh_volume(mesh: Meshes) -> torch.Tensor:
    """
    Differentiable enclosed volume of a SINGLE mesh (batch size 1) --
    see module docstring. Gradients flow to `mesh`'s vertex positions.
    """
    face_areas = mesh.faces_areas_packed()            # (F,)
    face_normals = mesh.faces_normals_packed()         # (F, 3)
    verts = mesh.verts_packed()
    faces = mesh.faces_packed()
    face_barycenters = verts[faces].mean(dim=1)        # (F, 3)
    volume_ele = (face_barycenters * face_normals).sum(dim=-1) * face_areas
    return volume_ele.sum() / 3.0


class VolumeLoss(nn.Module):
    """
    Parameters
    ----------
    target_volume  : float, the target mesh's (fixed) enclosed volume --
                      from mesh_volume_watertight(V_tgt, F_tgt).
    ceiling_ratio  : float, e.g. 1.2 -- see module docstring "Loss shape".
    """

    def __init__(self, target_volume: float, ceiling_ratio: float = 1.2) -> None:
        super().__init__()
        self.target_volume = float(target_volume)
        self.ceiling_ratio = float(ceiling_ratio)

    def forward(self, mesh: Meshes, target_frac: float = 1.0) -> torch.Tensor:
        """
        `target_frac` scales the reference target volume down for a
        scheduled "grow into it" ramp (see FitConfig.volume_target_frac_*
        in fitting.py) -- e.g. 0.5 during an early phase targets HALF the
        true target volume, tightening/relaxing the ceiling proportionally
        along with it. Defaults to 1.0 (the true target), unchanged from
        before this was added.
        """
        volume = differentiable_mesh_volume(mesh)
        effective_target = self.target_volume * target_frac
        ratio = volume / effective_target
        under = torch.relu(1.0 - ratio)
        over = torch.relu(ratio - 1.0)
        return under ** 2 + (over / (self.ceiling_ratio - 1.0)) ** 2
