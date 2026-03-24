"""
aneu_ghd — GHD fitting pipeline for intracranial aneurysm meshes.

Quick start
-----------
from aneu_ghd import (
    load_obj, save_obj,
    load_landmarks, normalize_lm, compute_landmarks_from_rings,
    anatomy_align, cpd_align,
    FitConfig, FitResult, ghd_fit, prepare_dvs_samples,
)
"""

from .io import load_obj, save_obj
from .utils import compute_eigenvectors

from .landmarks import (
    load_landmarks,
    normalize_lm,
    compute_landmarks_from_rings,
)

from .alignment import (
    build_frame,
    anatomy_align,
    cpd_align,
)

from .fitting import (
    FitConfig,
    FitResult,
    ghd_fit,
    prepare_dvs_samples,
)

from .losses import (
    chamfer_loss,
    BinaryDiceLoss,
    DVSOccupancyLoss,
    winding_occupancy,
    LandmarkLoss,
    laplacian_loss,
    normal_consistency_loss,
    EdgeLengthLoss,
    GHDRigidLoss,
)

__all__ = [
    # I/O
    "load_obj",
    "save_obj",
    # Utils
    "compute_eigenvectors",
    # Landmarks
    "load_landmarks",
    "normalize_lm",
    "compute_landmarks_from_rings",
    # Alignment
    "build_frame",
    "anatomy_align",
    "cpd_align",
    # Fitting
    "FitConfig",
    "FitResult",
    "ghd_fit",
    "prepare_dvs_samples",
    # Losses
    "chamfer_loss",
    "BinaryDiceLoss",
    "DVSOccupancyLoss",
    "winding_occupancy",
    "LandmarkLoss",
    "laplacian_loss",
    "normal_consistency_loss",
    "EdgeLengthLoss",
    "GHDRigidLoss",
]
