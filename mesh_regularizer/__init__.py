"""Normalize the surface roughness of MRI-derived vessel meshes to a physiological reference.

    from mesh_regularizer import MeshRegularizer

    reg = MeshRegularizer()                     # loads the cached reference model
    mesh, report = reg.forward("case.obj")      # regularized mesh + audit trail

See README.md for the method, and mesh_regularizer.py's module docstring for why the
roughness measure is built the way it is.
"""

from .config import (REFERENCE_ROOT, REFERENCE_FILENAME, MODEL_DIR, INPUT_ROOT,
                     INPUT_FILENAME, DOWNSTREAM_ROOT, OUTPUT_ROOT)
from .mesh_regularizer import (
    MeshRegularizer,
    ReferenceModel,
    DOWNSTREAM_EDGE_MM,
    PYMESHLAB_EDGE_BIAS,
    DEFAULT_TARGET_SCALES,
    DEFAULT_GUARD_SCALE,
    QUANTILES,
    SMOOTHERS,
    REMESHERS,
    load_mesh,
    remesh,
    multiscale_roughness,
    compensate_shrinkage,
    surface_deviation,
    plot_reference,
    plot_convergence,
    config_stem,
    model_path,
)

__all__ = [
    "MeshRegularizer", "ReferenceModel", "DOWNSTREAM_EDGE_MM", "PYMESHLAB_EDGE_BIAS",
    "QUANTILES", "SMOOTHERS", "REMESHERS",
    "DEFAULT_TARGET_SCALES", "DEFAULT_GUARD_SCALE", "load_mesh", "remesh",
    "multiscale_roughness", "compensate_shrinkage", "surface_deviation",
    "plot_reference", "plot_convergence", "config_stem", "model_path",
    "REFERENCE_ROOT", "REFERENCE_FILENAME", "MODEL_DIR", "INPUT_ROOT",
    "INPUT_FILENAME", "DOWNSTREAM_ROOT", "OUTPUT_ROOT",
]
