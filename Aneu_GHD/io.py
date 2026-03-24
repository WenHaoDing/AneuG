"""
Mesh I/O utilities
Thin wrappers around pytorch3d IO kept framework-agnostic (returns numpy arrays).
"""
from __future__ import annotations

import numpy as np
import torch
from pytorch3d.io import load_obj as _load_obj, save_obj as _save_obj


def load_obj(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a .obj file and return vertices and faces as numpy arrays.

    Parameters
    ----------
    path : str  path to .obj file

    Returns
    -------
    verts : (N, 3) float32 numpy array
    faces : (M, 3) int64  numpy array
    """
    verts_t, faces_t, _ = _load_obj(path)
    return (
        verts_t.numpy().astype(np.float32),
        faces_t.verts_idx.numpy().astype(np.int64),
    )


def save_obj(path: str, verts: np.ndarray, faces: np.ndarray) -> None:
    """
    Save vertices and faces to a .obj file.

    Parameters
    ----------
    path  : str         output path
    verts : (N, 3)      vertex positions (numpy or torch)
    faces : (M, 3)      face indices     (numpy or torch)
    """
    v = torch.as_tensor(verts, dtype=torch.float32)
    f = torch.as_tensor(faces, dtype=torch.int64)
    _save_obj(path, v, f)
