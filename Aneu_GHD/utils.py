"""
Mesh utilities for GHD fitting.

compute_eigenvectors()
    Build a mixed Laplacian from the canonical mesh and compute its
    eigenvector basis U used by ghd_fit() as the deformation space.
    Call once per canonical mesh and cache the result (eigenvectors.npy).
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh


# ── Laplacian constructors ────────────────────────────────────────────────────

def cotangent_laplacian(V: np.ndarray, F: np.ndarray) -> object:
    """Cotangent-weighted Laplacian (geometry-aware, preferred for smooth meshes)."""
    N = V.shape[0]
    i, j, k = F[:, 0], F[:, 1], F[:, 2]
    vi, vj, vk = V[i], V[j], V[k]

    def _cot(a, b):
        return (a * b).sum(1) / (np.linalg.norm(np.cross(a, b), axis=1) + 1e-12)

    ci = _cot(vj - vi, vk - vi)
    cj = _cot(vk - vj, vi - vj)
    ck = _cot(vi - vk, vj - vk)

    rows = np.concatenate([j, k, k, i, i, j])
    cols = np.concatenate([k, j, i, k, j, i])
    vals = 0.5 * np.concatenate([ci, ci, cj, cj, ck, ck])
    W = coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    W = 0.5 * (W + W.T)
    return diags(np.asarray(W.sum(1)).ravel()) - W


def invlength_laplacian(V: np.ndarray, F: np.ndarray) -> object:
    """Inverse edge-length weighted Laplacian (scale-sensitive)."""
    N = V.shape[0]
    edges = np.unique(np.sort(
        np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1
    ), axis=0)
    w = 1.0 / (np.linalg.norm(V[edges[:, 0]] - V[edges[:, 1]], axis=1) + 1e-12)
    r = np.concatenate([edges[:, 0], edges[:, 1]])
    c = np.concatenate([edges[:, 1], edges[:, 0]])
    W = coo_matrix((np.concatenate([w, w]), (r, c)), shape=(N, N)).tocsr()
    return diags(np.asarray(W.sum(1)).ravel()) - W


def uniform_laplacian(V: np.ndarray, F: np.ndarray) -> object:
    """Uniform (unweighted) Laplacian."""
    N = V.shape[0]
    edges = np.unique(np.sort(
        np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1
    ), axis=0)
    r = np.concatenate([edges[:, 0], edges[:, 1]])
    c = np.concatenate([edges[:, 1], edges[:, 0]])
    A = coo_matrix((np.ones(r.shape[0]), (r, c)), shape=(N, N)).tocsr()
    return diags(np.asarray(A.sum(1)).ravel()) - A


# ── Public API ────────────────────────────────────────────────────────────────

def compute_eigenvectors(
    V_n:     np.ndarray,
    F:       np.ndarray,
    n_basis: int   = 121,
    lam:     float = 1e-3,
) -> np.ndarray:
    """
    Build a mixed Laplacian and compute its n_basis smallest eigenvectors.

    The mixed Laplacian combines cotangent, inverse-length and uniform
    weights to capture both geometric and topological structure:
        L = L_cot + lam * L_invlen + lam * L_uniform

    Parameters
    ----------
    V_n     : (N, 3) normalised canonical vertices
    F       : (M, 3) canonical faces
    n_basis : number of eigenvectors (columns of U); default 121 = 11²
    lam     : weight for the two auxiliary Laplacians; default 1e-3

    Returns
    -------
    U : (N, n_basis) float32  eigenvector matrix, sorted by eigenvalue
    """
    print(f"  Building mixed Laplacian ({V_n.shape[0]} verts)...")
    L = (cotangent_laplacian(V_n, F)
         + lam * invlength_laplacian(V_n, F)
         + lam * uniform_laplacian(V_n, F)).tocsr()

    print(f"  Computing {n_basis} eigenvectors...")
    vals, vecs = eigsh(L, k=n_basis, which='SM')
    idx = np.argsort(vals)
    print(f"  Done.  near-zero eigenvalues: {int((vals < 1e-8).sum())}")
    return vecs[:, idx].astype(np.float32)
