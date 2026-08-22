"""
Geodesic Correspondence Loss

Standard chamfer distance matches points by pure Euclidean proximity in the
CURRENT (possibly still badly wrong) fitted pose -- it has no notion of
"which part of the vessel" a point belongs to, only where it currently sits
in space. Early in training this can happily match the wrong cross-section
of a tube-like vessel to a spatially-nearby-but-topologically-wrong target
region.

This augments each point with its graph geodesic distance from each of the
mesh's openings (shortest path along mesh edges, not Euclidean straight-line
distance), then runs chamfer distance on the augmented
[x, y, z, w*g_0, w*g_1, ...] coordinates instead of raw [x, y, z]. A point
deep in the wrong part of the vessel now has a large mismatch in its
geodesic coordinates even if it happens to currently sit close in 3D space
to some target point, which chamfer alone would happily (and wrongly)
match.

Channel design -- ONE channel per opening
-------------------------------------------
geodesic_from_openings(V, F, opening_idx_sets) -> (N, K) per-vertex,
per-opening geodesic distance, normalised per-channel to [0, 1] (0 on that
opening's own ring, 1 at the vertex furthest from it). K channels for K
openings -- 2 for sidewall, 3 for bifurcation.

Each channel has a genuinely different source (a different opening), so
even where two channels look visually similar for a particular mesh (e.g.
a sidewall case where both openings sit close together relative to a large
dome, making both channels large over most of the dome and small near the
neck -- see conversation notes / geodesic_distance_check.png), they remain
mathematically distinct fields, not a duplicate. An alternative,
`geodesic_pairwise_ratio`, collapses each PAIR of openings into a single
relative-position channel (C(K,2) channels total) -- kept in this module as
an option, but `geodesic_from_openings` (K raw per-opening channels) is the
one actually used by the loss / precompute below.

Precompute (once)
------------------
geodesic_from_openings(V, F, opening_idx_sets) -> (N, K) per-vertex,
per-opening channels. Computed ONCE on each mesh's fixed topology:
    - canonical side: reused every iteration since GHD deformation only
      moves vertex POSITIONS, never changes vertex count/topology/faces.
    - target side: fixed throughout (target never moves).

Masking
-------
Points very close to an opening are excluded from the loss (both sides) --
the ratio is close to degenerate there (dominated by whichever opening is
nearly-0-distance away), and the actual ring position/shape is already
handled by the dedicated ring losses. Implemented as a threshold directly
on the raw (pre-ratio) geodesic distances: exclude any vertex whose
distance to ANY opening (normalised by that opening's own max distance) is
below `mask_eps`, so this doesn't need a separate cap-vs-body vertex
classification.
"""
from __future__ import annotations

import itertools

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
import torch
import torch.nn as nn
from pytorch3d.loss import chamfer_distance


def build_edge_graph(V: np.ndarray, F: np.ndarray) -> sp.csr_matrix:
    """Undirected mesh edge graph, edge weight = Euclidean edge length."""
    edges = set()
    for tri in F:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            a, b = int(a), int(b)
            edges.add((a, b))
            edges.add((b, a))
    edges = np.array(list(edges), dtype=np.int64)
    weights = np.linalg.norm(V[edges[:, 0]] - V[edges[:, 1]], axis=1)
    N = len(V)
    return sp.coo_matrix((weights, (edges[:, 0], edges[:, 1])), shape=(N, N)).tocsr()


def raw_geodesic_distances(V: np.ndarray, F: np.ndarray, opening_idx_sets: list) -> np.ndarray:
    """
    Graph geodesic distance from every vertex to each opening (multi-source
    shortest path from the whole ring at once), UNNORMALISED (raw mesh
    units). Building block for both geodesic_from_openings and
    geodesic_pairwise_ratio below.

    Returns
    -------
    (N, K) float64 array.
    """
    graph = build_edge_graph(V, F)
    N = len(V)
    K = len(opening_idx_sets)
    raw = np.zeros((N, K), dtype=np.float64)
    for k, idxs in enumerate(opening_idx_sets):
        d = dijkstra(graph, indices=np.asarray(idxs), directed=False)  # (len(idxs), N)
        raw[:, k] = d.min(axis=0)
    return raw


def geodesic_from_openings(V: np.ndarray, F: np.ndarray, opening_idx_sets: list) -> np.ndarray:
    """
    Primary channel design: ONE channel per opening (K channels for K
    openings -- 2 for sidewall, 3 for bifurcation), each the per-vertex
    graph geodesic distance to that opening, normalised per-channel to
    [0, 1] (0 on that opening's own ring, 1 at the vertex furthest from it).

    Each channel has a genuinely different source (a different opening), so
    even where two channels end up visually similar for a particular mesh
    (e.g. two openings sitting close together relative to a large dome --
    see conversation notes), they remain mathematically distinct fields,
    not a redundant duplicate -- kept as K separate channels rather than
    collapsed into pairwise ratios (see geodesic_pairwise_ratio for that
    alternative).

    Returns
    -------
    (N, K) float64 array.
    """
    raw = raw_geodesic_distances(V, F, opening_idx_sets)
    return raw / raw.max(axis=0, keepdims=True)


def geodesic_pairwise_ratio(V: np.ndarray, F: np.ndarray, opening_idx_sets: list):
    """
    Per-vertex, per-PAIR-of-openings relative geodesic coordinate -- see
    module docstring "Channel design".

    Parameters
    ----------
    V, F              : mesh vertices / faces
    opening_idx_sets  : list of K arrays of vertex indices, one per opening

    Returns
    -------
    ratio : (N, C(K,2)) float64, ratio[:, c] in [0, 1] for pair pairs[c]
    pairs : list of C(K,2) (i, j) tuples, channel order matches `ratio`'s
            columns (itertools.combinations(range(K), 2))
    raw   : (N, K) float64, the underlying raw per-opening distances (also
            returned since the masking step needs them)
    """
    raw = raw_geodesic_distances(V, F, opening_idx_sets)
    K = len(opening_idx_sets)
    pairs = list(itertools.combinations(range(K), 2))
    ratio = np.zeros((len(V), len(pairs)), dtype=np.float64)
    for c, (i, j) in enumerate(pairs):
        ratio[:, c] = raw[:, i] / (raw[:, i] + raw[:, j] + 1e-12)
    return ratio, pairs, raw


def geodesic_mask(raw: np.ndarray, mask_eps: float = 0.05) -> np.ndarray:
    """
    True = keep this vertex -- excludes anything within `mask_eps` (as a
    fraction of that opening's own max distance) of ANY opening. See module
    docstring "Masking".
    """
    raw_norm = raw / raw.max(axis=0, keepdims=True)
    return (raw_norm > mask_eps).all(axis=1)


class GeodesicChamferLoss(nn.Module):
    """
    Chamfer distance on [x, y, z, w*r_0, ..., w*r_{C-1}] instead of raw
    [x, y, z], where r_c are the per-pair geodesic ratio channels -- see
    module docstring.

    Parameters
    ----------
    geo_can    : (N_can, C) canonical per-vertex ratio channels
                 (geodesic_pairwise_ratio on the canonical mesh)
    mask_can   : (N_can,) bool, True = include this canonical vertex
                 (see geodesic_mask)
    V_tgt      : (N_tgt, 3) target vertices (fixed, normalised space)
    geo_tgt    : (N_tgt, C) target per-vertex ratio channels
    mask_tgt   : (N_tgt,) bool, True = include this target vertex
    geo_weight : scales the geodesic channels relative to the xyz channels
                 in the augmented coordinate (both are O(0-1), so 1.0 is a
                 reasonable starting point -- tune to trade off positional
                 vs. topological matching strength)
    """

    def __init__(
        self,
        geo_can: np.ndarray, mask_can: np.ndarray,
        V_tgt: np.ndarray, geo_tgt: np.ndarray, mask_tgt: np.ndarray,
        geo_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.geo_weight = geo_weight
        self.register_buffer("geo_can", torch.as_tensor(geo_can, dtype=torch.float32))
        self.register_buffer("mask_can", torch.as_tensor(mask_can, dtype=torch.bool))

        tgt_pts = np.asarray(V_tgt)[mask_tgt]
        tgt_geo = np.asarray(geo_tgt)[mask_tgt] * geo_weight
        tgt_aug = np.concatenate([tgt_pts, tgt_geo], axis=1)
        self.register_buffer("tgt_aug", torch.as_tensor(tgt_aug, dtype=torch.float32).unsqueeze(0))

    def forward(self, V_rendered: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        V_rendered : (N_can, 3) deformed canonical vertices

        Returns
        -------
        Scalar chamfer distance in augmented [xyz, w*geo] space.
        """
        src_pts = V_rendered[self.mask_can]
        src_geo = self.geo_can[self.mask_can] * self.geo_weight
        src_aug = torch.cat([src_pts, src_geo], dim=-1).unsqueeze(0)
        loss, _ = chamfer_distance(src_aug, self.tgt_aug)
        return loss
