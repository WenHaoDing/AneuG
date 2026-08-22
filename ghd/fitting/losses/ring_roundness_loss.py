"""
Ring Roundness Loss

Aneurysm sidewall/bifurcation rings (vessel-opening boundary loops) are
anatomically expected to be close to circular. Nothing currently in the
loss stack actually constrains the ring's IN-PLANE SHAPE:
  - RingChamferLoss    pulls each point toward its nearest target point,
                        but doesn't preserve the ring's cyclic topology --
                        points can bunch up or fold over each other.
  - CentroidLoss / RingLateralLoss   only constrain the ring's mean position.
  - RingPlanarityLoss  only constrains distance to a fixed plane, not the
                        in-plane distribution within it.
  - RingNormalAlignmentLoss   only constrains the *orientation* of a proxy
                        normal, which can be satisfied even when the ring's
                        real shape is badly distorted (see that module's
                        docstring).

So nothing stops a "Pac-Man" ring: points bunched on one side, a bite taken
out, wildly uneven radius -- as long as each point individually lands near
*some* target point and near the target plane, the existing losses are
satisfied.

This adds the classic shape-analysis circularity metric, the isoperimetric
ratio, computed on the ring as a closed planar polygon:

    roundness = 4 * pi * Area / Perimeter^2

roundness == 1 for a perfect circle and decreases for any elongation,
bulge, or concavity. It is more sensitive to a Pac-Man-style bite than a
simple per-point radius-variance metric would be, because a concavity
shrinks enclosed area relative to perimeter much faster than it changes
average radius.

    loss_ring = 1 - roundness

Requirements to compute this correctly
---------------------------------------
1. A FIXED plane (normal + centroid) to project the ring onto. Reuse an
   already-trusted normal (e.g. from centerline_tangent.py or
   ring_normal_and_pairing on the target ring) -- NOT a live SVD on the
   deforming points, which is numerically fragile when singular values are
   close together (see ring_normal_alignment_loss.py docstring).
2. A cyclic WALK ORDER for the ring vertices. The raw ring_up_idxs /
   ring_dn_idxs arrays in landmarks.npz are NOT stored in walk order --
   verified empirically: consecutive-in-array edge lengths for the
   canonical ring are comparable to *randomly shuffled* edge lengths, i.e.
   effectively unordered (sorted by raw vertex id instead). The true walk
   order is a topological property of the ring's ridge loop in the mesh
   and is recovered ONCE from face connectivity via
   `extract_boundary_loop_order`, then reused every iteration -- the
   vertex/face connectivity of the canonical mesh doesn't change during
   GHD fitting, only positions do.

Usage
-----
order_up = extract_boundary_loop_order(F_can, can_up_idxs)
order_dn = extract_boundary_loop_order(F_can, can_dn_idxs)
loss_fn = RingRoundnessLoss([
    (order_up, n_up, c_up),
    (order_dn, n_dn, c_dn),
]).to(device)
loss = loss_fn(V_rendered)
"""
import math

import numpy as np
import torch
import torch.nn as nn


def extract_boundary_loop_order(F: np.ndarray, ring_idxs) -> np.ndarray:
    """
    Recover the cyclic walk order of a mesh boundary loop from face
    connectivity.

    Parameters
    ----------
    F         : (M, 3) int array of face indices for the whole mesh
                (the ring must be one of its boundary loops).
    ring_idxs : array-like (k,) vertex indices belonging to the ring.

    Note: on a CAPPED mesh (the canonical / fitted meshes are capped during
    fitting -- uncapping is a separate post-process, see uncap_fitted.py)
    the ring is not an open mesh boundary; it's a closed "ridge" loop
    separating the cap triangles from the body triangles, so all its edges
    have face-count 2 like any other manifold edge. We therefore don't
    filter by boundary (face-count == 1) edges -- we take ALL mesh edges
    and keep the ones with both endpoints in the ring; by construction each
    ring vertex has exactly 2 such neighbours (its predecessor/successor
    around the loop), regardless of whether the mesh is capped or open.

    Returns
    -------
    ordered : (k,) int64 array -- ring_idxs permuted into walk order, i.e.
              ordered[i] and ordered[(i + 1) % k] are adjacent along the
              loop.
    """
    ring_set = {int(i) for i in ring_idxs}

    edges: set[tuple[int, int]] = set()
    for tri in F:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            a, b = int(a), int(b)
            edges.add((a, b) if a < b else (b, a))

    adjacency: dict[int, list] = {i: [] for i in ring_set}
    for a, b in edges:
        if a in ring_set and b in ring_set:
            adjacency[a].append(b)
            adjacency[b].append(a)

    bad = [v for v, nbrs in adjacency.items() if len(nbrs) != 2]
    if bad:
        raise ValueError(
            f"extract_boundary_loop_order: {len(bad)} ring vertices don't have "
            f"exactly 2 ring-neighbours (degrees="
            f"{[len(adjacency[v]) for v in bad[:5]]}...). The ring may not be a "
            f"single simple loop, or ring_idxs includes vertices whose loop "
            f"neighbours aren't directly mesh-connected (e.g. a coarsely "
            f"subsampled ring)."
        )

    start = next(iter(ring_set))
    order = [start]
    prev, curr = None, start
    while True:
        nbrs = adjacency[curr]
        nxt = nbrs[0] if nbrs[0] != prev else nbrs[1]
        if nxt == start:
            break
        order.append(nxt)
        prev, curr = curr, nxt

    if len(order) != len(ring_set):
        raise ValueError(
            f"extract_boundary_loop_order: walked {len(order)} vertices but the "
            f"ring has {len(ring_set)} -- loop may be disconnected."
        )
    return np.array(order, dtype=np.int64)


class RingRoundnessLoss(nn.Module):
    """
    Isoperimetric roundness loss: mean over rings of (1 - 4*pi*Area/Perimeter^2),
    computed on each ring's projection into a FIXED plane (given normal +
    centroid), walked in a FIXED cyclic order.

    Parameters
    ----------
    ring_specs : list of (ordered_idxs, normal, centroid) tuples
        ordered_idxs : (k,) int, ring vertex indices in WALK order
                       (see extract_boundary_loop_order)
        normal       : (3,) plane normal to project onto (fixed / target)
        centroid     : (3,) plane origin (fixed / target)
    """

    def __init__(self, ring_specs: list) -> None:
        super().__init__()
        self.n_rings = len(ring_specs)
        for i, (idxs, n, c) in enumerate(ring_specs):
            self.register_buffer(f"idxs_{i}", torch.as_tensor(idxs, dtype=torch.long))
            nt = torch.as_tensor(n, dtype=torch.float32)
            self.register_buffer(f"normal_{i}", nt / nt.norm())
            self.register_buffer(f"centroid_{i}", torch.as_tensor(c, dtype=torch.float32))

    @staticmethod
    def _plane_basis(n: torch.Tensor):
        tmp = torch.tensor([1.0, 0.0, 0.0], device=n.device, dtype=n.dtype)
        if abs(n[0].item()) > 0.9:
            tmp = torch.tensor([0.0, 1.0, 0.0], device=n.device, dtype=n.dtype)
        e1 = torch.cross(n, tmp, dim=-1)
        e1 = e1 / e1.norm()
        e2 = torch.cross(n, e1, dim=-1)
        return e1, e2

    def _ring_roundness(self, pts: torch.Tensor, normal: torch.Tensor, centroid: torch.Tensor) -> torch.Tensor:
        """pts must already be in WALK order. Returns scalar roundness in (0, 1]."""
        e1, e2 = self._plane_basis(normal)
        rel = pts - centroid
        x = rel @ e1
        y = rel @ e2
        x_next = torch.roll(x, -1)
        y_next = torch.roll(y, -1)
        area = 0.5 * torch.abs(torch.sum(x * y_next - x_next * y))          # shoelace
        perim = torch.sum(torch.sqrt((x_next - x) ** 2 + (y_next - y) ** 2 + 1e-12))
        return 4 * math.pi * area / (perim ** 2 + 1e-12)

    def roundness_per_ring(self, verts: torch.Tensor) -> list:
        """Diagnostic: raw isoperimetric roundness score per ring (1.0 = perfect circle)."""
        out = []
        for i in range(self.n_rings):
            idxs = getattr(self, f"idxs_{i}")
            normal = getattr(self, f"normal_{i}")
            centroid = getattr(self, f"centroid_{i}")
            with torch.no_grad():
                out.append(self._ring_roundness(verts[idxs], normal, centroid).item())
        return out

    def forward(self, verts: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        verts : (N, 3) deformed canonical vertices (V_rendered)

        Returns
        -------
        Scalar: mean(1 - roundness) across rings.
        """
        total = verts.new_zeros(())
        for i in range(self.n_rings):
            idxs = getattr(self, f"idxs_{i}")
            normal = getattr(self, f"normal_{i}")
            centroid = getattr(self, f"centroid_{i}")
            roundness = self._ring_roundness(verts[idxs], normal, centroid)
            total = total + (1.0 - roundness)
        return total / self.n_rings
