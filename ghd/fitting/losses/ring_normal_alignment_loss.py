"""
Ring Normal Alignment Loss

Penalises misalignment between the canonical ring's plane orientation
(normal direction) and the corresponding target ring's plane orientation —
i.e. keeps the vessel opening's cross-section from tilting relative to GT,
independent of ring position (CentroidLoss / RingLateralLoss) or in-plane
distance (RingPlanarityLoss). Those losses collapse a ring to a centroid or
measure distance-to-plane; none of them penalise the plane's orientation
directly — this one does.

Normal estimation method
------------------------
For a ring of k ordered vertices with centroid c:
    radial_i = normalize(v_i - c)
For each i, pick the OTHER ring point j whose radial vector is closest to
perpendicular to radial_i (|dot(radial_i, radial_j)| minimal). This pairing
is determined ONCE from the target ring's fixed rest geometry and reused
every iteration — an argmin in the training loop would be
non-differentiable, so the *choice* of pairs is precomputed and only the
resulting cross products are computed live:
    normal_i = normalize(cross(radial_i, radial_{pairing[i]}))
    normal   = normalize(mean_i(normal_i))

Per-point sign correction (bug fix)
------------------------------------
cross(a, b) = -cross(b, a). For a given point i, its near-perpendicular
partner can legitimately sit on either side (~+90 deg CCW or ~-90 deg CW)
in the ring's plane — both are equally valid "closest to perpendicular"
matches, but they give OPPOSITE-signed cross products. Left unresolved,
this is close to a per-point coin flip: the individual per-point cross
products are each individually accurate (~99% aligned with the ring's true
plane normal in *magnitude*), but averaging them without a consistent
orientation lets them partially or (in one observed case, a bifurcation
parent/stem ring) almost completely cancel out, producing a near-random
"normal" instead of the ring's actual plane orientation.
Fix: once, at precompute time, also pick a reference normal and flip the
sign of each per-point cross product to agree with it before averaging:
    sign_i   = +1 if dot(normal_i, normal_ref) >= 0 else -1
    normal   = normalize(mean_i(sign_i * normal_i))
`sign` is precomputed alongside `pairing` (same fixed/rest geometry, same
differentiability constraint) and reused every iteration.

Outward orientation (no longer arbitrary)
-------------------------------------------
`normal_ref` used to always be an SVD least-squares plane fit, which has an
arbitrary sign — so the resulting `normal`'s sign was arbitrary too, and
the loss had to compare via squared cosine (`1 - cos^2`) to stay invariant
to it, which made it blind to a ring that's actually flipped 180 degrees
(cos^2 can't tell "aligned" from "anti-aligned").

Now that ring vertex order is verified (see
scripts/compute_canonical_ring_order.py) and each ring's own mesh has a
well-defined true outward vertex normal, callers can pass a real outward
`normal_ref` (via `outward_reference()` below) instead of relying on the
SVD fallback. With BOTH the target and canonical/fitted sides oriented
outward this way, `RingNormalAlignmentLoss` compares direction directly
(`1 - cos`, not squared) — see its docstring.

Usage
-----
tgt_ref = outward_reference(tgt_up_pts, V_tgt, tgt_vnormals)
n_up, pairing_up, sign_up = ring_normal_and_pairing(tgt_up_pts, normal_ref=tgt_ref)

can_ref = can_vnormals[can_up_idxs].mean(0)
_, pairing_can, sign_can = ring_normal_and_pairing(V_init[can_up_idxs], normal_ref=can_ref)

loss_fn = RingNormalAlignmentLoss([
    (can_up_idxs, pairing_can, sign_can, n_up),
    ...
]).to(device)

# Each iteration:
loss = loss_fn(V_rendered)
"""
import numpy as np
import torch
import torch.nn as nn


def outward_reference(pts: np.ndarray, V: np.ndarray, vnormals: np.ndarray) -> np.ndarray:
    """
    Mean outward vertex normal of the mesh vertices nearest to `pts`.

    Works whether or not `pts` are exactly rows of `V` -- if they are (e.g.
    `pts = V[idxs]`), the nearest-vertex match is exact (distance 0); if
    `pts` are positions without known indices into `V` (e.g. a target ring
    whose indices weren't threaded through), this still recovers the right
    local reference via nearest-neighbour lookup.

    Parameters
    ----------
    pts      : (k, 3) query positions
    V        : (N, 3) mesh vertices
    vnormals : (N, 3) per-vertex outward normals for V (e.g.
               trimesh.Trimesh(...).vertex_normals)
    """
    pts = np.asarray(pts, dtype=np.float64)
    d = np.linalg.norm(V[None, :, :] - pts[:, None, :], axis=-1)
    return vnormals[d.argmin(axis=1)].mean(axis=0)


def ring_normal_pairing_from_order(k: int, offset_frac: float = 0.25) -> tuple:
    """
    Build a winding-consistent pairing directly from a VERIFIED cyclic walk
    order (see scripts/compute_canonical_ring_order.py), instead of
    searching for a near-perpendicular partner per point.

    Pairs point i with the point `offset_frac` of the way around the ring
    in the SAME verified walking direction (default a quarter-turn, i.e.
    roughly perpendicular for a ring close to circular -- true for these
    canonicals, measured roundness 0.90-0.98 once correctly ordered):
        pairing[i] = (i + round(k * offset_frac)) % k

    Because the walk order's own winding is outward by construction (see
    compute_canonical_ring_order.py), and every pair uses the same forward
    step in that same direction, cross(radial_i, radial_{pairing[i]}) =
    sin(delta_theta) * outward_normal with delta_theta ~ +90 deg for EVERY
    i -- no per-point cancellation is possible, unlike the nearest-
    perpendicular search this replaces, so unlike `ring_normal_and_pairing`
    no per-point sign correction against a reference is needed either
    (sign is always +1).

    Only valid when the ring's vertex order is actually verified walk
    order -- on an arbitrarily-ordered ring (e.g. the raw landmarks.npz
    arrays before scripts/compute_canonical_ring_order.py), "i + k/4 index
    positions away" has no relationship to "90 degrees around the ring", so
    this would silently produce garbage. Use `ring_normal_and_pairing`
    (order-independent) instead when the order isn't verified.

    Returns
    -------
    pairing : (k,) int64
    sign    : (k,) float32, always +1 (kept for interface compatibility
        with `ring_normal_and_pairing`'s output / RingNormalAlignmentLoss).
    """
    offset = max(1, round(k * offset_frac))
    pairing = (np.arange(k) + offset) % k
    sign = np.ones(k, dtype=np.float32)
    return pairing.astype(np.int64), sign


def ring_normal_and_pairing(pts: np.ndarray, normal_ref: np.ndarray | None = None):
    """
    Precompute (once, from a fixed/target ring) the near-perpendicular
    index pairing and the resulting ring normal.

    Parameters
    ----------
    pts        : (k, 3) ring vertex positions, in ring order.
    normal_ref : (3,) optional reference direction to orient the result
        against (e.g. from `outward_reference()`, so the result reliably
        points outward). Falls back to an SVD least-squares plane fit
        (arbitrary sign) if not given.

    Returns
    -------
    normal  : (3,) float32 unit vector. Outward-oriented if `normal_ref`
        was given; otherwise arbitrary sign (tied to the SVD fallback).
    pairing : (k,) int64, pairing[i] = j such that radial_i, radial_j are
        the closest-to-perpendicular pair for point i (j != i).
    sign    : (k,) float32, +1/-1 per point — orients cross(radial_i,
        radial_{pairing[i]}) consistently before averaging (see module
        docstring "Per-point sign correction").
    """
    pts = np.asarray(pts, dtype=np.float64)
    k = len(pts)
    if k < 3:
        raise ValueError(f"ring_normal_and_pairing needs >= 3 points, got {k}")

    c = pts.mean(axis=0)
    radial = pts - c
    radial = radial / np.linalg.norm(radial, axis=1, keepdims=True)

    if normal_ref is None:
        # Robust reference normal (least-squares plane fit) used only to
        # consistently orient the per-point cross products below -- the
        # noisy per-point pairwise method is what's actually used for
        # `normal` (kept for consistency with the live/differentiable
        # path), but its sign ambiguity per point must be resolved against
        # a stable reference or the mean can cancel almost entirely (see
        # module docstring). Arbitrary sign if no true reference is given.
        _, _, Vt = np.linalg.svd(pts - c, full_matrices=False)
        normal_ref = Vt[-1]
    else:
        normal_ref = np.asarray(normal_ref, dtype=np.float64)
        normal_ref = normal_ref / np.linalg.norm(normal_ref)

    pairing = np.zeros(k, dtype=np.int64)
    sign = np.ones(k, dtype=np.float32)
    normals = []
    for i in range(k):
        dots = np.abs(radial @ radial[i])
        dots[i] = np.inf
        j = int(np.argmin(dots))
        pairing[i] = j
        cr = np.cross(radial[i], radial[j])
        n = np.linalg.norm(cr)
        if n > 1e-8:
            cr = cr / n
            if cr @ normal_ref < 0:
                sign[i] = -1.0
                cr = -cr
            normals.append(cr)

    if not normals:
        raise ValueError("ring_normal_and_pairing: all candidate pairs were degenerate")

    normal = np.mean(normals, axis=0)
    normal = normal / np.linalg.norm(normal)
    return normal.astype(np.float32), pairing, sign


class RingNormalAlignmentLoss(nn.Module):
    """
    Penalises the canonical ring's plane normal drifting away from the
    target ring's plane normal -- DIRECTIONALLY, not just up to sign.

    Requires both `normal_tgt` and the canonical/fitted side's `pairing`/
    `sign` to have been computed with a true outward `normal_ref` (see
    `outward_reference()` and `ring_normal_and_pairing()` above) -- if
    either side's orientation is arbitrary instead of genuinely outward,
    this loss will incorrectly penalise a correctly-outward-facing ring
    just because it disagrees with an arbitrarily-signed reference. Use the
    old `1 - cos^2` (sign-invariant) formulation instead if you can't
    guarantee both sides are outward-oriented.

    Parameters
    ----------
    specs : list of (can_ring_idxs, pairing, sign, normal_tgt) tuples
        can_ring_idxs : array-like (k,)  canonical ring vertex indices, in
                        the SAME per-vertex order as the target ring used
                        to build `pairing` / `sign` / `normal_tgt`.
        pairing       : array-like (k,) int, from ring_normal_and_pairing.
        sign          : array-like (k,) +-1, from ring_normal_and_pairing.
        normal_tgt    : (3,) target ring plane normal, outward-oriented.

    Supports any number of rings (2 for sidewall, 3 for bifurcation).
    """

    def __init__(self, specs: list) -> None:
        super().__init__()
        self.n_rings = len(specs)
        for i, (idxs, pairing, sign, n_tgt) in enumerate(specs):
            self.register_buffer(f"idxs_{i}", torch.as_tensor(idxs, dtype=torch.long))
            self.register_buffer(f"pairing_{i}", torch.as_tensor(pairing, dtype=torch.long))
            self.register_buffer(f"sign_{i}", torch.as_tensor(sign, dtype=torch.float32))
            nt = torch.as_tensor(n_tgt, dtype=torch.float32)
            self.register_buffer(f"normal_tgt_{i}", nt / nt.norm())

    @staticmethod
    def _ring_normal(pts: torch.Tensor, pairing: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
        centroid = pts.mean(dim=0)
        radial = pts - centroid
        radial = radial / radial.norm(dim=1, keepdim=True).clamp_min(1e-8)
        paired = radial[pairing]
        normals = torch.cross(radial, paired, dim=1)
        normals = normals / normals.norm(dim=1, keepdim=True).clamp_min(1e-8)
        normals = normals * sign.unsqueeze(1)
        normal = normals.mean(dim=0)
        return normal / normal.norm().clamp_min(1e-8)

    def forward(self, verts: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        verts : (N, 3) deformed canonical vertices (V_rendered)

        Returns
        -------
        Scalar: mean(1 - dot(normal_fit, normal_tgt)) across rings.
        0 when every ring's outward normal exactly matches its target's
        (both must be outward-oriented -- see class docstring); up to 2 when
        exactly anti-aligned (a ring flipped 180 degrees is now penalised,
        unlike the old squared-cosine version).
        """
        total = verts.new_zeros(())
        for i in range(self.n_rings):
            idxs = getattr(self, f"idxs_{i}")
            pairing = getattr(self, f"pairing_{i}")
            sign = getattr(self, f"sign_{i}")
            normal_tgt = getattr(self, f"normal_tgt_{i}")

            normal_fit = self._ring_normal(verts[idxs], pairing, sign)
            cos = (normal_fit * normal_tgt).sum()
            total = total + (1 - cos)
        return total / self.n_rings
