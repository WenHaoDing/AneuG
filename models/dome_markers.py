"""Differentiable morphological markers of the aneurysm dome: volume and aspect ratio.

The dome is where the morphology sensor's dome probability p is above 0.5, cut exactly on the
piecewise-linear mesh: a face with all three vertices above keeps the whole triangle, a face the
p = 0.5 line crosses keeps only its dome part (a triangle or a quad, cut at the crossing points
X on its edges, the same points the rim losses use). The dome is closed by a fan from the neck
centre o (the mean of the crossing points) to the rim.

  volume        closed-dome volume, by the divergence theorem with o as the origin: every fan
                triangle contains o, so contributes 0, and V = sum over dome (sub)faces of
                det(a - o, b - o, c - o) / 6. The same number as closing the mesh with the fan
                and taking its volume (trimesh does exactly this sum), without building the fan.
  neck_normal   direction of the dome's total area vector (the closed surface's area vectors sum
                to 0, so it is minus the neck cap's): needs no rim ordering, points into the dome
  height        dome height along the neck normal: smooth max of (v - o).n over the dome points
                (dome vertices and crossing points)
  neck_radius   mean distance of the crossing points to o
  aspect_ratio  height / neck_radius (2x the clinical height / neck diameter)

Differentiable in the vertex positions and in p (through the crossing points); continuous when a
vertex crosses 0.5 (the crossing point then reaches the vertex). Units are those of V; volume
scales with the cube. Faces must be consistently oriented outwards (volume > 0).

  dm = DomeMarkers()
  out = dm(V, p, faces)        # V (B, N, 3), p (B, N), faces (F, 3) shared by the batch
"""
import torch


class DomeMarkers(torch.nn.Module):
    def __init__(self, thr=0.5, tau_frac=0.005):
        super().__init__()
        self.thr = thr
        self.tau_frac = tau_frac        # smooth-max temperature, as a fraction of the neck radius

    @staticmethod
    def edges(faces):
        """Unique edges (E, 2) and, per face corner k, the edge from corner k to k + 1 (F, 3)."""
        fe = torch.stack([faces, faces.roll(-1, 1)], 2).sort(2).values           # (F, 3, 2)
        e, inv = torch.unique(fe.reshape(-1, 2), dim=0, return_inverse=True)
        return e, inv.reshape(-1, 3)

    def crossings(self, V, p, e):
        """Point where p = thr on every edge (B, E, 3) and the mask of edges it really crosses."""
        pi, pj = p[:, e[:, 0]], p[:, e[:, 1]]
        den = torch.where((pj - pi).abs() < 1e-6, torch.full_like(pi, 1e-6), pj - pi)
        lam = ((self.thr - pi) / den).clamp(0, 1)
        X = V[:, e[:, 0]] + lam[..., None] * (V[:, e[:, 1]] - V[:, e[:, 0]])
        return X, (pi > self.thr) != (pj > self.thr)

    def dome_triangles(self, V, p, faces):
        """The dome part of every face as up to two triangles: (B, F, 2, 3, 3) corners and a
        (B, F, 2) mask of the triangles that exist. Winding as the face's."""
        e, fe = self.edges(faces)
        X, cross = self.crossings(V, p, e)
        B, F = len(V), len(faces)
        up = p[:, faces] > self.thr                                              # (B, F, 3)
        n_up = up.sum(2)
        tri = V.new_zeros(B, F, 2, 3, 3)
        ok = torch.zeros(B, F, 2, dtype=torch.bool, device=V.device)
        P = V[:, faces]                                                          # (B, F, 3, 3)
        Xf = X[:, fe]                                                            # (B, F, 3, 3): X on edge k -> k+1
        full = n_up == 3
        tri[:, :, 0] = torch.where(full[..., None, None], P, tri[:, :, 0])
        ok[:, :, 0] |= full
        for r in range(3):                                                       # u0 = corner r
            i0, i1, i2 = r, (r + 1) % 3, (r + 2) % 3
            u0, u1 = P[:, :, i0], P[:, :, i1]
            x01, x12, x20 = Xf[:, :, i0], Xf[:, :, i1], Xf[:, :, i2]
            one = up[..., i0] & ~up[..., i1] & ~up[..., i2]                      # only u0: (u0, x01, x20)
            two = up[..., i0] & up[..., i1] & ~up[..., i2]                       # u0, u1: (u0, u1, x12, x20)
            t1 = torch.stack([u0, x01, x20], 2)
            q1, q2 = torch.stack([u0, u1, x12], 2), torch.stack([u0, x12, x20], 2)
            tri[:, :, 0] = torch.where(one[..., None, None], t1, tri[:, :, 0])
            tri[:, :, 0] = torch.where(two[..., None, None], q1, tri[:, :, 0])
            tri[:, :, 1] = torch.where(two[..., None, None], q2, tri[:, :, 1])
            ok[:, :, 0] |= one | two
            ok[:, :, 1] |= two
        return tri, ok, X, cross

    def forward(self, V, p, faces):
        tri, ok, X, cross = self.dome_triangles(V, p, faces)
        w = cross.to(V.dtype)
        n_cross = w.sum(1).clamp(min=1)
        o = (w[..., None] * X).sum(1) / n_cross[:, None]                         # (B, 3) neck centre
        a, b, c = (tri[..., k, :] - o[:, None, None] for k in range(3))
        okf = ok.to(V.dtype)
        volume = (torch.linalg.det(torch.stack([a, b, c], -1)) * okf).sum((1, 2)) / 6
        area = (torch.cross(b - a, c - a, dim=-1) * okf[..., None]).sum((1, 2)) / 2
        n = area / area.norm(dim=1, keepdim=True).clamp(min=1e-12)
        r = (w * (X - o[:, None]).norm(dim=2)).sum(1) / n_cross
        up = (p > self.thr).to(V.dtype)
        h_v = ((V - o[:, None]) * n[:, None]).sum(2)
        h_x = ((X - o[:, None]) * n[:, None]).sum(2)
        h_all = torch.cat([h_v, h_x], 1)
        m_all = torch.cat([up, w], 1)
        tau = (self.tau_frac * r).clamp(min=1e-6)[:, None]
        logit = torch.where(m_all > 0, h_all / tau, torch.full_like(h_all, -1e9))
        height = (torch.softmax(logit, 1) * h_all).sum(1)                       # smooth max, <= max
        return dict(volume=volume, height=height, neck_radius=r, aspect_ratio=height / r.clamp(min=1e-12),
                    neck_centre=o, neck_normal=n, rim_points=cross.sum(1))
