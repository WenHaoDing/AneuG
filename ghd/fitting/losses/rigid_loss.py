"""
ARAP (As-Rigid-As-Possible) Loss
Ported from GHDHeart/losses/mesh_loss.py::Rigid_Loss

Implements cotangent-weighted elastic ARAP using pre-computed dense
neighbourhood tensors (no torch_geometric required).

Algorithmic equivalence to GHDHeart/losses/rigid_deform.py::RigidLoss:
  - Both normalise edge vectors by sqrt(trace(X^T X) / 2) per vertex (elastic)
  - Both use cotangent weights and SVD for optimal rotation
  - rigid_deform.py uses MessagePassing; this uses batched dense matmul
  - Numerically equivalent; this version avoids the (V × E) sparse→dense
    materialisation in one_hot_sparse and is significantly faster
"""
import torch
import torch.nn as nn
from pytorch3d.ops import cot_laplacian


class GHDRigidLoss(nn.Module):
    """
    Cotangent-weighted local Procrustes rigidity loss (ARAP).

    For each vertex neighbourhood, finds the best rigid rotation mapping
    source edge vectors to deformed edge vectors (SVD), then penalises
    the residual.

    Parameters
    ----------
    verts_src  : (N, 3) source / template vertex positions
    faces_src  : (M, 3) source face indices
    if_elastic : bool — normalise edge vectors by local scale (elastic ARAP).
                 True by default, matches GHDHeart behaviour.

    GHDHeart: losses/mesh_loss.py::Rigid_Loss
              losses/rigid_deform.py::RigidLoss
    """

    def __init__(
        self,
        verts_src:  torch.Tensor,
        faces_src:  torch.Tensor,
        if_elastic: bool = True,
    ) -> None:
        super().__init__()
        self.if_elastic = if_elastic

        cot_w    = cot_laplacian(verts_src, faces_src.long())[0].coalesce()
        edges    = cot_w.indices()
        vals     = cot_w.values()

        edges_ud = torch.cat([edges, edges.flip(0)], dim=1)
        vals_ud  = torch.cat([vals,  vals],           dim=0)

        Vn      = verts_src.shape[0]
        deg     = torch.bincount(edges_ud[0], minlength=Vn)
        max_deg = int(deg.max().item())

        neigh_idx = torch.empty((Vn, max_deg), dtype=torch.long)
        neigh_w   = torch.zeros((Vn, max_deg), dtype=verts_src.dtype)

        buckets = [[] for _ in range(Vn)]
        for (i, j), wij in zip(edges_ud.t().tolist(), vals_ud.tolist()):
            buckets[i].append((j, wij))
        for i in range(Vn):
            items = buckets[i]
            if len(items) < max_deg:
                items = items + [(i, 0.0)] * (max_deg - len(items))
            neigh_idx[i] = torch.tensor([it[0] for it in items[:max_deg]], dtype=torch.long)
            neigh_w[i]   = torch.tensor([it[1] for it in items[:max_deg]],
                                         dtype=verts_src.dtype)

        dev = verts_src.device
        self.register_buffer("neigh_idx", neigh_idx.to(dev))
        self.register_buffer("neigh_w",   neigh_w.to(dev))
        src_neigh = verts_src[neigh_idx.to(dev)]
        self.register_buffer("src_rel",   src_neigh - verts_src.unsqueeze(1))

    def forward(
        self,
        new_verts:  torch.Tensor,       # (N, 3) deformed vertex positions
        if_elastic: bool | None = None, # override constructor default if needed
    ) -> torch.Tensor:
        elastic  = if_elastic if if_elastic is not None else self.if_elastic
        src_rel  = self.src_rel
        trg_rel  = new_verts[self.neigh_idx] - new_verts.unsqueeze(1)

        if elastic:
            def _normalise(rel: torch.Tensor) -> torch.Tensor:
                trace = torch.diagonal(
                    torch.bmm(rel.transpose(1, 2), rel), dim1=-2, dim2=-1
                ).sum(-1, keepdim=True) / 2.0
                return rel / (trace.clamp_min(1e-12).sqrt().unsqueeze(-1) + 1e-6)
            src_rel = _normalise(src_rel)
            trg_rel = _normalise(trg_rel)

        XtY  = torch.bmm(src_rel.transpose(1, 2), trg_rel)
        eye  = torch.eye(3, device=new_verts.device, dtype=new_verts.dtype).unsqueeze(0)
        U, _, Vh = torch.linalg.svd(XtY + 1e-6 * eye, full_matrices=False)
        R_iT = Vh.transpose(-2, -1).matmul(U.transpose(-2, -1))
        y_hat = src_rel.matmul(R_iT)

        diff = (y_hat - trg_rel).norm(dim=-1)
        return (diff * self.neigh_w).sum() / (self.neigh_w.sum() + 1e-6)
