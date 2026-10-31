"""Ostium model (1D graph -> site distribution + ostium code) and complex model (-> GHD latent z +
affine in the ostium frame). Design: TASK_vessel_conditioned_aneurysm.md section 5.6.

Batches come from dataset/ostium_dataset.py (PyG batches with coarse levels).

GraphEncoder        level 0: GINE convolutions on the full graph (edge feature = displacement)
                    level 1, 2: pool every finer point into its seed (geodesic Voronoi, from the
                    dataset), add positional encodings (Laplacian eigenvectors + eigenvalues,
                    random-walk), GPS blocks (local GINE + full multi-head attention)
                    up: broadcast coarse features back along the graph (the same geodesic
                    assignment), concatenate the skip, GINE convolutions -> H; global g pooled
                    from level 2
SiteHead            per-point logit; loss = soft cross-entropy against the dataset's soft label
OstiumHead          33-number code y given (H[P'], g, signed tangent, local radius):
                    conditional flow matching (velocity MLP) or a one-shot MLP (mode='mlp')
ComplexHead         26-number target x = [z, affine] given graph tokens near the site and the
                    ostium tokens (decoded ring in the ostium frame + a code token): transformer ->
                    condition -> flow matching or MLP; plus an aneurysm-type classifier (the VAE
                    decoder needs the type)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, GPSConv
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.utils import scatter, softmax


def mlp(i, h, o, n=2, act=nn.SiLU):
    layers, d = [], i
    for _ in range(n - 1):
        layers += [nn.Linear(d, h), act()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


def time_embedding(t, dim=32):
    half = dim // 2
    f = torch.exp(-math.log(1000.0) * torch.arange(half, device=t.device) / half)
    a = t[:, None] * f[None] * 1000.0
    return torch.cat([a.sin(), a.cos()], 1)


class GINEBlock(nn.Module):
    def __init__(self, d, edge_dim):
        super().__init__()
        self.conv = GINEConv(mlp(d, d, d), edge_dim=edge_dim)
        self.norm = nn.LayerNorm(d)

    def forward(self, x, ei, ea):
        return self.norm(x + self.conv(x, ei, ea))


def edge_attr(pos, ei):
    dv = pos[ei[1]] - pos[ei[0]]
    return torch.cat([dv, dv.norm(dim=1, keepdim=True)], 1)


# ------------------------------------------------------------------------------ encoder

class GraphEncoder(nn.Module):
    def __init__(self, in_dim=12, d=128, l0_layers=3, gps_layers=(2, 2), heads=4, pe_k=8, rw=8,
                 up_layers=2, dropout=0.0):
        super().__init__()
        self.inp = mlp(in_dim, d, d)
        self.l0 = nn.ModuleList([GINEBlock(d, 4) for _ in range(l0_layers)])
        self.pool = nn.ModuleList([mlp(d + 3, d, d), mlp(d + 3, d, d)])
        self.pe = nn.ModuleList([mlp(2 * pe_k + rw, d, d), mlp(2 * pe_k + rw, d, d)])
        self.gps = nn.ModuleList([
            nn.ModuleList([GPSConv(d, GINEConv(mlp(d, d, d), edge_dim=4), heads=heads, dropout=dropout,
                                   attn_type="multihead") for _ in range(n)])
            for n in gps_layers])
        self.up = nn.ModuleList([mlp(2 * d, d, d), mlp(2 * d, d, d)])
        self.up_conv = nn.ModuleList([nn.ModuleList([GINEBlock(d, 4) for _ in range(up_layers)]) for _ in range(2)])
        self.gpool = AttentionalAggregation(mlp(d, d, 1))
        self.gout = mlp(2 * d, d, d)

    def forward(self, b):
        x0 = self.inp(b.x)
        ea0 = edge_attr(b.pos, b.edge_index)
        for blk in self.l0:
            x0 = blk(x0, b.edge_index, ea0)
        feats, poss, bats, eis = [x0], [b.pos], [b.batch], [b.edge_index]
        lv = [(b.assign_01, b.pos_1, b.pos_1_batch, b.edge_index_1, b.pe_1, b.eig_1, b.rw_1),
              (b.assign_12, b.pos_2, b.pos_2_batch, b.edge_index_2, b.pe_2, b.eig_2, b.rw_2)]
        x = x0
        for k, (assign, pos, bat, ei, pe, eig, rw) in enumerate(lv):
            x = scatter(x, assign, dim=0, dim_size=pos.size(0), reduce="mean")
            x = self.pool[k](torch.cat([x, pos], 1)) + self.pe[k](torch.cat([pe, eig, rw], 1))
            ea = edge_attr(pos, ei)
            for layer in self.gps[k]:
                x = layer(x, ei, bat, edge_attr=ea)
            feats.append(x)
            poss.append(pos)
            bats.append(bat)
            eis.append(ei)
        g = self.gout(torch.cat([self.gpool(feats[2], bats[2]),
                                 scatter(feats[2], bats[2], dim=0, reduce="mean")], 1))
        # up: level 2 -> 1 -> 0, broadcast along the graph (geodesic assignment) + skip + convs
        up = feats[2]
        for k, assign in ((1, b.assign_12), (0, b.assign_01)):
            up = self.up[k](torch.cat([up[assign], feats[k]], 1))
            ea = edge_attr(poss[k], eis[k])
            for blk in self.up_conv[k]:
                up = blk(up, eis[k], ea)
        return dict(H=up, g=g, F1=feats[1], pos1=poss[1], batch1=bats[1])


# ------------------------------------------------------------------------------ heads

# log of the local radius s (units of 10 mm) over the dataset, for --norm_tokens
LOG_S_MEAN, LOG_S_STD = -2.1, 0.3


class VelocityMLP(nn.Module):
    """v(x_t, t, c) for flow matching, or (mode 'mlp') a direct predictor x = f(c)."""

    def __init__(self, x_dim, c_dim, h=256, n_blocks=4, mode="flow"):
        super().__init__()
        self.mode, self.x_dim = mode, x_dim
        i = c_dim + (x_dim + 32 if mode == "flow" else 0)
        self.inp = nn.Linear(i, h)
        self.blocks = nn.ModuleList([mlp(h, h, h) for _ in range(n_blocks)])
        self.norms = nn.ModuleList([nn.LayerNorm(h) for _ in range(n_blocks)])
        self.out = nn.Linear(h, x_dim)

    def forward(self, c, x_t=None, t=None):
        z = c if self.mode == "mlp" else torch.cat([c, x_t, time_embedding(t)], 1)
        h = self.inp(z)
        for blk, nrm in zip(self.blocks, self.norms):
            h = h + blk(nrm(h))
        return self.out(h)

    # ---- flow matching
    def fm_loss(self, c, x1):
        """Returns (loss, one-step estimate of x1, t)."""
        if self.mode == "mlp":
            pred = self(c)
            return F.mse_loss(pred, x1), pred, torch.ones(len(x1), device=x1.device)
        x0 = torch.randn_like(x1)
        t = torch.rand(len(x1), device=x1.device)
        xt = (1 - t[:, None]) * x0 + t[:, None] * x1
        v = self(c, xt, t)
        x1_hat = xt + (1 - t[:, None]) * v
        return F.mse_loss(v, x1 - x0), x1_hat, t

    @torch.no_grad()
    def sample(self, c, steps=32):
        if self.mode == "mlp":
            return self(c)
        x = torch.randn(len(c), self.x_dim, device=c.device)
        for k in range(steps):
            t = torch.full((len(c),), k / steps, device=c.device)
            x = x + self(c, x, t) / steps
        return x


def ring_from_code(y, H=5, n=32):
    """Destandardised 33-vector -> local ring points (B, n, 3) in units of s (ostium frame)."""
    u = torch.arange(n, device=y.device, dtype=y.dtype) / n
    k = torch.arange(1, H + 1, device=y.device, dtype=y.dtype)
    ab = y[:, 3:3 + 6 * H].reshape(-1, H, 6)
    a, bb = ab[..., :3], ab[..., 3:]
    cos, sin = torch.cos(2 * math.pi * k[None, :, None] * u[None, None]), torch.sin(2 * math.pi * k[None, :, None] * u[None, None])
    return torch.einsum("bkc,ku->buc", a, cos[0]) + torch.einsum("bkc,ku->buc", bb, sin[0])


def frame_from_code(offset, tangent):
    """Ostium frame rows (e1, e2, n) from the predicted offset (n = its direction) and the tangent."""
    n = F.normalize(offset, dim=1)
    e1 = F.normalize(tangent - (tangent * n).sum(1, keepdim=True) * n, dim=1)
    return torch.stack([e1, torch.cross(n, e1, dim=1), n], 1)


class OstiumModel(nn.Module):
    """Encoder + site head + ostium head + complex head."""

    def __init__(self, d=128, mode="flow", H=5, gps_layers=(2, 2), heads=4, n_graph_tokens=32,
                 cx_layers=2, with_complex=True, y_dim=33, x_dim=26, norm_tokens=False, flow_h=256, flow_blocks=4):
        super().__init__()
        self.H, self.n_graph_tokens, self.with_complex = H, n_graph_tokens, with_complex
        # norm_tokens: graph-token positions as [direction, log(1 + distance)] instead of raw rel
        # (raw |rel| ~ 8, up to ~80, in units of s), and log(s) centred with the dataset statistics
        self.norm_tokens = norm_tokens
        self.enc = GraphEncoder(d=d, gps_layers=gps_layers, heads=heads)
        self.site = mlp(2 * d, d, 1, n=3)
        self.ost_cond = mlp(2 * d + 4, d, d)
        # flow_h / flow_blocks: width and depth of the ostium and complex heads (VelocityMLP); the
        # defaults are the size every earlier checkpoint was trained with
        self.ost = VelocityMLP(y_dim, d, h=flow_h, n_blocks=flow_blocks, mode=mode)
        if with_complex:
            self.ring_tok = mlp(3 + 2, d, d)
            self.code_tok = mlp(y_dim, d, d)
            self.graph_tok = mlp(d + 4, d, d)
            self.cls = nn.Parameter(torch.zeros(1, 1, d))
            self.kind = nn.Embedding(4, d)          # cls / ring / code / graph
            self.cx_tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, 2 * d, batch_first=True,
                                                                          norm_first=True), cx_layers)
            self.cx_cond = mlp(2 * d, d, d)
            self.type_head = mlp(d, d, 3)            # p(type | graph, site, ostium), from c2
            self.type_emb = nn.Embedding(3, d)       # the complex flow is conditioned on the type
            self.cx = VelocityMLP(x_dim, d, h=flow_h, n_blocks=flow_blocks, mode=mode)

    # ---- conditions (sampling order: site -> ostium -> type -> complex)
    def site_logits(self, enc, b):
        return self.site(torch.cat([enc["H"], enc["g"][b.batch]], 1)).squeeze(1)

    def ostium_condition(self, enc, b, site_idx, tangent=None, scale=None):
        """tangent/scale default to the batch's (at the true site); pass them for a sampled site."""
        tangent = b.tangent if tangent is None else tangent
        log_s = torch.log(b.scale if scale is None else scale)
        if self.norm_tokens:
            log_s = (log_s - LOG_S_MEAN) / LOG_S_STD
        return self.ost_cond(torch.cat([enc["H"][site_idx], enc["g"], tangent, log_s], 1))

    def complex_condition(self, enc, b, y_std, y_raw, site_idx, tangent=None, scale=None):
        """y_std: standardised code (token); y_raw: destandardised code (geometry). tangent/scale
        default to the batch's (at the true site); pass them for another site."""
        tangent = b.tangent if tangent is None else tangent
        scale = b.scale if scale is None else scale
        B, dev = y_raw.size(0), y_raw.device
        ring = ring_from_code(y_raw, self.H, 32)                                    # (B, 32, 3), local
        u = torch.arange(32, device=dev) / 32.0
        ue = torch.stack([torch.cos(2 * math.pi * u), torch.sin(2 * math.pi * u)], 1)[None].expand(B, -1, -1)
        rt = self.ring_tok(torch.cat([ring, ue], 2)) + self.kind.weight[1]
        ct = (self.code_tok(y_std) + self.kind.weight[2])[:, None]
        # graph tokens: level-1 tokens nearest the ostium centre, positions in the ostium frame
        Rf = frame_from_code(y_raw[:, :3], tangent)                                # (B, 3, 3)
        C = b.pos[site_idx] + y_raw[:, :3] * scale
        F1, P1, B1 = enc["F1"], enc["pos1"], enc["batch1"]
        rel = torch.einsum("nij,nj->ni", Rf[B1], P1 - C[B1]) / scale[B1]
        gt = []
        for i in range(B):
            m = torch.nonzero(B1 == i).squeeze(1)
            k = min(self.n_graph_tokens, len(m))
            near = m[torch.topk(-rel[m].norm(dim=1), k).indices]
            r = rel[near].norm(dim=1, keepdim=True)
            geo = (torch.cat([rel[near] / (r + 1e-6), torch.log1p(r)], 1) if self.norm_tokens
                   else torch.cat([rel[near], r], 1))
            tok = self.graph_tok(torch.cat([F1[near], geo], 1))
            if k < self.n_graph_tokens:
                tok = torch.cat([tok, tok.new_zeros(self.n_graph_tokens - k, tok.size(1))])
            gt.append(tok)
        gt = torch.stack(gt) + self.kind.weight[3]
        seq = torch.cat([self.cls.expand(B, -1, -1) + self.kind.weight[0], ct, rt, gt], 1)
        h = self.cx_tf(seq)[:, 0]
        return self.cx_cond(torch.cat([h, enc["g"]], 1))

    def with_type(self, c2, atype):
        """Complex-flow condition: c2 plus the aneurysm type (true type in training, a sample from
        type_head(c2) at inference). The type head itself sees c2 only."""
        return c2 + self.type_emb(atype)


class MorphOstiumModel(OstiumModel):
    """OstiumModel conditioned on the dome's morphology: m = [log volume (mm3), aspect ratio]
    (models/dome_markers.py; labels from scripts/ostium/morph_reference.py). The standardised m is
    embedded and added to the ostium condition (so the neck can follow the requested size) and to
    the complex condition (so the type and the dome follow it). m defaults to the batch's own
    (b.morph, the case's real markers); pass morph= for a requested one (raw units)."""

    def __init__(self, morph_mean=(0.0, 0.0), morph_std=(1.0, 1.0), **kw):
        super().__init__(**kw)
        d = self.ost_cond[-1].out_features
        self.register_buffer("morph_mean", torch.tensor(morph_mean, dtype=torch.float32))
        self.register_buffer("morph_std", torch.tensor(morph_std, dtype=torch.float32))
        self.morph_ost = mlp(2, d, d)
        if self.with_complex:
            self.morph_cx = mlp(2, d, d)

    def morph_std_(self, b, morph):
        m = b.morph if morph is None else morph
        return (m - self.morph_mean) / self.morph_std

    def ostium_condition(self, enc, b, site_idx, tangent=None, scale=None, morph=None):
        return super().ostium_condition(enc, b, site_idx, tangent, scale) + self.morph_ost(self.morph_std_(b, morph))

    def complex_condition(self, enc, b, y_std, y_raw, site_idx, tangent=None, scale=None, morph=None):
        c = super().complex_condition(enc, b, y_std, y_raw, site_idx, tangent, scale)
        return c + self.morph_cx(self.morph_std_(b, morph))
