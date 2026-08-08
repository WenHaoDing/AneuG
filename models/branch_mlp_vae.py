"""
Simple conditional MLP-VAE over per-branch Fourier coefficients.

Companion to dataset.skeleton_dataset_fourier.VesselSkeletonDatasetFourier.
Instead of generating a point sequence (see models.branch_transformer), the model
predicts, per branch:
    branch_vector  [3]      the chord  end - start  (start is conditioned)
    fourier_coeffs [k, 3]   sine-series coefficients of the chord-relative shape
A branch is then reconstructed as
    p(s) = start + branch_vector * (s/π) + Σ_n coeffs[n] · sin(n·s),   s ∈ [0, π].

Design
------
* The phi conditioning reuses branch_transformer's encoders (GHDTokenEncoder for
  the simple variant, GCNMeshEncoder for the GCN variant) → a low-dim ghd_embed.
* Everything after that is plain MLP — the target is a tiny fixed-size vector
  (3 + 3k per branch), so Conv1d/transformer machinery buys nothing.
* One latent z per sample encodes the joint multi-branch shape; per-branch
  decoding is conditioned on z + the branch's geometry (slot, start, direction)
  and the global condition (type, scale, ghd_embed).
* A per-branch Bernoulli presence head predicts whether each *candidate opening*
  (the openings that exist for the aneurysm type) actually carries a branch, so
  the model can leave openings empty (e.g. a bifurcated case with only 2 real
  branches). Two masks are involved at training time:
    opening_mask  — candidate openings (b < n_open(type)); the presence head's
                    support. At generation this is the GHD openings mask.
    branch_mask   — branches actually present (the BCE target + recon support).

Normalization (coeffs and chord) is owned by the model via buffers set with
set_normalization_stats(); targets are normalized for the encoder/loss and
outputs are denormalized in sample().
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.branch_transformer import MultiBranchConditions, GHDTokenEncoder, GCNMeshEncoder


def _mlp(dims, act=nn.ReLU, last_act=False):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or last_act:
            layers.append(act())
    return nn.Sequential(*layers)


class BranchCondEmbedding(nn.Module):
    """Per-branch condition → [B, max_branches, cond_dim].

    Combines per-branch geometry (slot id, start point, outward direction) with
    the global condition (aneurysm type, scale, ghd_embed).
    """

    def __init__(self, cond_dim, num_types=3, ghd_dim=16, max_branches=3, part_dim=16):
        super().__init__()
        self.slot_embed = nn.Embedding(max_branches, part_dim)
        self.start_proj = nn.Linear(3, part_dim)
        self.dir_proj   = nn.Linear(3, part_dim)
        self.type_embed = nn.Embedding(num_types, part_dim)
        self.scale_proj = nn.Linear(1, part_dim)
        # slot + start + dir + type + scale = 5 part_dim pieces, plus ghd_embed
        self.out_proj   = nn.Linear(part_dim * 5 + ghd_dim, cond_dim)

    def forward(self, cond):
        B, mb = cond.start_points.shape[:2]
        dev = cond.start_points.device
        slot = self.slot_embed(torch.arange(mb, device=dev)).unsqueeze(0).expand(B, -1, -1)  # [B, mb, p]
        start = self.start_proj(cond.start_points)                                            # [B, mb, p]
        dirv  = self.dir_proj(cond.branch_direction)                                          # [B, mb, p]
        glob = torch.cat([
            self.type_embed(cond.aneurysm_type.long()),     # [B, p]
            self.scale_proj(cond.scale.unsqueeze(-1)),       # [B, p]
            cond.ghd_embed,                                  # [B, ghd_dim]
        ], dim=-1).unsqueeze(1).expand(-1, mb, -1)           # [B, mb, 2p + ghd_dim]
        return self.out_proj(torch.cat([slot, start, dirv, glob], dim=-1))                    # [B, mb, cond_dim]


class BranchFourierVAE(nn.Module):
    def __init__(self, k=8, hidden_dim=128, latent_dim=16, num_layers=2,
                 num_types=3, max_branches=3, num_coeffs=144, ghd_dim=16, cond_dim=64):
        super().__init__()
        self.k = k
        self.latent_dim = latent_dim
        self.max_branches = max_branches
        self.target_dim = 3 + 3 * k                          # chord(3) + coeffs(k×3)

        self.ghd_encoder = GHDTokenEncoder(ghd_dim, num_coeffs)
        self.cond_embed  = BranchCondEmbedding(cond_dim, num_types, ghd_dim, max_branches)

        self.enc    = _mlp([self.target_dim + cond_dim] + [hidden_dim] * num_layers, last_act=True)
        self.fc_mu  = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        self.dec    = _mlp([latent_dim + cond_dim] + [hidden_dim] * num_layers + [self.target_dim])
        # per-branch presence logit: z + branch condition → P(branch exists)
        self.presence_head = _mlp([latent_dim + cond_dim, hidden_dim, 1])

        # target normalization stats — set via set_normalization_stats() before training
        self.register_buffer("coeff_mean", torch.zeros(k, 3))
        self.register_buffer("coeff_std",  torch.ones(k, 3))
        self.register_buffer("vec_mean",   torch.zeros(3))
        self.register_buffer("vec_std",    torch.ones(3))

    # ── normalization ────────────────────────────────────────────────────────
    def set_normalization_stats(self, coeff_mean, coeff_std, vec_mean, vec_std):
        self.coeff_mean.copy_(coeff_mean); self.coeff_std.copy_(coeff_std)
        self.vec_mean.copy_(vec_mean);     self.vec_std.copy_(vec_std)

    def _norm_target(self, branch_vector, fourier_coeffs):
        v = (branch_vector - self.vec_mean) / self.vec_std                       # [B, mb, 3]
        c = (fourier_coeffs - self.coeff_mean) / self.coeff_std                  # [B, mb, k, 3]
        return torch.cat([v, c.flatten(start_dim=2)], dim=-1)                    # [B, mb, target_dim]

    def _denorm_out(self, out):
        v = out[..., :3] * self.vec_std + self.vec_mean                          # [B, mb, 3]
        c = out[..., 3:].reshape(*out.shape[:-1], self.k, 3) * self.coeff_std + self.coeff_mean
        return v, c

    @staticmethod
    def initial_tangent(branch_vector, coeffs):
        """Unit tangent of the reconstructed branch at the start (s=0).

        p(s) = start + branch_vector·(s/π) + Σ_n coeffs[n]·sin(n·s)
        dp/ds|_{s=0} = branch_vector/π + Σ_{n=1..k} n·coeffs[n]

        branch_vector: [..., 3]   coeffs: [..., k, 3]   →   unit dir [..., 3]
        (use physical / denormalized inputs).
        """
        k = coeffs.shape[-2]
        n = torch.arange(1, k + 1, device=coeffs.device, dtype=coeffs.dtype)     # [k]
        deriv = branch_vector / math.pi + (coeffs * n[:, None]).sum(dim=-2)       # [..., 3]
        return F.normalize(deriv, dim=-1)

    # ── phi conditioning (overridden by the GCN variant) ──────────────────────
    def encode_phi(self, phi, aneurysm_types=None):
        return self.ghd_encoder(phi)                                            # [B, ghd_dim]

    # ── core ─────────────────────────────────────────────────────────────────
    @staticmethod
    def reparameterize(mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def _encode(self, target, cond_emb, branch_mask):
        h = self.enc(torch.cat([target, cond_emb], dim=-1))                     # [B, mb, hidden]
        m = branch_mask.unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)                        # [B, hidden]  masked mean
        logvar = self.fc_var(pooled).clamp(-10.0, 10.0)
        return self.fc_mu(pooled), logvar

    def _decode(self, z, cond_emb):
        z_exp = z.unsqueeze(1).expand(-1, cond_emb.size(1), -1)                  # [B, mb, latent]
        return self.dec(torch.cat([z_exp, cond_emb], dim=-1))                    # [B, mb, target_dim]

    def _decode_presence(self, z, cond_emb):
        z_exp = z.unsqueeze(1).expand(-1, cond_emb.size(1), -1)                  # [B, mb, latent]
        return self.presence_head(torch.cat([z_exp, cond_emb], dim=-1)).squeeze(-1)  # [B, mb] logits

    def forward(self, branch_vector, fourier_coeffs, phi, cond):
        cond     = cond._replace(ghd_embed=self.encode_phi(phi, cond.aneurysm_type))
        cond_emb = self.cond_embed(cond)
        target   = self._norm_target(branch_vector, fourier_coeffs)
        mu, logvar = self._encode(target, cond_emb, cond.branch_mask)
        z   = self.reparameterize(mu, logvar)
        out = self._decode(z, cond_emb)
        presence_logit = self._decode_presence(z, cond_emb)
        return out, presence_logit, mu, logvar

    def get_loss(self, out, branch_vector, fourier_coeffs, branch_mask, opening_mask,
                 presence_logit, mu, logvar):
        target = self._norm_target(branch_vector, fourier_coeffs)               # [B, mb, target_dim]
        # shape losses: only over branches actually present (branch_mask)
        m = branch_mask.unsqueeze(-1).float()
        denom = m.sum().clamp(min=1.0)
        diff2 = (out - target) ** 2 * m
        vec_loss   = diff2[..., :3].sum()  / (denom * 3)
        coeff_loss = diff2[..., 3:].sum() / (denom * (self.target_dim - 3))
        # presence loss: BCE over candidate openings (opening_mask), target = present
        om = opening_mask.float()
        bce = F.binary_cross_entropy_with_logits(presence_logit, branch_mask.float(), reduction="none")
        presence_loss = (bce * om).sum() / om.sum().clamp(min=1.0)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)
        return coeff_loss, vec_loss, presence_loss, kl_loss

    @torch.no_grad()
    def sample(self, phi, cond, z=None):
        """Returns (branch_vector [B, mb, 3], coeffs [B, mb, k, 3], presence [B, mb]).

        presence is P(branch exists) ∈ (0,1) per branch slot; the caller masks it
        against the candidate openings and thresholds (e.g. > 0.5) to drop branches.
        """
        cond     = cond._replace(ghd_embed=self.encode_phi(phi, cond.aneurysm_type))
        cond_emb = self.cond_embed(cond)
        if z is None:
            z = torch.randn(phi.size(0), self.latent_dim, device=phi.device)
        out = self._decode(z, cond_emb)
        presence = torch.sigmoid(self._decode_presence(z, cond_emb))
        vec, coeffs = self._denorm_out(out)
        return vec, coeffs, presence


class BranchFourierVAE_GCNConditioner(BranchFourierVAE):
    """BranchFourierVAE with a GCN mesh encoder (reconstructs the mesh from phi
    via MultiCanonicalGHDReconstruct, then encodes it) instead of GHDTokenEncoder."""

    def __init__(self, multi_recon, k=8, hidden_dim=128, latent_dim=16, num_layers=2,
                 num_types=3, max_branches=3, num_coeffs=144, ghd_dim=16, cond_dim=64,
                 gcn_hidden=32, gcn_pool_ratio=0.5):
        super().__init__(k, hidden_dim, latent_dim, num_layers,
                         num_types, max_branches, num_coeffs, ghd_dim, cond_dim)
        self.multi_recon = multi_recon
        self.ghd_encoder = GCNMeshEncoder(ghd_dim, in_channels=3,
                                          hidden=gcn_hidden, pool_ratio=gcn_pool_ratio)

    def encode_phi(self, phi, aneurysm_types):
        # point_std=None → mesh verts stay in physical GHD space (same frame as
        # start_points / branch_vector targets).
        pyg_batch = self.multi_recon.to_pyg_batch(phi, aneurysm_types, point_std=None)
        return self.ghd_encoder(pyg_batch)
