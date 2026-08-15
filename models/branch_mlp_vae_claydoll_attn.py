"""
Branch Fourier VAE — unclipped-centerline variant with local cross-attention.

Parallel to models/branch_mlp_vae_claydoll.py (which conditions every branch on the
same single global-pooled ghd_embed, differentiated only by a learned slot
embedding and the predicted start point). This variant additionally lets each
branch attend over an intermediate GCN pooling level — the pool1-level node
features, each concatenated with that node's *original* mesh position — so a
branch's conditioning can reflect the local geometry near its own predicted
start point, not just one global summary shared by every branch.

Design (see models/branch_transformer.py's SlotQueryCrossAttention docstring
for the reusable half of this):
  - GCNMeshEncoderWithLocalAttention (this file, GCN-specific) builds one
    query per branch slot from a learned slot embedding + the predicted start
    point (from the existing start_points_head), and gathers pool1-level
    tokens: [x1 (pool1 features), x[perm1] (original position of each
    surviving node, recovered via SAGPooling's permutation)].
  - SlotQueryCrossAttention (models/branch_transformer.py, generic/reusable)
    does the actual masked multi-head attention: per-slot query attends over
    the padded token set (torch_geometric.utils.to_dense_batch handles the
    ragged-batch padding/masking, since pool1 node counts vary per graph,
    especially across a batch mixing bifurcated/sidewall aneurysm types).
  - The result, local_embed [B, mb, local_dim], is added to
    BranchCondEmbeddingClaydollAttn as a fifth per-slot conditioning piece.

Kept as a separate file/class from branch_mlp_vae_claydoll.py (not a flag on it) so
the simpler global-pooling variant stays exactly as validated, and the two can
be compared directly.

conda activate new
"""

import torch
import torch.nn as nn

from models.branch_mlp_vae_claydoll import BranchFourierVAE_GCNConditionerClaydoll
from models.branch_transformer import (
    BranchConditionsClaydollAttn,
    GCNMeshEncoderWithLocalAttention,
)


class BranchCondEmbeddingClaydollAttn(nn.Module):
    """Per-branch condition → [B, max_branches, cond_dim].

    Same as models.branch_mlp_vae_claydoll.BranchCondEmbeddingClaydoll, plus a fifth
    per-slot piece: the local-attention readout from GCNMeshEncoderWithLocalAttention.
    """

    def __init__(self, cond_dim, num_types=3, ghd_dim=16, max_branches=3, part_dim=16, local_dim=32):
        super().__init__()
        self.slot_embed = nn.Embedding(max_branches, part_dim)
        self.start_proj = nn.Linear(3, part_dim)
        self.local_proj = nn.Linear(local_dim, part_dim)
        self.type_embed = nn.Embedding(num_types, part_dim)
        self.scale_proj = nn.Linear(1, part_dim)
        # slot + start + local + type + scale = 5 part_dim pieces, plus ghd_embed
        self.out_proj   = nn.Linear(part_dim * 5 + ghd_dim, cond_dim)

    def forward(self, cond):
        B, mb = cond.start_points.shape[:2]
        dev = cond.start_points.device
        slot  = self.slot_embed(torch.arange(mb, device=dev)).unsqueeze(0).expand(B, -1, -1)  # [B, mb, p]
        start = self.start_proj(cond.start_points)                                            # [B, mb, p]
        local = self.local_proj(cond.local_embed)                                              # [B, mb, p]
        glob = torch.cat([
            self.type_embed(cond.aneurysm_type.long()),     # [B, p]
            self.scale_proj(cond.scale.unsqueeze(-1)),       # [B, p]
            cond.ghd_embed,                                  # [B, ghd_dim]
        ], dim=-1).unsqueeze(1).expand(-1, mb, -1)           # [B, mb, 2p + ghd_dim]
        return self.out_proj(torch.cat([slot, start, local, glob], dim=-1))                   # [B, mb, cond_dim]


class BranchFourierVAE_GCNConditionerClaydollAttn(BranchFourierVAE_GCNConditionerClaydoll):
    """BranchFourierVAE_GCNConditionerClaydoll with local cross-attention conditioning
    instead of a single global-pooled ghd_embed broadcast to every branch."""

    def __init__(self, multi_recon, k=8, hidden_dim=128, latent_dim=16, num_layers=2,
                 num_types=3, max_branches=3, num_coeffs=144, ghd_dim=16, cond_dim=64,
                 gcn_hidden=32, gcn_pool_ratio=0.5, local_dim=32, attn_heads=4):
        super().__init__(multi_recon, k, hidden_dim, latent_dim, num_layers,
                         num_types, max_branches, num_coeffs, ghd_dim, cond_dim,
                         gcn_hidden, gcn_pool_ratio)
        # Replace the base variant's non-attention encoder/conditioning modules.
        self.ghd_encoder = GCNMeshEncoderWithLocalAttention(
            ghd_dim, max_branches, in_channels=3, hidden=gcn_hidden, pool_ratio=gcn_pool_ratio,
            local_dim=local_dim, attn_heads=attn_heads,
        )
        self.cond_embed = BranchCondEmbeddingClaydollAttn(
            cond_dim, num_types, ghd_dim, max_branches, local_dim=local_dim,
        )

    def encode_phi(self, phi, aneurysm_types):
        """Returns (ghd_embed, start_points_pred, local_embed)."""
        pyg_batch = self.multi_recon.to_pyg_batch(phi, aneurysm_types, point_std=None)
        return self.ghd_encoder(pyg_batch)

    def forward(self, branch_vector, fourier_coeffs, phi, cond):
        ghd_embed, start_points_pred, local_embed = self.encode_phi(phi, cond.aneurysm_type)
        cond = cond._replace(ghd_embed=ghd_embed, start_points=start_points_pred, local_embed=local_embed)
        cond_emb = self.cond_embed(cond)
        target = self._norm_target(branch_vector, fourier_coeffs)
        mu, logvar = self._encode(target, cond_emb, cond.branch_mask)
        z = self.reparameterize(mu, logvar)
        out = self._decode(z, cond_emb)
        presence_logit = self._decode_presence(z, cond_emb)
        return out, presence_logit, mu, logvar, start_points_pred

    @torch.no_grad()
    def sample(self, phi, cond, z=None):
        ghd_embed, start_points_pred, local_embed = self.encode_phi(phi, cond.aneurysm_type)
        cond = cond._replace(ghd_embed=ghd_embed, start_points=start_points_pred, local_embed=local_embed)
        cond_emb = self.cond_embed(cond)
        if z is None:
            z = torch.randn(phi.size(0), self.latent_dim, device=phi.device)
        out = self._decode(z, cond_emb)
        presence = torch.sigmoid(self._decode_presence(z, cond_emb))
        vec, coeffs = self._denorm_out(out)
        return vec, coeffs, presence, start_points_pred
