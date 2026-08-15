"""
Multi-branch curve VAE — unclipped-centerline variant with local cross-attention.

Parallel to models/branch_transformer_claydoll.py (which conditions every
token on a single global-pooled ghd_embed, differentiated only by a learned
slot embedding and the predicted start point). This variant additionally lets
each branch attend over an intermediate GCN pooling level — the pool1-level
node features, each concatenated with that node's *original* mesh position —
so a branch's per-token conditioning can reflect the local geometry near its
own predicted start point, not just one global summary shared by every
branch. Exactly the same idea as models/branch_mlp_vae_claydoll_attn.py,
applied to the point-sequence transformer family instead of the Fourier-MLP
family — reusing the identical GCN-side machinery
(models.branch_transformer.GCNMeshEncoderWithLocalAttention,
BranchConditionsClaydollAttn, SlotQueryCrossAttention); only the consumer side
(how local_embed gets folded into per-token/per-branch conditioning) differs
because the decoder here is a per-token transformer, not a per-branch MLP.

local_embed is added as a third per-branch conditioning piece (alongside slot
identity and predicted start) in both PerTokenBranchEmbeddingClaydollAttn
(injected into every token of that branch) and DecoderClaydollAttn's length
predictor — mirroring how BranchCondEmbeddingClaydollAttn added it as a fifth
piece for the Fourier-MLP family.

Kept as a separate file/class from branch_transformer_claydoll.py (not a flag
on it) so the simpler global-pooling variant stays exactly as validated, and
the two can be compared directly.

conda activate new
"""

import torch
import torch.nn as nn

from models.branch_transformer import (
    Decoder,
    Encoder,
    GCNMeshEncoderWithLocalAttention,
)
from models.branch_transformer_claydoll import MultiBranchVAEClaydollGCNConditioner


class PerTokenBranchEmbeddingClaydollAttn(nn.Module):
    """PerTokenBranchEmbeddingClaydoll plus a third per-slot piece: the
    local-attention readout from GCNMeshEncoderWithLocalAttention."""

    def __init__(self, hidden_dim, max_branches, max_local_points, part_dim=16, local_dim=32):
        super().__init__()
        self.slot_embed = nn.Embedding(max_branches, part_dim)
        self.start_proj = nn.Linear(3, part_dim)
        self.local_proj = nn.Linear(local_dim, part_dim)
        self.out_proj   = nn.Linear(part_dim * 3, hidden_dim)

        branch_ids = torch.arange(max_branches).repeat_interleave(max_local_points)
        self.register_buffer("branch_ids", branch_ids)   # [seq_len]

    def forward(self, cond):
        b_ids   = self.branch_ids                                            # [seq_len]
        B       = cond.start_points.size(0)
        slot_t  = self.slot_embed(b_ids).unsqueeze(0).expand(B, -1, -1)    # [B, seq_len, part_dim]
        start_t = self.start_proj(cond.start_points)[:, b_ids, :]          # [B, seq_len, part_dim]
        local_t = self.local_proj(cond.local_embed)[:, b_ids, :]           # [B, seq_len, part_dim]
        return self.out_proj(torch.cat([slot_t, start_t, local_t], dim=-1))  # [B, seq_len, hidden_dim]


class EncoderClaydollAttn(Encoder):
    """Encoder with PerTokenBranchEmbeddingClaydollAttn instead of
    PerTokenBranchEmbedding. forward() is inherited unchanged."""

    def __init__(self, hidden_dim, latent_dim, num_layers, nhead, seq_len,
                 num_types=3, max_branches=3, max_local_points=127, ghd_dim=16,
                 part_dim=16, local_dim=32):
        super().__init__(hidden_dim, latent_dim, num_layers, nhead, seq_len,
                         num_types, max_branches, max_local_points, ghd_dim)
        self.branch_embed = PerTokenBranchEmbeddingClaydollAttn(
            hidden_dim, max_branches, max_local_points, part_dim, local_dim)


class DecoderClaydollAttn(Decoder):
    """Decoder with PerTokenBranchEmbeddingClaydollAttn and a local-attention-
    aware per-branch length predictor. forward()/sample() are inherited
    unchanged — both route through self.branch_embed and self._length_logits."""

    def __init__(self, hidden_dim, latent_dim, num_layers, nhead, seq_len,
                 num_types=3, max_branches=3, max_local_points=127, ghd_dim=16,
                 part_dim=16, local_dim=32):
        super().__init__(hidden_dim, latent_dim, num_layers, nhead, seq_len,
                         num_types, max_branches, max_local_points, ghd_dim, part_dim)
        self.branch_embed = PerTokenBranchEmbeddingClaydollAttn(
            hidden_dim, max_branches, max_local_points, part_dim, local_dim)
        del self.dir_proj_len
        self.local_proj_len = nn.Linear(local_dim, part_dim)
        self.length_head = nn.Linear(latent_dim + part_dim * 3, max_local_points)

    def _length_logits(self, z, cond):
        # z: [B, latent_dim] → [B, max_branches, max_local_points]
        B         = z.size(0)
        slots     = torch.arange(self.max_branches, device=z.device)
        slot_emb  = self.slot_embed_len(slots).unsqueeze(0).expand(B, -1, -1)   # [B, max_branches, part_dim]
        start_emb = self.start_proj_len(cond.start_points)                       # [B, max_branches, part_dim]
        local_emb = self.local_proj_len(cond.local_embed)                        # [B, max_branches, part_dim]
        branch_feat = torch.cat([slot_emb, start_emb, local_emb], dim=-1)       # [B, max_branches, part_dim*3]
        z_exp     = z.unsqueeze(1).expand(-1, self.max_branches, -1)            # [B, max_branches, latent_dim]
        return self.length_head(torch.cat([z_exp, branch_feat], dim=-1))        # [B, max_branches, max_local_points]


class MultiBranchVAEClaydollAttnGCNConditioner(MultiBranchVAEClaydollGCNConditioner):
    """MultiBranchVAEClaydollGCNConditioner with local cross-attention
    conditioning instead of a single global-pooled ghd_embed broadcast to
    every token."""

    def __init__(self, multi_recon, hidden_dim=64, latent_dim=32, num_layers=4, nhead=4,
                 max_local_points=127, num_types=3, max_branches=3,
                 num_coeffs=144, ghd_dim=16, gcn_hidden=32, gcn_pool_ratio=0.5,
                 local_dim=32, attn_heads=4):
        super().__init__(multi_recon, hidden_dim, latent_dim, num_layers, nhead,
                         max_local_points, num_types, max_branches, num_coeffs, ghd_dim,
                         gcn_hidden, gcn_pool_ratio)
        # Replace the base claydoll variant's non-attention mesh encoder/encoder/decoder.
        seq_len = max_branches * max_local_points
        self.ghd_encoder = GCNMeshEncoderWithLocalAttention(
            ghd_dim, max_branches, in_channels=3, hidden=gcn_hidden, pool_ratio=gcn_pool_ratio,
            local_dim=local_dim, attn_heads=attn_heads,
        )
        self.encoder = EncoderClaydollAttn(hidden_dim, latent_dim, num_layers, nhead, seq_len,
                                           num_types, max_branches, max_local_points, ghd_dim,
                                           local_dim=local_dim)
        self.decoder = DecoderClaydollAttn(hidden_dim, latent_dim, num_layers, nhead, seq_len,
                                           num_types, max_branches, max_local_points, ghd_dim,
                                           local_dim=local_dim)

    def encode_phi(self, phi, aneurysm_types):
        """Returns (ghd_embed, start_points_pred, local_embed)."""
        pyg_batch = self.multi_recon.to_pyg_batch(phi, aneurysm_types, self.point_std)
        return self.ghd_encoder(pyg_batch)

    def forward(self, local_points, token_mask, phi, cond):
        ghd_embed, start_points_pred, local_embed = self.encode_phi(phi, cond.aneurysm_type)
        cond = cond._replace(ghd_embed=ghd_embed, start_points=start_points_pred, local_embed=local_embed)
        mu, logvar = self.encoder(local_points, token_mask, cond)
        z          = self.reparameterize(mu, logvar)
        recon, length_logit = self.decoder(z, local_points, token_mask, cond)
        return recon, length_logit, mu, logvar, start_points_pred

    @torch.no_grad()
    def sample(self, phi, cond, z=None):
        ghd_embed, start_points_pred, local_embed = self.encode_phi(phi, cond.aneurysm_type)
        cond = cond._replace(ghd_embed=ghd_embed, start_points=start_points_pred, local_embed=local_embed)
        if z is None:
            z = torch.randn(phi.size(0), self.latent_dim, device=phi.device)
        outputs, lengths, token_mask = self.decoder.sample(z, cond)
        return outputs, lengths, token_mask, start_points_pred
