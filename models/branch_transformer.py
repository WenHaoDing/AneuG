"""
Multi-branch curve VAE.

All branches of one case are concatenated into a single token sequence:
  seq_len = max_branches * max_local_points   (e.g. 3 × 127 = 381)

A causal mask lets each token attend to all earlier tokens, so when the
decoder generates branch k it sees the full geometry of branches 0..k-1.
One latent z [B, latent_dim] encodes the complete multi-branch topology.

Dataset fields consumed (shapes after DataLoader batch dimension B):
  local_points:     [B, max_branches, max_local_points, 3] → flatten → [B, seq_len, 3]
  token_mask:       [B, seq_len]          already flat (= point_mask.flatten())
  branch_length:    [B, max_branches]
  branch_mask:      [B, max_branches]     True = valid branch (state == 1)
  start_points:     [B, max_branches, 3]
  branch_direction: [B, max_branches, 3]
  phi:              [B, 144, 3]
  aneurysm_type:    [B]
  scale:            [B]
"""

import math
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiBranchConditions(NamedTuple):
    aneurysm_type:    torch.Tensor         # int64   [B]
    scale:            torch.Tensor         # float32 [B]
    start_points:     torch.Tensor         # float32 [B, max_branches, 3]
    branch_direction: torch.Tensor         # float32 [B, max_branches, 3]
    branch_mask:      torch.Tensor         # bool    [B, max_branches]
    ghd_embed:        torch.Tensor = None  # float32 [B, ghd_dim]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))    # [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]           # [B, T, d_model]


def causal_mask(sz, device=None):
    return torch.triu(torch.ones(sz, sz, dtype=torch.bool, device=device), diagonal=1)


class GHDTokenEncoder(nn.Module):
    def __init__(self, ghd_dim, num_coeffs=144, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_coeffs * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, ghd_dim),
        )

    def forward(self, phi):
        return self.net(phi.flatten(start_dim=1))      # [B, ghd_dim]


class PerTokenBranchEmbedding(nn.Module):
    """Injects per-branch geometry into each sequence token.
    Token at position t belongs to branch b = t // max_local_points."""

    def __init__(self, hidden_dim, max_branches, max_local_points, part_dim=16):
        super().__init__()
        self.slot_embed = nn.Embedding(max_branches, part_dim)
        self.start_proj = nn.Linear(3, part_dim)
        self.dir_proj   = nn.Linear(3, part_dim)
        self.out_proj   = nn.Linear(part_dim * 3, hidden_dim)

        # fixed: token position t → branch index  [seq_len]
        branch_ids = torch.arange(max_branches).repeat_interleave(max_local_points)
        self.register_buffer("branch_ids", branch_ids)   # [seq_len]

    def forward(self, cond):
        b_ids   = self.branch_ids                                            # [seq_len]
        B       = cond.start_points.size(0)
        slot_t  = self.slot_embed(b_ids).unsqueeze(0).expand(B, -1, -1)    # [B, seq_len, part_dim]
        start_t = self.start_proj(cond.start_points)[:, b_ids, :]          # [B, seq_len, part_dim]
        dir_t   = self.dir_proj(cond.branch_direction)[:, b_ids, :]        # [B, seq_len, part_dim]
        return self.out_proj(torch.cat([slot_t, start_t, dir_t], dim=-1))  # [B, seq_len, hidden_dim]


class GlobalCondEmbedding(nn.Module):
    """Embeds aneurysm_type, scale, ghd_embed (and optionally z) → [B, hidden_dim]."""

    def __init__(self, hidden_dim, num_types=3, ghd_dim=16, latent_dim=None, part_dim=16):
        super().__init__()
        self.type_embed = nn.Embedding(num_types, part_dim)
        self.scale_proj = nn.Linear(1, part_dim)
        in_dim = part_dim * 2 + ghd_dim + (latent_dim or 0)
        self.out_proj = nn.Linear(in_dim, hidden_dim)

    def forward(self, cond, z=None):
        parts = [
            self.type_embed(cond.aneurysm_type.long()),   # [B, part_dim]
            self.scale_proj(cond.scale.unsqueeze(-1)),    # [B, part_dim]
            cond.ghd_embed,                               # [B, ghd_dim]
        ]
        if z is not None:
            parts.append(z)                               # [B, latent_dim]
        return self.out_proj(torch.cat(parts, dim=-1))    # [B, hidden_dim]


class Encoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim, num_layers, nhead, seq_len,
                 num_types=3, max_branches=3, max_local_points=127, ghd_dim=16):
        super().__init__()
        working_dim = hidden_dim * 2
        self.input_proj   = nn.Linear(3, hidden_dim)
        self.pos_enc      = PositionalEncoding(hidden_dim, seq_len)
        self.branch_embed = PerTokenBranchEmbedding(hidden_dim, max_branches, max_local_points)
        self.global_cond  = GlobalCondEmbedding(hidden_dim, num_types, ghd_dim)

        layer = nn.TransformerEncoderLayer(working_dim, nhead, working_dim * 2, dropout=0.1, batch_first=True)
        self.transformer  = nn.TransformerEncoder(layer, num_layers)

        self.fc_mu  = nn.Linear(working_dim + hidden_dim, latent_dim)
        self.fc_var = nn.Linear(working_dim + hidden_dim, latent_dim)

    def forward(self, local_points, token_mask, cond):
        # local_points: [B, seq_len, 3]
        # token_mask:   [B, seq_len]   True = valid token
        tokens  = self.pos_enc(self.input_proj(local_points))                  # [B, seq_len, hidden_dim]
        tokens  = tokens + self.branch_embed(cond)                             # [B, seq_len, hidden_dim]
        g_emb   = self.global_cond(cond)                                       # [B, hidden_dim]
        tokens  = torch.cat([tokens, g_emb.unsqueeze(1).expand(-1, tokens.size(1), -1)], dim=-1)  # [B, seq_len, hidden_dim*2]

        hidden  = self.transformer(tokens, src_key_padding_mask=~token_mask)   # [B, seq_len, hidden_dim*2]

        mask    = token_mask.unsqueeze(-1).float()                             # [B, seq_len, 1]
        # pooling with masking, essential as different cases have very different numbers of valid tokens.
        pooled  = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)        # [B, hidden_dim*2]
        pooled  = torch.cat([pooled, g_emb], dim=-1)                          # [B, hidden_dim*2 + hidden_dim]
        # Clamp logvar so exp(0.5*logvar) (reparameterize) and logvar.exp() (KL)
        # cannot overflow to inf -> NaN.
        logvar  = self.fc_var(pooled).clamp(-10.0, 10.0)
        return self.fc_mu(pooled), logvar                                     # [B, latent_dim] × 2


class Decoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim, num_layers, nhead, seq_len,
                 num_types=3, max_branches=3, max_local_points=127, ghd_dim=16, part_dim=16):
        super().__init__()
        self.seq_len          = seq_len
        self.max_branches     = max_branches
        self.max_local_points = max_local_points
        working_dim = hidden_dim * 2

        self.input_proj   = nn.Linear(3, hidden_dim)
        self.pos_enc      = PositionalEncoding(hidden_dim, seq_len)
        self.branch_embed = PerTokenBranchEmbedding(hidden_dim, max_branches, max_local_points)
        self.global_cond  = GlobalCondEmbedding(hidden_dim, num_types, ghd_dim, latent_dim=latent_dim)

        layer = nn.TransformerEncoderLayer(working_dim, nhead, working_dim * 2, dropout=0.1, batch_first=True)
        self.transformer  = nn.TransformerEncoder(layer, num_layers)
        self.output_proj  = nn.Linear(working_dim, 3)
        self.start_token  = nn.Parameter(torch.randn(1, 1, 3) * 0.02)

        # per-branch length predictor: z + branch geometry → length logits
        self.slot_embed_len = nn.Embedding(max_branches, part_dim)
        self.start_proj_len = nn.Linear(3, part_dim)
        self.dir_proj_len   = nn.Linear(3, part_dim)
        self.length_head    = nn.Linear(latent_dim + part_dim * 3, max_local_points)

    def _length_logits(self, z, cond):
        # z: [B, latent_dim] → [B, max_branches, max_local_points]
        B         = z.size(0)
        slots     = torch.arange(self.max_branches, device=z.device)
        slot_emb  = self.slot_embed_len(slots).unsqueeze(0).expand(B, -1, -1)   # [B, max_branches, part_dim]
        start_emb = self.start_proj_len(cond.start_points)                       # [B, max_branches, part_dim]
        dir_emb   = self.dir_proj_len(cond.branch_direction)                     # [B, max_branches, part_dim]
        branch_feat = torch.cat([slot_emb, start_emb, dir_emb], dim=-1)         # [B, max_branches, part_dim*3]
        z_exp     = z.unsqueeze(1).expand(-1, self.max_branches, -1)            # [B, max_branches, latent_dim]
        return self.length_head(torch.cat([z_exp, branch_feat], dim=-1))        # [B, max_branches, max_local_points]

    def forward(self, z, local_points, token_mask, cond):
        # local_points: [B, seq_len, 3]   ground truth (teacher forcing)
        # token_mask:   [B, seq_len]
        B      = local_points.size(0)
        g_emb  = self.global_cond(cond, z=z)                                    # [B, hidden_dim]

        seq_in = torch.cat([self.start_token.expand(B, 1, -1), local_points[:, :-1, :]], dim=1)  # [B, seq_len, 3]
        tokens = self.pos_enc(self.input_proj(seq_in))                           # [B, seq_len, hidden_dim]
        tokens = tokens + self.branch_embed(cond)                                # [B, seq_len, hidden_dim]
        tokens = torch.cat([tokens, g_emb.unsqueeze(1).expand(-1, self.seq_len, -1)], dim=-1)    # [B, seq_len, hidden_dim*2]

        # shifted mask: position 0 is the start_token (always valid);
        # positions 1..seq_len-1 correspond to local_points[:, 0..seq_len-2]
        dec_mask = torch.cat([token_mask.new_ones(B, 1), token_mask[:, :-1]], dim=1)  # [B, seq_len]
        tgt_mask = causal_mask(self.seq_len, device=local_points.device)               # [seq_len, seq_len]
        hidden   = self.transformer(tokens, mask=tgt_mask, src_key_padding_mask=~dec_mask)       # [B, seq_len, hidden_dim*2]
        recon    = self.output_proj(hidden) * token_mask.unsqueeze(-1).float()  # [B, seq_len, 3]

        return recon, self._length_logits(z, cond)                               # [B, seq_len, 3], [B, max_branches, max_local_points]

    @torch.no_grad()
    def sample(self, z, cond):
        device   = z.device
        B        = z.size(0)
        g_emb    = self.global_cond(cond, z=z)                                  # [B, hidden_dim]
        lengths  = (self._length_logits(z, cond).argmax(-1) + 1).clamp(max=self.max_local_points)  # [B, max_branches]

        branch_embed_full = self.branch_embed(cond)                              # [B, seq_len, hidden_dim]
        sequence  = self.start_token.expand(B, 1, -1)                           # [B, 1, 3]
        outputs   = torch.zeros(B, self.seq_len, 3, device=device)              # [B, seq_len, 3]
        # start_token is always a valid key; grow the mask alongside sequence
        key_valid = torch.ones(B, 1, dtype=torch.bool, device=device)           # [B, 1]

        for step in range(self.seq_len):
            b           = step // self.max_local_points
            branch_valid = cond.branch_mask[:, b]                               # [B]

            tokens = self.pos_enc(self.input_proj(sequence))                    # [B, step+1, hidden_dim]
            tokens = tokens + branch_embed_full[:, :step + 1]                  # [B, step+1, hidden_dim]
            tokens = torch.cat([tokens, g_emb.unsqueeze(1).expand(-1, step + 1, -1)], dim=-1)
            hidden = self.transformer(tokens, mask=causal_mask(step + 1, device),
                                      src_key_padding_mask=~key_valid)          # [B, step+1, hidden_dim*2]

            # zero out generated point for absent branch slots (matches training zeros)
            next_pt = self.output_proj(hidden[:, -1:])                          # [B, 1, 3]
            next_pt = next_pt * branch_valid.float()[:, None, None]            # [B, 1, 3]
            outputs[:, step:step + 1] = next_pt
            sequence  = torch.cat([sequence, next_pt], dim=1)                  # [B, step+2, 3]
            key_valid = torch.cat([key_valid, branch_valid.unsqueeze(1)], dim=1)  # [B, step+2]

        # reconstruct token_mask from predicted per-branch lengths
        b_ids      = torch.arange(self.max_branches, device=device).repeat_interleave(self.max_local_points)  # [seq_len]
        local_ids  = torch.arange(self.max_local_points, device=device).repeat(self.max_branches)             # [seq_len]
        token_mask = local_ids.unsqueeze(0) < lengths[:, b_ids]                                               # [B, seq_len]
        return outputs * token_mask.unsqueeze(-1).float(), lengths, token_mask


class MultiBranchVAE(nn.Module):
    def __init__(self, hidden_dim=64, latent_dim=32, num_layers=4, nhead=4,
                 max_local_points=127, num_types=3, max_branches=3,
                 num_coeffs=144, ghd_dim=16):
        super().__init__()
        self.latent_dim       = latent_dim
        self.max_branches     = max_branches
        self.max_local_points = max_local_points
        seq_len = max_branches * max_local_points

        self.ghd_encoder = GHDTokenEncoder(ghd_dim, num_coeffs)
        self.encoder = Encoder(hidden_dim, latent_dim, num_layers, nhead, seq_len,
                               num_types, max_branches, max_local_points, ghd_dim)
        self.decoder = Decoder(hidden_dim, latent_dim, num_layers, nhead, seq_len,
                               num_types, max_branches, max_local_points, ghd_dim)

        # dataset normalization stats — set via set_normalization_stats() before training
        self.register_buffer("point_mean", torch.zeros(1, 3))  # [1, 3]
        self.register_buffer("point_std",  torch.ones(1, 3))   # [1, 3]

    def set_normalization_stats(self, point_mean, point_std):
        self.point_mean.copy_(point_mean)
        self.point_std.copy_(point_std)

    @staticmethod
    def reparameterize(mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def forward(self, local_points, token_mask, phi, cond):
        # local_points: [B, seq_len, 3]
        # token_mask:   [B, seq_len]
        # phi:          [B, 144, 3]
        cond       = cond._replace(ghd_embed=self.ghd_encoder(phi))             # ghd_embed: [B, ghd_dim]
        mu, logvar = self.encoder(local_points, token_mask, cond)               # [B, latent_dim] × 2
        z          = self.reparameterize(mu, logvar)                            # [B, latent_dim]
        recon, length_logit = self.decoder(z, local_points, token_mask, cond)
        return recon, length_logit, mu, logvar

    def get_loss(self, recon, local_points, length_logit, branch_length, token_mask,
                 branch_mask, mu, logvar, point_weights=None):
        # recon:             [B, seq_len, 3]
        # local_points:      [B, seq_len, 3]
        # length_logit:      [B, max_branches, max_local_points]
        # branch_length:     [B, max_branches]
        # token_mask:        [B, seq_len]
        # branch_mask:       [B, max_branches]   True = valid branch
        # point_weights:     [B, seq_len] or None — per-point recon weights
        #                    (sum to the point count per branch, so the loss
        #                    scale matches the unweighted case).
        mask       = token_mask.unsqueeze(-1).float()                           # [B, seq_len, 1]
        if point_weights is None:
            recon_loss = F.mse_loss(recon * mask, local_points * mask, reduction="sum") / mask.sum().clamp(min=1.0)
        else:
            w = point_weights.unsqueeze(-1) * mask                              # [B, seq_len, 1]
            recon_loss = (((recon - local_points) ** 2) * w).sum() / mask.sum().clamp(min=1.0)
        kl_loss    = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)

        length_target = (branch_length - 1).clamp(min=0, max=self.max_local_points - 1)  # [B, max_branches]
        length_loss   = F.cross_entropy(
            length_logit[branch_mask],    # [N_valid, max_local_points]
            length_target[branch_mask],   # [N_valid]
        )

        return recon_loss, kl_loss, length_loss

    def get_dir_loss(self, recon, branch_length, branch_mask, branch_direction, n_points=10):
        # recon:            [B, seq_len, 3]
        # branch_length:    [B, max_branches]
        # branch_mask:      [B, max_branches]
        # branch_direction: [B, max_branches, 3]  unit target vectors
        K        = min(n_points, self.max_local_points)
        Lp       = self.max_local_points
        recon_br = recon.view(recon.size(0), self.max_branches, Lp, 3)
        local_K  = recon_br[:, :, :K, :] * self.point_std + self.point_mean  # [B, max_branches, K, 3]

        k_ids    = torch.arange(K, device=recon.device)
        k_valid  = k_ids.unsqueeze(0).unsqueeze(0) < branch_length.unsqueeze(-1)  # [B, max_branches, K]
        avg_off  = (local_K * k_valid.unsqueeze(-1).float()).sum(2) / k_valid.float().sum(2).clamp(min=1.0).unsqueeze(-1)

        pred_dir = F.normalize(avg_off, dim=-1)
        cos_sim  = (pred_dir * branch_direction).sum(dim=-1)                  # [B, max_branches]
        return (1.0 - cos_sim)[branch_mask].mean()

    def encode_phi(self, phi, _aneurysm_types=None):
        """Encode phi → ghd_embed [B, ghd_dim].  Override in GCN variant."""
        return self.ghd_encoder(phi)

    @torch.no_grad()
    def sample(self, phi, cond, z=None):
        cond = cond._replace(ghd_embed=self.ghd_encoder(phi))
        if z is None:
            z = torch.randn(phi.size(0), self.latent_dim, device=phi.device)
        return self.decoder.sample(z, cond)



class GCNMeshEncoder(nn.Module):
    """GCN UNet encoder: PyG Batch of reconstructed meshes → [B, ghd_dim].

    Mesh reconstruction is handled externally (MultiCanonicalGHDReconstruct).
    This module only contains the learnable GCN layers.

    Three levels of GCNConv + SAGPooling; global readout at each level is
    concatenated and projected to ghd_dim.

    Input (forward):
        data: torch_geometric.data.Batch with
            x:          [total_N, 3]   vertex positions
            edge_index: [2, total_E]
            batch:      [total_N]
    """

    def __init__(self, ghd_dim, in_channels=3, hidden=32, pool_ratio=0.5):
        super().__init__()
        from torch_geometric.nn import GCNConv, SAGPooling
        self.conv1 = GCNConv(in_channels, hidden)
        self.pool1 = SAGPooling(hidden, ratio=pool_ratio)
        self.conv2 = GCNConv(hidden, hidden * 2)
        self.pool2 = SAGPooling(hidden * 2, ratio=pool_ratio)
        self.conv3 = GCNConv(hidden * 2, hidden * 2)
        self.mlp = nn.Sequential(
            nn.Linear(hidden + hidden * 2 + hidden * 2, hidden * 4),
            nn.ReLU(),
            nn.Linear(hidden * 4, ghd_dim),
        )

    def _pool(self, data):
        """GCN trunk + multi-level global pooling, before the final ghd_dim
        projection. Exposed separately so subclasses can add extra heads off
        the same pooled features without duplicating the GCN/pooling logic."""
        from torch_geometric.nn import global_mean_pool
        x, ei, batch = data.x, data.edge_index, data.batch

        x1 = F.relu(self.conv1(x, ei))                                     # [total_N, hidden]
        x1, ei1, _, batch1, _, _ = self.pool1(x1, ei, batch=batch)
        g1 = global_mean_pool(x1, batch1)                                   # [B, hidden]

        x2 = F.relu(self.conv2(x1, ei1))                                   # [*, hidden*2]
        x2, ei2, _, batch2, _, _ = self.pool2(x2, ei1, batch=batch1)
        g2 = global_mean_pool(x2, batch2)                                   # [B, hidden*2]

        x3 = F.relu(self.conv3(x2, ei2))                                   # [*, hidden*2]
        g3 = global_mean_pool(x3, batch2)                                   # [B, hidden*2]

        return torch.cat([g1, g2, g3], dim=-1)                             # [B, hidden*5]

    def forward(self, data):
        return self.mlp(self._pool(data))                                  # [B, ghd_dim]


class BranchConditionsClaydoll(NamedTuple):
    """Conditions for the claydoll (unclipped-centerline, predicted per-branch
    start point) model variants — shared across the Fourier-MLP-VAE and
    point-sequence-transformer families (models/branch_mlp_vae_claydoll.py,
    models/branch_transformer_claydoll.py). No branch_direction field: unlike
    MultiBranchConditions, claydoll variants drop per-slot direction
    conditioning by default (see models/branch_mlp_vae_claydoll.py's module
    docstring)."""
    aneurysm_type: torch.Tensor          # int64   [B]
    scale:         torch.Tensor          # float32 [B]
    branch_mask:   torch.Tensor          # bool    [B, max_branches]
    start_points:  torch.Tensor = None   # float32 [B, max_branches, 3] — filled in by forward/sample (predicted)
    ghd_embed:     torch.Tensor = None   # float32 [B, ghd_dim]         — filled in by forward/sample


class GCNMeshEncoderWithStartPoints(GCNMeshEncoder):
    """GCNMeshEncoder plus a second head predicting each branch's own start
    point off the same pooled mesh features — no duplicated GCN/pooling logic.
    Shared across claydoll model families; see BranchConditionsClaydoll."""

    def __init__(self, ghd_dim, max_branches, in_channels=3, hidden=32, pool_ratio=0.5):
        super().__init__(ghd_dim, in_channels=in_channels, hidden=hidden, pool_ratio=pool_ratio)
        self.max_branches = max_branches
        pooled_dim = hidden + hidden * 2 + hidden * 2
        self.start_points_head = nn.Linear(pooled_dim, max_branches * 3)

    def forward(self, data):
        pooled = self._pool(data)
        ghd_embed = self.mlp(pooled)                                            # [B, ghd_dim]
        start_points_pred = self.start_points_head(pooled).view(-1, self.max_branches, 3)
        return ghd_embed, start_points_pred


class BranchConditionsClaydollAttn(NamedTuple):
    """BranchConditionsClaydoll plus a fifth per-slot piece: the local-attention
    readout from GCNMeshEncoderWithLocalAttention. Shared across claydoll model
    families; see that class's docstring."""
    aneurysm_type: torch.Tensor          # int64   [B]
    scale:         torch.Tensor          # float32 [B]
    branch_mask:   torch.Tensor          # bool    [B, max_branches]
    start_points:  torch.Tensor = None   # float32 [B, max_branches, 3] — filled in by forward/sample (predicted)
    ghd_embed:     torch.Tensor = None   # float32 [B, ghd_dim]         — filled in by forward/sample
    local_embed:   torch.Tensor = None   # float32 [B, max_branches, local_dim] — filled in by forward/sample


class GCNMeshEncoderWithLocalAttention(GCNMeshEncoderWithStartPoints):
    """GCNMeshEncoderWithStartPoints plus a per-branch local-attention readout
    over pool1-level tokens (features + original mesh position). Shared across
    claydoll model families (models/branch_mlp_vae_claydoll_attn.py,
    models/branch_transformer_claydoll_attn.py) — see
    models/branch_mlp_vae_claydoll_attn.py's module docstring for the full
    design rationale (split between this GCN-specific plumbing and
    SlotQueryCrossAttention's generic attention readout)."""

    def __init__(self, ghd_dim, max_branches, in_channels=3, hidden=32, pool_ratio=0.5,
                local_dim=32, attn_heads=4):
        super().__init__(ghd_dim, max_branches, in_channels=in_channels,
                         hidden=hidden, pool_ratio=pool_ratio)
        token_dim = hidden + 3   # pool1 feature dim + concatenated original xyz position
        self.query_slot_embed = nn.Embedding(max_branches, 16)
        self.query_in_proj = nn.Linear(16 + 3, local_dim)   # slot embed + predicted start xyz -> query
        self.local_attn = SlotQueryCrossAttention(
            query_dim=local_dim, token_dim=token_dim, out_dim=local_dim, num_heads=attn_heads,
        )

    def forward(self, data):
        from torch_geometric.nn import global_mean_pool
        from torch_geometric.utils import to_dense_batch
        x, ei, batch = data.x, data.edge_index, data.batch

        x1 = F.relu(self.conv1(x, ei))                                     # [total_N, hidden]
        x1, ei1, _, batch1, perm1, _ = self.pool1(x1, ei, batch=batch)
        g1 = global_mean_pool(x1, batch1)                                   # [B, hidden]

        x2 = F.relu(self.conv2(x1, ei1))                                   # [*, hidden*2]
        x2, ei2, _, batch2, _, _ = self.pool2(x2, ei1, batch=batch1)
        g2 = global_mean_pool(x2, batch2)                                   # [B, hidden*2]

        x3 = F.relu(self.conv3(x2, ei2))                                   # [*, hidden*2]
        g3 = global_mean_pool(x3, batch2)                                   # [B, hidden*2]

        pooled = torch.cat([g1, g2, g3], dim=-1)                           # [B, hidden*5]
        ghd_embed = self.mlp(pooled)                                        # [B, ghd_dim]
        start_points_pred = self.start_points_head(pooled).view(-1, self.max_branches, 3)

        # Intermediate-level tokens: pool1 features + each surviving node's
        # ORIGINAL mesh position — perm1 indexes into the pre-pool1 node set,
        # which is the same node set/order as `x` (conv1 doesn't change node
        # count), so x[perm1] recovers the true (unlearned) xyz per token.
        node_xyz = x[perm1]                                                 # [N1, 3]
        tokens = torch.cat([x1, node_xyz], dim=-1)                         # [N1, hidden+3]
        tokens_dense, token_mask = to_dense_batch(tokens, batch1)          # [B, Nmax, hidden+3], [B, Nmax]

        B = pooled.size(0)
        slot_ids = torch.arange(self.max_branches, device=x.device)
        slot_q = self.query_slot_embed(slot_ids).unsqueeze(0).expand(B, -1, -1)   # [B, mb, 16]
        query = self.query_in_proj(torch.cat([slot_q, start_points_pred], dim=-1))  # [B, mb, local_dim]

        local_embed = self.local_attn(query, tokens_dense, token_mask)     # [B, mb, local_dim]

        return ghd_embed, start_points_pred, local_embed


class SlotQueryCrossAttention(nn.Module):
    """Per-slot (branch) query attends over a padded/masked token set.

    Generic cross-attention readout, deliberately unaware of where its inputs
    come from: given one query vector per branch slot and a batch of
    variable-length token sets (e.g. intermediate per-node encoder features,
    optionally concatenated with position), returns one attended vector per
    slot. Query construction and token gathering are the caller's job — e.g.
    models/branch_mlp_vae_v2_attn.py's GCNMeshEncoderWithLocalAttention builds
    queries from a per-slot embedding + a predicted spatial anchor, and gathers
    tokens from an intermediate GCN pooling level via that pool's permutation
    indices. Kept here (rather than inline in that GCN-specific file) so a
    future transformer-token-based encoder in this module can reuse the same
    attention readout without duplicating it.

    Input:
        query:      [B, max_branches, query_dim]
        tokens:     [B, N, token_dim]   padded/batched token set
        token_mask: [B, N] bool, True where tokens[:, n] is a real (non-pad) token
    Output:
        [B, max_branches, out_dim]
    """

    def __init__(self, query_dim, token_dim, out_dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=query_dim, num_heads=num_heads,
            kdim=token_dim, vdim=token_dim, batch_first=True,
        )
        self.out_proj = nn.Linear(query_dim, out_dim) if out_dim != query_dim else nn.Identity()

    def forward(self, query, tokens, token_mask, need_weights=False):
        """need_weights=True additionally returns the per-slot attention
        distribution over tokens, [B, max_branches, N] (averaged across
        heads, softmax already applied, ~0 on padded tokens) — e.g. for a
        soft-argmax readout over token positions, or an auxiliary
        classification-style loss on the distribution itself."""
        attended, weights = self.attn(query, tokens, tokens,
                                      key_padding_mask=~token_mask, need_weights=need_weights,
                                      average_attn_weights=True)
        out = self.out_proj(attended)
        return (out, weights) if need_weights else out


class MultiBranchVAE_GCNConditioner(MultiBranchVAE):
    """MultiBranchVAE with a GCN UNet mesh encoder instead of GHDTokenEncoder.

    multi_recon (MultiCanonicalGHDReconstruct) reconstructs the mesh from phi
    and returns a PyG Batch; GCNMeshEncoder encodes that into ghd_embed.

    Usage:
        multi_recon = MultiCanonicalGHDReconstruct(canonical_root, device=DEVICE)
        model = MultiBranchVAE_GCNConditioner(multi_recon, ...)
        model.set_normalization_stats(point_mean, point_std)
    """

    def __init__(self, multi_recon, hidden_dim=64, latent_dim=32, num_layers=4, nhead=4,
                 max_local_points=127, num_types=3, max_branches=3,
                 num_coeffs=144, ghd_dim=16, gcn_hidden=32, gcn_pool_ratio=0.5):
        super().__init__(hidden_dim, latent_dim, num_layers, nhead,
                         max_local_points, num_types, max_branches, num_coeffs, ghd_dim)
        self.multi_recon = multi_recon
        self.ghd_encoder = GCNMeshEncoder(ghd_dim, in_channels=3,
                                          hidden=gcn_hidden, pool_ratio=gcn_pool_ratio)

    def forward(self, local_points, token_mask, phi, cond):
        pyg_batch = self.multi_recon.to_pyg_batch(phi, cond.aneurysm_type, self.point_std)
        cond      = cond._replace(ghd_embed=self.ghd_encoder(pyg_batch))
        mu, logvar = self.encoder(local_points, token_mask, cond)
        z          = self.reparameterize(mu, logvar)
        recon, length_logit = self.decoder(z, local_points, token_mask, cond)
        return recon, length_logit, mu, logvar

    def encode_phi(self, phi, aneurysm_types):
        pyg_batch = self.multi_recon.to_pyg_batch(phi, aneurysm_types, self.point_std)
        return self.ghd_encoder(pyg_batch)

    @torch.no_grad()
    def sample(self, phi, cond, z=None):
        pyg_batch = self.multi_recon.to_pyg_batch(phi, cond.aneurysm_type, self.point_std)
        cond      = cond._replace(ghd_embed=self.ghd_encoder(pyg_batch))
        if z is None:
            z = torch.randn(phi.size(0), self.latent_dim, device=phi.device)
        return self.decoder.sample(z, cond)