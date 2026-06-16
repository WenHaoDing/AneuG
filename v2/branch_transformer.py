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
        return self.fc_mu(pooled), self.fc_var(pooled)                        # [B, latent_dim] × 2


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

    def get_loss(self, recon, local_points, length_logit, branch_length, token_mask, branch_mask, mu, logvar):
        # recon:         [B, seq_len, 3]
        # local_points:  [B, seq_len, 3]
        # length_logit:  [B, max_branches, max_local_points]
        # branch_length: [B, max_branches]
        # token_mask:    [B, seq_len]
        # branch_mask:   [B, max_branches]   True = valid branch
        mask       = token_mask.unsqueeze(-1).float()                           # [B, seq_len, 1]
        recon_loss = F.mse_loss(recon * mask, local_points * mask, reduction="sum") / mask.sum().clamp(min=1.0)
        kl_loss    = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)

        length_target = (branch_length - 1).clamp(min=0, max=self.max_local_points - 1)  # [B, max_branches]
        length_loss   = F.cross_entropy(
            length_logit[branch_mask],    # [N_valid, max_local_points]
            length_target[branch_mask],   # [N_valid]
        )
        return recon_loss, kl_loss, length_loss

    @torch.no_grad()
    def sample(self, phi, cond, z=None):
        cond = cond._replace(ghd_embed=self.ghd_encoder(phi))
        if z is None:
            z = torch.randn(phi.size(0), self.latent_dim, device=phi.device)
        return self.decoder.sample(z, cond)
