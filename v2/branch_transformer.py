"""
Per-branch curve VAE for the second-stage branch-point Transformer.

Operates on ONE branch at a time (the caller flattens the
[max_branches] dimension from VesselSkeletonDataset into the batch
dimension, keeping only branch_state == 1 "valid" branches).

Conditions:
  - phi [144, 3]: compressed by GHDTokenEncoder (a small MLP) into a
    low-dimensional embedding (`ghd_dim`), folded into `cond.ghd_embed`.
  - "small" conditions, bundled into a `BranchConditions` namedtuple
    (aneurysm_type, scale, start_points, branch_direction, branch_slot,
    ghd_embed [+ z for the Decoder]): combined into one embedding,
    broadcast and concatenated to every sequence token (channel-doubling,
    as in PartVessel/model/model_seq.py).

local_points: float32 [B, max_local_points, 3], relative to start_points,
              zero-padded beyond branch_length.
point_mask:   bool    [B, max_local_points]
branch_length: int64  [B], number of valid local_points (>=1).
branch_slot:  int64   [B], generation-order index (0..max_branches-1).
"""

import math
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BranchConditions(NamedTuple):
    """Per-branch conditions, broadcast-concatenated to every sequence
    token by Encoder and Decoder (see ConditionEmbedding). `ghd_embed`
    is filled in by BranchSequenceVAE from `phi` via GHDTokenEncoder."""

    aneurysm_type: torch.Tensor     # int64   [B]
    scale: torch.Tensor             # float32 [B]
    start_points: torch.Tensor      # float32 [B, 3]
    branch_direction: torch.Tensor  # float32 [B, 3]
    branch_slot: torch.Tensor       # int64   [B]
    ghd_embed: torch.Tensor = None  # float32 [B, ghd_dim]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


def causal_mask(sz, device=None):
    return torch.triu(torch.ones(sz, sz, dtype=torch.bool, device=device), diagonal=1)


class GHDTokenEncoder(nn.Module):
    """Compresses GHD phi [B, 144, 3] into a low-dimensional embedding via a small MLP."""

    def __init__(self, ghd_dim, num_coeffs=144, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_coeffs * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, ghd_dim),
        )

    def forward(self, phi):
        return self.net(phi.flatten(start_dim=1))


class ConditionEmbedding(nn.Module):
    """Combines aneurysm_type, scale, start_points, branch_direction,
    branch_slot, ghd_embed (and optionally z) into a single
    [B, hidden_dim] embedding."""

    def __init__(self, hidden_dim, num_types=3, max_branches=3, ghd_dim=16, latent_dim=None, part_dim=16):
        super().__init__()
        self.type_embedding = nn.Embedding(num_types, part_dim)
        self.slot_embedding = nn.Embedding(max_branches, part_dim)
        self.scale_proj = nn.Linear(1, part_dim)
        self.start_proj = nn.Linear(3, part_dim)
        self.dir_proj = nn.Linear(3, part_dim)
        in_dim = part_dim * 5 + ghd_dim + (latent_dim or 0)
        self.out_proj = nn.Linear(in_dim, hidden_dim)

    def forward(self, cond, z=None):
        parts = [
            self.type_embedding(cond.aneurysm_type.long()),
            self.slot_embedding(cond.branch_slot.long()),
            self.scale_proj(cond.scale.unsqueeze(-1)),
            self.start_proj(cond.start_points),
            self.dir_proj(cond.branch_direction),
            cond.ghd_embed,
        ]
        if z is not None:
            parts.append(z)
        return self.out_proj(torch.cat(parts, dim=-1))


class Encoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim, num_layers, nhead, max_local_points,
                 num_types=3, max_branches=3, ghd_dim=16):
        super().__init__()
        working_dim = hidden_dim * 2
        self.input_proj = nn.Linear(3, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, max_local_points)
        self.condition_embed = ConditionEmbedding(hidden_dim, num_types, max_branches, ghd_dim)

        layer = nn.TransformerEncoderLayer(working_dim, nhead, working_dim * 2,
                                            dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers)

        self.fc_mu = nn.Linear(working_dim + hidden_dim, latent_dim)
        self.fc_var = nn.Linear(working_dim + hidden_dim, latent_dim)

    def forward(self, local_points, point_mask, cond):
        tokens = self.pos_encoder(self.input_proj(local_points))
        cond_embed = self.condition_embed(cond)
        cond_expanded = cond_embed.unsqueeze(1).expand(-1, tokens.size(1), -1)
        tokens = torch.cat([tokens, cond_expanded], dim=-1)

        hidden = self.transformer(tokens, src_key_padding_mask=~point_mask)

        mask = point_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        pooled = torch.cat([pooled, cond_embed], dim=-1)
        return self.fc_mu(pooled), self.fc_var(pooled)


class Decoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim, num_layers, nhead, max_local_points,
                 num_types=3, max_branches=3, ghd_dim=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.max_local_points = max_local_points
        working_dim = hidden_dim * 2

        self.input_proj = nn.Linear(3, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, max_local_points)
        self.condition_embed = ConditionEmbedding(hidden_dim, num_types, max_branches, ghd_dim, latent_dim=latent_dim)
        self.length_predictor = nn.Linear(latent_dim + hidden_dim, max_local_points)

        layer = nn.TransformerEncoderLayer(working_dim, nhead, working_dim * 2,
                                            dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers)
        self.output_proj = nn.Linear(working_dim, 3)
        self.start_token = nn.Parameter(torch.randn(1, 1, 3) * 0.02)

    def _condition_and_length(self, z, cond):
        cond_embed = self.condition_embed(cond, z=z)
        length_logit = self.length_predictor(torch.cat([z, cond_embed], dim=-1))
        return cond_embed, length_logit

    def forward(self, z, local_points, point_mask, cond, tgt_mask):
        batch_size = local_points.size(0)
        cond_embed, length_logit = self._condition_and_length(z, cond)

        start_tokens = self.start_token.expand(batch_size, 1, -1)
        seq_in = torch.cat([start_tokens, local_points[:, :-1, :]], dim=1)

        tokens = self.pos_encoder(self.input_proj(seq_in))
        cond_expanded = cond_embed.unsqueeze(1).expand(-1, tokens.size(1), -1)
        tokens = torch.cat([tokens, cond_expanded], dim=-1)

        hidden = self.transformer(tokens, mask=tgt_mask, src_key_padding_mask=~point_mask)
        output = self.output_proj(hidden)
        output = output * point_mask.unsqueeze(-1).float()
        return output, length_logit

    @torch.no_grad()
    def sample(self, z, cond):
        device = z.device
        batch_size = z.size(0)
        cond_embed, length_logit = self._condition_and_length(z, cond)
        lengths = (length_logit.argmax(dim=-1) + 1).clamp(max=self.max_local_points)

        sequence = self.start_token.expand(batch_size, 1, -1)
        outputs = torch.zeros(batch_size, self.max_local_points, 3, device=device)
        for step in range(self.max_local_points):
            tokens = self.pos_encoder(self.input_proj(sequence))
            cond_expanded = cond_embed.unsqueeze(1).expand(-1, tokens.size(1), -1)
            tokens = torch.cat([tokens, cond_expanded], dim=-1)
            mask = causal_mask(tokens.size(1), device=device)
            hidden = self.transformer(tokens, mask=mask)
            next_point = self.output_proj(hidden[:, -1:])
            outputs[:, step:step + 1] = next_point
            sequence = torch.cat([sequence, next_point], dim=1)

        point_mask = torch.arange(self.max_local_points, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        outputs = outputs * point_mask.unsqueeze(-1).float()
        return outputs, lengths, point_mask


class BranchSequenceVAE(nn.Module):
    def __init__(self, hidden_dim=64, latent_dim=32, num_layers=4, nhead=4,
                 max_local_points=127, num_types=3, max_branches=3, num_coeffs=144, ghd_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_local_points = max_local_points

        self.ghd_encoder = GHDTokenEncoder(ghd_dim, num_coeffs)
        self.encoder = Encoder(hidden_dim, latent_dim, num_layers, nhead, max_local_points,
                                num_types, max_branches, ghd_dim)
        self.decoder = Decoder(hidden_dim, latent_dim, num_layers, nhead, max_local_points,
                                num_types, max_branches, ghd_dim)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, local_points, point_mask, phi, cond):
        cond = cond._replace(ghd_embed=self.ghd_encoder(phi))
        mu, logvar = self.encoder(local_points, point_mask, cond)
        z = self.reparameterize(mu, logvar)

        tgt_mask = causal_mask(self.max_local_points, device=local_points.device)
        recon, length_logit = self.decoder(z, local_points, point_mask, cond, tgt_mask)
        return recon, length_logit, mu, logvar

    def get_loss(self, recon, local_points, length_logit, branch_length, point_mask, mu, logvar):
        mask = point_mask.unsqueeze(-1).float()
        recon_loss = F.mse_loss(recon * mask, local_points * mask, reduction="sum") / mask.sum().clamp(min=1.0)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)

        length_target = (branch_length - 1).clamp(min=0, max=self.max_local_points - 1)
        length_loss = F.cross_entropy(length_logit, length_target)

        return recon_loss, kl_loss, length_loss

    @torch.no_grad()
    def sample(self, phi, cond, z=None):
        cond = cond._replace(ghd_embed=self.ghd_encoder(phi))
        if z is None:
            z = torch.randn(phi.size(0), self.latent_dim, device=phi.device)
        return self.decoder.sample(z, cond)
