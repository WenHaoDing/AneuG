"""
Multi-branch curve VAE — unclipped-centerline variant.

Parallel to models/branch_transformer.py's MultiBranchVAE_GCNConditioner
(autoregressive, causal-masked transformer over a flattened multi-branch
point sequence). That model conditions every token on a real/mesh-opening
start point per branch (MultiBranchConditions.start_points, read straight
from the clipped-centerline dataset at train time and from
MultiCanonicalGHDReconstruct.compute_branch_conditions at generation time) —
this variant generates the full unclipped branch (dome-to-tip) instead, which
has no such fixed generation-time anchor: a real unclipped branch's start is
deep near the aneurysm dome, not on any hand-annotated canonical landmark.

Resolution — identical in spirit to models/branch_mlp_vae_claydoll.py, reusing
its exact mechanism (models.branch_transformer.GCNMeshEncoderWithStartPoints,
models.branch_transformer.BranchConditionsClaydoll): the GCN mesh encoder gets
a second head that PREDICTS each branch's own start point directly from the
phi-deformed mesh, [B, max_branches, 3], supervised by the real per-branch
unclipped_centerline["branch_start_points"] at train time (via
dataset/skeleton_dataset_claydoll.py's start_points field). This is available
identically at train and generation time (both mesh-derived), so there's no
train/generate conditioning mismatch to patch — this variant needs no
synthetic-direction-style loss and no compute_branch_conditions dependency.

Per-token direction conditioning (branch_transformer.py's
PerTokenBranchEmbedding.dir_proj / Decoder._length_logits's dir_proj_len,
both driven by MultiBranchConditions.branch_direction) is dropped by default,
same call as models/branch_mlp_vae_claydoll.py: there's no clean
generation-time source for a real per-branch direction either (same mismatch
the start-point head solves, but for direction), and z + the learned per-slot
embedding + the predicted start already give the model room to differentiate
branches. Revisit only if generated branches don't diverge directionally
enough in practice.

Everything else — the causal autoregressive decoder, the per-branch length
predictor's overall shape, the point-space (not coefficient-space) MSE
reconstruction loss, presence/topology handling via branch_mask — is reused
unchanged from models/branch_transformer.py. Note the point-space recon_loss
here already IS a direct geometric loss (unlike the Fourier-coefficient
family, which needed a bespoke point_mse term added on top of its
normalized-coefficient losses).

conda activate new
"""

import torch
import torch.nn as nn

from models.branch_transformer import (
    BranchConditionsClaydoll,
    Decoder,
    Encoder,
    GCNMeshEncoderWithStartPoints,
    MultiBranchVAE_GCNConditioner,
)


class PerTokenBranchEmbeddingClaydoll(nn.Module):
    """Like branch_transformer.PerTokenBranchEmbedding, minus the per-slot
    direction feature (dir_proj/branch_direction) — see module docstring."""

    def __init__(self, hidden_dim, max_branches, max_local_points, part_dim=16):
        super().__init__()
        self.slot_embed = nn.Embedding(max_branches, part_dim)
        self.start_proj = nn.Linear(3, part_dim)
        self.out_proj   = nn.Linear(part_dim * 2, hidden_dim)

        # fixed: token position t → branch index  [seq_len]
        branch_ids = torch.arange(max_branches).repeat_interleave(max_local_points)
        self.register_buffer("branch_ids", branch_ids)   # [seq_len]

    def forward(self, cond):
        b_ids   = self.branch_ids                                            # [seq_len]
        B       = cond.start_points.size(0)
        slot_t  = self.slot_embed(b_ids).unsqueeze(0).expand(B, -1, -1)    # [B, seq_len, part_dim]
        start_t = self.start_proj(cond.start_points)[:, b_ids, :]          # [B, seq_len, part_dim]
        return self.out_proj(torch.cat([slot_t, start_t], dim=-1))         # [B, seq_len, hidden_dim]


class EncoderClaydoll(Encoder):
    """Encoder with PerTokenBranchEmbeddingClaydoll (predicted start, no
    direction) instead of PerTokenBranchEmbedding. forward() is inherited
    unchanged — it calls self.branch_embed(cond), so replacing the submodule
    here is sufficient."""

    def __init__(self, hidden_dim, latent_dim, num_layers, nhead, seq_len,
                 num_types=3, max_branches=3, max_local_points=127, ghd_dim=16):
        super().__init__(hidden_dim, latent_dim, num_layers, nhead, seq_len,
                         num_types, max_branches, max_local_points, ghd_dim)
        self.branch_embed = PerTokenBranchEmbeddingClaydoll(hidden_dim, max_branches, max_local_points)


class DecoderClaydoll(Decoder):
    """Decoder with PerTokenBranchEmbeddingClaydoll and a direction-free
    per-branch length predictor. forward()/sample() are inherited unchanged —
    both route through self.branch_embed and self._length_logits, so
    replacing those is sufficient."""

    def __init__(self, hidden_dim, latent_dim, num_layers, nhead, seq_len,
                 num_types=3, max_branches=3, max_local_points=127, ghd_dim=16, part_dim=16):
        super().__init__(hidden_dim, latent_dim, num_layers, nhead, seq_len,
                         num_types, max_branches, max_local_points, ghd_dim, part_dim)
        self.branch_embed = PerTokenBranchEmbeddingClaydoll(hidden_dim, max_branches, max_local_points)
        del self.dir_proj_len
        self.length_head = nn.Linear(latent_dim + part_dim * 2, max_local_points)

    def _length_logits(self, z, cond):
        # z: [B, latent_dim] → [B, max_branches, max_local_points]
        B         = z.size(0)
        slots     = torch.arange(self.max_branches, device=z.device)
        slot_emb  = self.slot_embed_len(slots).unsqueeze(0).expand(B, -1, -1)   # [B, max_branches, part_dim]
        start_emb = self.start_proj_len(cond.start_points)                       # [B, max_branches, part_dim]
        branch_feat = torch.cat([slot_emb, start_emb], dim=-1)                  # [B, max_branches, part_dim*2]
        z_exp     = z.unsqueeze(1).expand(-1, self.max_branches, -1)            # [B, max_branches, latent_dim]
        return self.length_head(torch.cat([z_exp, branch_feat], dim=-1))        # [B, max_branches, max_local_points]


class MultiBranchVAEClaydollGCNConditioner(MultiBranchVAE_GCNConditioner):
    """MultiBranchVAE_GCNConditioner with a per-branch start-point prediction
    head instead of real/mesh-opening start conditioning, and no per-token
    direction conditioning. See module docstring."""

    def __init__(self, multi_recon, hidden_dim=64, latent_dim=32, num_layers=4, nhead=4,
                 max_local_points=127, num_types=3, max_branches=3,
                 num_coeffs=144, ghd_dim=16, gcn_hidden=32, gcn_pool_ratio=0.5):
        super().__init__(multi_recon, hidden_dim, latent_dim, num_layers, nhead,
                         max_local_points, num_types, max_branches, num_coeffs, ghd_dim,
                         gcn_hidden, gcn_pool_ratio)
        # Replace the base class's mesh encoder + transformer encoder/decoder
        # with the start-point-predicting, direction-free versions.
        seq_len = max_branches * max_local_points
        self.ghd_encoder = GCNMeshEncoderWithStartPoints(
            ghd_dim, max_branches, in_channels=3, hidden=gcn_hidden, pool_ratio=gcn_pool_ratio,
        )
        self.encoder = EncoderClaydoll(hidden_dim, latent_dim, num_layers, nhead, seq_len,
                                       num_types, max_branches, max_local_points, ghd_dim)
        self.decoder = DecoderClaydoll(hidden_dim, latent_dim, num_layers, nhead, seq_len,
                                       num_types, max_branches, max_local_points, ghd_dim)

    def encode_phi(self, phi, aneurysm_types):
        """Returns (ghd_embed [B,ghd_dim], start_points_pred [B,mb,3]) — overrides
        the base class's single-tensor return, so forward/sample are also overridden
        below rather than reusing the base class's (which assume one return value)."""
        pyg_batch = self.multi_recon.to_pyg_batch(phi, aneurysm_types, self.point_std)
        return self.ghd_encoder(pyg_batch)

    def forward(self, local_points, token_mask, phi, cond):
        ghd_embed, start_points_pred = self.encode_phi(phi, cond.aneurysm_type)
        cond = cond._replace(ghd_embed=ghd_embed, start_points=start_points_pred)
        mu, logvar = self.encoder(local_points, token_mask, cond)
        z          = self.reparameterize(mu, logvar)
        recon, length_logit = self.decoder(z, local_points, token_mask, cond)
        return recon, length_logit, mu, logvar, start_points_pred

    def get_loss(self, recon, local_points, length_logit, branch_length, token_mask,
                 branch_mask, mu, logvar, start_points_pred, start_points_real,
                 point_weights=None):
        recon_loss, kl_loss, length_loss = super().get_loss(
            recon, local_points, length_logit, branch_length, token_mask,
            branch_mask, mu, logvar, point_weights,
        )
        m = branch_mask.unsqueeze(-1).float()
        denom = m.sum().clamp(min=1.0)
        start_points_loss = ((start_points_pred - start_points_real) ** 2 * m).sum() / (denom * 3)
        return recon_loss, kl_loss, length_loss, start_points_loss

    @torch.no_grad()
    def sample(self, phi, cond, z=None):
        """Returns (outputs [B,seq_len,3], lengths [B,mb], token_mask [B,seq_len],
        start_points_pred [B,mb,3]) — reconstruct each branch's absolute points via
        start_points_pred[:,b] + denormalize_local_points(outputs[b, :lengths[b]])."""
        ghd_embed, start_points_pred = self.encode_phi(phi, cond.aneurysm_type)
        cond = cond._replace(ghd_embed=ghd_embed, start_points=start_points_pred)
        if z is None:
            z = torch.randn(phi.size(0), self.latent_dim, device=phi.device)
        outputs, lengths, token_mask = self.decoder.sample(z, cond)
        return outputs, lengths, token_mask, start_points_pred
