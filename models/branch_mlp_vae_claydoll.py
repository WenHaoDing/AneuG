"""
Branch Fourier VAE — unclipped-centerline variant.

Parallel to models/branch_mlp_vae.py (which conditions on a real/mesh-opening
start point per branch and generates a short clipped stub). This variant
generates the full unclipped branch (dome-to-tip), which has no fixed
generation-time anchor the way a mesh opening does — a real unclipped branch's
start is deep near the aneurysm, not on any hand-annotated canonical landmark.

Resolution (see dataset/skeleton_dataset_fourier_claydoll.py's module docstring for
the data side): the GCN mesh encoder gets a second head that PREDICTS each
branch's own start point directly from the phi-deformed mesh —
GCNMeshEncoderWithStartPoints, [B, max_branches, 3] — supervised by the real
per-branch unclipped_centerline["branch_start_points"] at train time. This is
available identically at train and generation time (both mesh-derived), so
there's no train/generate conditioning mismatch to patch — unlike
models/branch_mlp_vae.py, this variant needs no SyntheticDirectionLoss and no
MultiCanonicalGHDReconstruct.compute_branch_conditions dependency at all.

Per-slot direction conditioning (branch_mlp_vae.py's dir_proj/branch_direction)
is dropped by default: there's no clean generation-time source for a real
per-branch direction either (same mismatch problem the start-point head
solves, but for direction), and z + the learned per-slot embedding already
give the decoder room to differentiate branches. Revisit only if generated
branches don't diverge directionally enough in practice.

target_dim stays 3 + 3k (chord + coeffs) — unchanged from branch_mlp_vae.py —
since forcing each branch's chord to start at its own (predicted/real) start
point needs no extra offset term.

conda activate new
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.branch_mlp_vae import BranchFourierVAE_GCNConditioner, _mlp
from models.branch_transformer import BranchConditionsClaydoll, GCNMeshEncoderWithStartPoints


def reconstruct_branch_torch(start, branch_vector, coeffs, n_points=128):
    """Differentiable, batched counterpart to
    dataset.skeleton_dataset_fourier.reconstruct_branch (duplicated here,
    not imported, to keep models/ from depending on dataset/) — reconstructs
    branch point sequences from a (predicted or real) start point, chord
    vector, and Fourier coefficients, for use in a point-space training loss.

    start, branch_vector: [B, mb, 3]
    coeffs:                [B, mb, k, 3]
    Returns:                [B, mb, n_points, 3]
    """
    k = coeffs.shape[-2]
    s = torch.linspace(0.0, math.pi, n_points, device=coeffs.device, dtype=coeffs.dtype)   # [n_points]
    frac = s / math.pi                                                                       # [n_points]
    line = start.unsqueeze(-2) + branch_vector.unsqueeze(-2) * frac.view(1, 1, -1, 1)        # [B, mb, n_points, 3]
    n = torch.arange(1, k + 1, device=coeffs.device, dtype=coeffs.dtype)                     # [k]
    A = torch.sin(torch.outer(s, n))                                                          # [n_points, k]
    bulge = torch.einsum('pk,bmkc->bmpc', A, coeffs)                                          # [B, mb, n_points, 3]
    return line + bulge


class BranchCondEmbeddingClaydoll(nn.Module):
    """Per-branch condition → [B, max_branches, cond_dim].

    Same as models.branch_mlp_vae.BranchCondEmbedding, minus the per-slot
    direction feature (dir_proj/branch_direction) — see module docstring.
    """

    def __init__(self, cond_dim, num_types=3, ghd_dim=16, max_branches=3, part_dim=16):
        super().__init__()
        self.slot_embed = nn.Embedding(max_branches, part_dim)
        self.start_proj = nn.Linear(3, part_dim)
        self.type_embed = nn.Embedding(num_types, part_dim)
        self.scale_proj = nn.Linear(1, part_dim)
        # slot + start + type + scale = 4 part_dim pieces, plus ghd_embed
        self.out_proj   = nn.Linear(part_dim * 4 + ghd_dim, cond_dim)

    def forward(self, cond):
        B, mb = cond.start_points.shape[:2]
        dev = cond.start_points.device
        slot  = self.slot_embed(torch.arange(mb, device=dev)).unsqueeze(0).expand(B, -1, -1)  # [B, mb, p]
        start = self.start_proj(cond.start_points)                                            # [B, mb, p]
        glob = torch.cat([
            self.type_embed(cond.aneurysm_type.long()),     # [B, p]
            self.scale_proj(cond.scale.unsqueeze(-1)),       # [B, p]
            cond.ghd_embed,                                  # [B, ghd_dim]
        ], dim=-1).unsqueeze(1).expand(-1, mb, -1)           # [B, mb, 2p + ghd_dim]
        return self.out_proj(torch.cat([slot, start, glob], dim=-1))                          # [B, mb, cond_dim]


class BranchFourierVAE_GCNConditionerClaydoll(BranchFourierVAE_GCNConditioner):
    """BranchFourierVAE_GCNConditioner with a per-branch start-point prediction
    head instead of real/mesh-opening start conditioning. See module docstring."""

    def __init__(self, multi_recon, k=8, hidden_dim=128, latent_dim=16, num_layers=2,
                 num_types=3, max_branches=3, num_coeffs=144, ghd_dim=16, cond_dim=64,
                 gcn_hidden=32, gcn_pool_ratio=0.5):
        super().__init__(multi_recon, k, hidden_dim, latent_dim, num_layers,
                         num_types, max_branches, num_coeffs, ghd_dim, cond_dim,
                         gcn_hidden, gcn_pool_ratio)
        # Replace the base class's encoder/conditioning modules with the
        # start-point-predicting versions.
        self.ghd_encoder = GCNMeshEncoderWithStartPoints(
            ghd_dim, max_branches, in_channels=3, hidden=gcn_hidden, pool_ratio=gcn_pool_ratio,
        )
        self.cond_embed = BranchCondEmbeddingClaydoll(cond_dim, num_types, ghd_dim, max_branches)

    def encode_phi(self, phi, aneurysm_types):
        """Returns (ghd_embed [B,ghd_dim], start_points_pred [B,mb,3]) — overrides
        the base class's single-tensor return, so forward/sample are also overridden
        below rather than reusing the base class's (which assume one return value)."""
        pyg_batch = self.multi_recon.to_pyg_batch(phi, aneurysm_types, point_std=None)
        return self.ghd_encoder(pyg_batch)

    def forward(self, branch_vector, fourier_coeffs, phi, cond):
        ghd_embed, start_points_pred = self.encode_phi(phi, cond.aneurysm_type)
        cond = cond._replace(ghd_embed=ghd_embed, start_points=start_points_pred)
        cond_emb = self.cond_embed(cond)
        target = self._norm_target(branch_vector, fourier_coeffs)
        mu, logvar = self._encode(target, cond_emb, cond.branch_mask)
        z = self.reparameterize(mu, logvar)
        out = self._decode(z, cond_emb)
        presence_logit = self._decode_presence(z, cond_emb)
        return out, presence_logit, mu, logvar, start_points_pred

    def get_loss(self, out, branch_vector, fourier_coeffs, branch_mask, opening_mask,
                 presence_logit, mu, logvar, start_points_pred, start_points_real,
                 point_n=128):
        coeff_loss, vec_loss, presence_loss, kl_loss = super().get_loss(
            out, branch_vector, fourier_coeffs, branch_mask, opening_mask,
            presence_logit, mu, logvar,
        )
        m = branch_mask.unsqueeze(-1).float()
        denom = m.sum().clamp(min=1.0)
        start_points_loss = ((start_points_pred - start_points_real) ** 2 * m).sum() / (denom * 3)

        # Point-space shape loss: reconstruct predicted vs. real branch curves
        # from the SAME (real) start point, so this measures chord+coeff shape
        # error only — start-point error is already supervised separately,
        # above, and isn't double-counted here. Complements coeff_loss/vec_loss
        # (which are MSE on normalized, per-harmonic-weighted parameters, not
        # on physical-space geometry — see reconstruct_branch_torch).
        vec_pred, coeffs_pred = self._denorm_out(out)
        pred_points = reconstruct_branch_torch(start_points_real, vec_pred, coeffs_pred, point_n)
        real_points = reconstruct_branch_torch(start_points_real, branch_vector, fourier_coeffs, point_n)
        pm = branch_mask.view(*branch_mask.shape, 1, 1).float()
        point_denom = pm.sum().clamp(min=1.0) * pred_points.shape[-2] * 3
        point_mse = ((pred_points - real_points) ** 2 * pm).sum() / point_denom

        return coeff_loss, vec_loss, presence_loss, kl_loss, start_points_loss, point_mse

    @torch.no_grad()
    def sample(self, phi, cond, z=None):
        """Returns (branch_vector [B,mb,3], coeffs [B,mb,k,3], presence [B,mb],
        start_points_pred [B,mb,3]) — reconstruct each branch via
        dataset.skeleton_dataset_fourier_claydoll.reconstruct_branch(
            start_points_pred[:,b], start_points_pred[:,b] + branch_vector[:,b], coeffs[:,b]).
        """
        ghd_embed, start_points_pred = self.encode_phi(phi, cond.aneurysm_type)
        cond = cond._replace(ghd_embed=ghd_embed, start_points=start_points_pred)
        cond_emb = self.cond_embed(cond)
        if z is None:
            z = torch.randn(phi.size(0), self.latent_dim, device=phi.device)
        out = self._decode(z, cond_emb)
        presence = torch.sigmoid(self._decode_presence(z, cond_emb))
        vec, coeffs = self._denorm_out(out)
        return vec, coeffs, presence, start_points_pred
