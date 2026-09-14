"""Probabilistic MorphoFormer: cap and dome regions as a DISTRIBUTION of plausible
labellings rather than one answer.

THE PROBLEM. A human brushing a cap does not produce a repeatable boundary. Two
passes over the same mesh give different vertex sets, and two people give more
different ones still. A deterministic head trained on one label per case is
forced to collapse that ambiguity into a single output, and what it converges to
is roughly the per-vertex mean -- which is not any labelling a human would draw.

WHY A NAKED RANDOM INPUT DOES NOT WORK. The obvious move is to concatenate a
random variable to the features and hope the model produces different regions
for different draws. It does not. Nothing in the loss rewards using the noise:
for any fixed input the loss is minimised by ignoring z and predicting the same
conditional mean as before, so gradient descent drives the noise weights to
zero. This is the standard noise-ignoring failure of conditional generative
models, and it is silent -- the model trains fine and simply produces identical
samples.

WHAT ACTUALLY MAKES z CARRY INFORMATION. The fix, following the Probabilistic
U-Net (Kohl et al. 2018), is to train z as a CVAE:

  prior(z | mesh)              what the model may sample at inference
  posterior(z | mesh, label)   sees the ACTUAL label, so it can encode which
                               of the plausible labellings this one is
  loss = segmentation loss on a sample from the POSTERIOR
       + beta * KL(posterior || prior)

Now z is useful: the posterior has to put the label's identity into z for the
segmentation loss to fall, and the KL forces the prior to cover the same range,
so sampling the prior at inference yields distinct plausible labellings.

HONEST LIMITATION. The Probabilistic U-Net was trained on LIDC, which has four
annotations per scan, so its latent learns genuine inter-annotator variation.
We have ONE label per case. The latent here can only model variation ACROSS
cases -- the residual the deterministic part cannot explain from geometry -- not
repeated disagreement on the same mesh. That is weaker, and it means this
variant should be judged by whether its samples look like labellings a human
might plausibly have drawn, not by a claim that it has learned annotator noise.
Collecting two labellings of the same shapes would upgrade this considerably.

z affects the region heads (cap localisation and dome) ONLY. Tangent, phi and
rotation stay deterministic: those are geometric quantities with a single right
answer, and making them stochastic would add variance with nothing to model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

from models.morphoformer import MorphoFormer


class _Gaussian(nn.Module):
    """Diagonal Gaussian head: vector -> (mu, logvar)."""

    def __init__(self, in_dim, latent_dim, hidden=None):
        super().__init__()
        h = hidden or max(in_dim, 2 * latent_dim)
        self.net = nn.Sequential(nn.Linear(in_dim, h), nn.ReLU(),
                                 nn.Linear(h, 2 * latent_dim))
        self.latent_dim = latent_dim

    def forward(self, x):
        mu, logvar = self.net(x).chunk(2, dim=-1)
        return mu, logvar.clamp(-8.0, 8.0)


class ProbMorphoFormer(MorphoFormer):
    """MorphoFormer whose region heads are conditioned on a latent z.

    Built by subclassing rather than editing MorphoFormer so the deterministic
    sensor -- which is the frozen FPD/KPD feature extractor -- cannot be
    disturbed by work on this variant.
    """

    def __init__(self, multi_recon, latent_dim=8, beta=1.0, **kwargs):
        super().__init__(multi_recon, **kwargs)
        hidden = self.loc_head.in_features
        self.latent_dim = latent_dim
        self.beta = beta

        # HOW z IS FUSED, and why not by concatenation.
        #
        # The obvious design -- broadcast z to every vertex, concatenate it onto
        # the feature, widen the head -- CANNOT WORK for the cap head, and fails
        # silently. The same z reaches every vertex, so a linear head turns it
        # into the SAME constant added to every vertex's logit, and the cap
        # distribution is a softmax OVER VERTICES, which is shift-invariant. The
        # constant cancels exactly. Measured: max |p(z1) - p(z2)| = 1.9e-09.
        #
        # So z modulates the features MULTIPLICATIVELY instead (FiLM, Perez et
        # al. 2018): it emits a per-channel scale and shift, and because the
        # features themselves differ from vertex to vertex, scaling them moves
        # vertices by different amounts. That survives the softmax.
        #
        # Zero-initialised, so gamma = beta = 0 gives the identity at step 0 and
        # the heads keep their original input width -- the deterministic sensor's
        # weights load exactly, with no widening.
        self.film = nn.Linear(latent_dim, 2 * hidden)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

        self.prior_net = _Gaussian(hidden, latent_dim)
        # Posterior additionally sees a summary of the label: features pooled
        # over each labelled region. Grounding the label in the SAME feature
        # space the segmentation reads means z has to encode where the boundary
        # was drawn, not re-encode the shape.
        n_regions = self.max_branches + (1 if self.dome_head is not None else 0)
        self.label_encoder = nn.Sequential(
            nn.Linear(hidden * n_regions, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden))
        self.posterior_net = _Gaussian(hidden * 2, latent_dim)

    # ---- initialisation from the deterministic sensor -----------------------

    def load_from_deterministic(self, state_dict, strict_shapes=True):
        """Copy a trained MorphoFormer into this one so fine-tuning starts from
        it rather than from scratch.

        Every shape matches, because FiLM leaves the heads at their original
        width. The FiLM layer is zero-initialised, so at step 0 this model
        reproduces the deterministic one exactly for any z. The latent starts
        with no effect and has to earn one, which is the right starting point
        for a fine-tune: nothing already learned is disturbed.
        """
        own = self.state_dict()
        loaded, skipped = [], []
        for k, v in state_dict.items():
            if k not in own:
                skipped.append(k); continue
            if own[k].shape == v.shape:
                own[k].copy_(v); loaded.append(k)
            elif strict_shapes:
                raise ValueError(f"shape mismatch for {k}: {tuple(v.shape)} -> {tuple(own[k].shape)}")
            else:
                skipped.append(k)
        self.load_state_dict(own)
        return loaded, skipped

    # ---- latent plumbing ----------------------------------------------------

    @staticmethod
    def _sample(mu, logvar):
        return mu + torch.randn_like(mu) * (0.5 * logvar).exp()

    def _encode_label(self, feat_dense, token_mask, in_patch, dome):
        """Pool per-vertex features over each labelled region -> one vector."""
        parts = []
        m = token_mask.unsqueeze(-1).float()
        for b in range(self.max_branches):
            w = (in_patch[:, b].float() * m.squeeze(-1)).unsqueeze(-1)
            parts.append((feat_dense * w).sum(1) / w.sum(1).clamp(min=1))
        if self.dome_head is not None:
            w = (dome.float() * m.squeeze(-1)).unsqueeze(-1) if dome is not None \
                else torch.zeros_like(m)
            parts.append((feat_dense * w).sum(1) / w.sum(1).clamp(min=1))
        return self.label_encoder(torch.cat(parts, dim=-1))

    def forward(self, phi=None, aneurysm_type=None, pyg_batch=None,
                in_patch=None, dome=None, z=None, n_samples=1):
        """Same 5-tuple as MorphoFormer, with extras.

        in_patch/dome: the ground-truth regions. Passing them selects the
        POSTERIOR (training). Omitting them samples the PRIOR (inference).
        n_samples > 1 returns several labellings for the same mesh, stacked on
        dim 0 of the region outputs -- that is the point of the variant.
        """
        if pyg_batch is None:
            assert phi is not None and aneurysm_type is not None
            pyg_batch = self.multi_recon.to_pyg_batch(
                phi, aneurysm_type, point_std=None, include_normals=self.use_normals)
        pos, feat, batch = self.encoder(pyg_batch)
        pos_dense, token_mask = to_dense_batch(pos, batch)
        feat_dense, _ = to_dense_batch(feat, batch)

        m = token_mask.unsqueeze(-1).float()
        embedding = (feat_dense * m).sum(1) / m.sum(1).clamp(min=1)

        p_mu, p_logvar = self.prior_net(embedding)
        out = {"embedding": embedding, "prior_mu": p_mu, "prior_logvar": p_logvar}

        if in_patch is not None:
            q_mu, q_logvar = self.posterior_net(
                torch.cat([embedding, self._encode_label(feat_dense, token_mask, in_patch, dome)], -1))
            out["posterior_mu"], out["posterior_logvar"] = q_mu, q_logvar
            # KL(posterior || prior), closed form for two diagonal Gaussians.
            out["kl"] = 0.5 * (p_logvar - q_logvar
                               + (q_logvar.exp() + (q_mu - p_mu) ** 2) / p_logvar.exp()
                               - 1.0).sum(-1).mean()
            src = (q_mu, q_logvar)
        else:
            src = (p_mu, p_logvar)

        zs = [z] if z is not None else [self._sample(*src) for _ in range(n_samples)]

        loc_list, dome_list = [], []
        for zi in zs:
            gamma, beta_ = self.film(zi).chunk(2, dim=-1)
            fz = feat_dense * (1.0 + gamma.unsqueeze(1)) + beta_.unsqueeze(1)
            logits = self.loc_head(fz).masked_fill(~token_mask.unsqueeze(-1), float("-inf"))
            loc_list.append(F.softmax(logits, dim=1).permute(0, 2, 1))
            if self.dome_head is not None:
                dome_list.append(self.dome_head(fz).squeeze(-1))

        loc_probs = loc_list[0]
        out["loc_samples"] = torch.stack(loc_list, 0)
        if dome_list:
            out["dome_logits"] = dome_list[0]
            out["dome_samples"] = torch.stack(dome_list, 0)

        # Deterministic heads, unchanged: one right answer, no latent.
        dir_logits = self.dir_head(feat_dense).masked_fill(
            ~token_mask.unsqueeze(-1), float("-inf"))
        dir_probs = F.softmax(dir_logits, dim=1).permute(0, 2, 1)
        endpoint = torch.einsum("bmn,bnc->bmc", loc_probs, pos_dense)
        tangent = F.normalize(
            self.tangent_head(torch.einsum("bmn,bnc->bmc", dir_probs, feat_dense)), dim=-1)
        if self.phi_head is not None:
            out["phi_pred"] = self.phi_head(embedding)
        if self.rotation_head is not None:
            out["rotation_pred"] = self.rotation_head(embedding)
        return endpoint, tangent, loc_probs, token_mask, out
