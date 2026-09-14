"""WGAN-GP critic over generated branch CENTERLINES.

The critic judges the curve itself, not the merged mesh. The mesh is built by a
deterministic, non-differentiable fusion (planarize, sweep, merge), so nothing
could back-propagate through it; and everything the branch VAE controls is
already in the curve. Feeding it the mesh would add a large fixed cost and a
gradient barrier for no extra signal.

WHY A 1-D CONV NET. The generator is an MLP over Fourier coefficients, and the
stage-1 critic is a PointNet++. A dilated 1-D convolution along arc length is a
third family, and it is the one that matches what makes a centerline look wrong:
curvature and torsion are LOCAL properties of consecutive samples, which a
convolution sees directly and a coefficient-space MLP does not.

No BatchNorm anywhere, for the same reason as the stage-1 critic: the gradient
penalty is defined per-sample and BatchNorm couples samples within a batch.
"""

import numpy as np
import torch
import torch.nn as nn

S_MAX = float(np.pi)      # arc-length range of the DST-I sine basis


def curve_from_coeffs(start, branch_vector, coeffs, n_points=128):
    """Differentiable torch port of skeleton_dataset_fourier.reconstruct_branch.

        p(s) = start + branch_vector * (s/pi) + sum_n coeffs[n] * sin(n*s)

    The numpy original cannot carry a gradient, and the critic has to push one
    back into the coefficients, so the sweep is rebuilt here in torch.

    start/branch_vector: [..., 3]   coeffs: [..., k, 3]  ->  [..., n_points, 3]
    """
    k = coeffs.shape[-2]
    dev, dt = coeffs.device, coeffs.dtype
    s = torch.linspace(0.0, S_MAX, n_points, device=dev, dtype=dt)          # [L]
    line = start.unsqueeze(-2) + branch_vector.unsqueeze(-2) * (s / S_MAX)[:, None]
    n = torch.arange(1, k + 1, device=dev, dtype=dt)                        # [k]
    A = torch.sin(s[:, None] * n[None, :])                                  # [L, k]
    return line + torch.einsum('lk,...kc->...lc', A, coeffs)


class CurveCritic(nn.Module):
    """Scores one branch centerline. Higher = more real.

    The curve is fed relative to its start point, so the critic cannot key on
    where in space the branch sits -- only on its shape. Aneurysm type and
    branch slot are supplied as embeddings, because a first branch off a
    bifurcation and a sidewall's outflow are not drawn from the same
    distribution and the critic should be allowed to know which it is looking at.
    """

    def __init__(self, hidden=128, n_types=3, max_branches=3, cond_dim=8):
        super().__init__()
        self.type_emb = nn.Embedding(n_types, cond_dim)
        self.slot_emb = nn.Embedding(max_branches, cond_dim)
        c = 3 + 2 * cond_dim
        act = lambda: nn.LeakyReLU(0.2, inplace=True)
        # Growing dilation: the receptive field reaches the whole curve without
        # striding away the fine spacing that curvature lives in.
        self.conv = nn.Sequential(
            nn.Conv1d(c, hidden, 5, padding=2), act(),
            nn.Conv1d(hidden, hidden, 5, padding=4, dilation=2), act(),
            nn.Conv1d(hidden, hidden, 5, padding=8, dilation=4), act(),
            nn.Conv1d(hidden, hidden, 5, padding=16, dilation=8), act(),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), act(),
            nn.Linear(hidden, hidden // 2), act(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, curve, types, slots):
        """curve [B, L, 3] (absolute), types [B], slots [B] -> [B] scores."""
        x = curve - curve[:, :1]                       # translation-invariant
        L = x.shape[1]
        ct = self.type_emb(types).unsqueeze(1).expand(-1, L, -1)
        cs = self.slot_emb(slots).unsqueeze(1).expand(-1, L, -1)
        h = torch.cat([x, ct, cs], dim=-1).transpose(1, 2)     # [B, C, L]
        h = self.conv(h)
        return self.head(torch.cat([h.amax(-1), h.mean(-1)], dim=-1)).squeeze(-1)


def gradient_penalty(critic, real, fake, types, slots):
    """Penalise gradient norm != 1 on curves interpolated between a real and a
    generated branch, which is what enforces the 1-Lipschitz condition that makes
    the critic's output a Wasserstein estimate."""
    b = real.size(0)
    eps = torch.rand(b, 1, 1, device=real.device, dtype=real.dtype)
    x = (eps * real + (1 - eps) * fake).detach().requires_grad_(True)
    g = torch.autograd.grad(critic(x, types, slots).sum(), x, create_graph=True)[0]
    return ((g.reshape(b, -1).norm(2, dim=1) - 1.0) ** 2).mean()
