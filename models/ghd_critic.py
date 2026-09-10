"""PointNet++ WGAN-GP critic over GHD-reconstructed aneurysm meshes."""

import torch
import torch.nn as nn
from torch_cluster import fps, knn
from torch_geometric.nn import PointNetConv, global_max_pool


def _mlp(dims, final_act=True):
    """LayerNorm, never BatchNorm.

    WGAN-GP's gradient penalty constrains d(critic)/d(one input), so any layer
    that mixes samples across the batch makes the penalty ill-defined. The
    reference PointNet++ uses BatchNorm throughout, which is exactly the thing
    that cannot be carried over. LayerNorm normalises each point's own feature
    vector, so it is sample-independent and safe.
    """
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or final_act:
            layers += [nn.LayerNorm(dims[i + 1]), nn.LeakyReLU(0.2, inplace=True)]
    return nn.Sequential(*layers)


class SAModule(nn.Module):
    """PointNet++ set abstraction: farthest-point-sample centroids, group their
    kNN, run a shared PointNet on each local neighbourhood, max-pool.

    kNN grouping rather than the paper's ball query. Ball query needs a radius
    in absolute units, and these meshes vary in physical scale case to case, so
    a fixed radius would gather a different amount of surface on a large
    aneurysm than a small one. kNN adapts to local point density instead.
    """

    def __init__(self, ratio, k, mlp):
        super().__init__()
        self.ratio, self.k = ratio, k
        self.conv = PointNetConv(mlp, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = knn(pos, pos[idx], self.k, batch, batch[idx])
        edge_index = torch.stack([col, row], dim=0)      # source -> centroid
        x = self.conv((x, None if x is None else x[idx]), (pos, pos[idx]), edge_index)
        return x, pos[idx], batch[idx]


class PointNetPPCritic(nn.Module):
    """Hierarchical critic. Two set-abstraction levels then a global one, so it
    judges local surface detail and overall silhouette rather than only the
    global shape a flat PointNet would see.

    WHY NOT A GRAPH TRANSFORMER, given both were on the table. The FPD/KPD
    evaluator is a GCN + GPS graph transformer. If the critic shared that
    architecture the generator would be trained to satisfy the evaluator's own
    inductive bias, and the metric could improve without the shapes improving.
    Different families keep the evaluation honest.
    """

    def __init__(self, hidden=128, num_types=3, cond_dim=8, k=16, ratio=0.25):
        super().__init__()
        self.type_emb = nn.Embedding(num_types, cond_dim)
        self.sa1 = SAModule(ratio, k, _mlp([cond_dim + 3, 64, 64]))
        self.sa2 = SAModule(ratio, k, _mlp([64 + 3, hidden, hidden]))
        self.global_mlp = _mlp([hidden + 3, 2 * hidden, 2 * hidden])
        self.head = _mlp([2 * hidden, hidden, 1], final_act=False)

    def forward(self, verts, types):
        """verts [B, N, 3] for ONE aneurysm type, types [B] -> [B] scores.

        One type per call: the two canonical templates have different vertex
        counts, so they cannot share a padded tensor without the padding itself
        becoming a cue the critic can exploit.
        """
        b, n, _ = verts.shape
        pos = verts.reshape(b * n, 3)
        batch = torch.arange(b, device=verts.device).repeat_interleave(n)
        x = self.type_emb(types).repeat_interleave(n, dim=0)
        x, pos1, b1 = self.sa1(x, pos, batch)
        x, pos2, b2 = self.sa2(x, pos1, b1)
        x = global_max_pool(self.global_mlp(torch.cat([x, pos2], dim=1)), b2)
        return self.head(x).squeeze(-1)


def gradient_penalty(critic, real_verts, fake_verts, types):
    """Penalise the critic for gradient norm != 1 on points interpolated between
    a real and a generated mesh. This is what enforces the 1-Lipschitz condition
    that makes the critic's output a Wasserstein estimate rather than an
    arbitrary score."""
    b = real_verts.size(0)
    eps = torch.rand(b, 1, 1, device=real_verts.device, dtype=real_verts.dtype)
    x = (eps * real_verts + (1 - eps) * fake_verts).detach().requires_grad_(True)
    g = torch.autograd.grad(critic(x, types).sum(), x, create_graph=True)[0]
    return ((g.reshape(b, -1).norm(2, dim=1) - 1.0) ** 2).mean()
