"""
Endcap predictor — Plan C's replacement for both claydoll's ungrounded
start-point regression (models/branch_mlp_vae_claydoll.py's
GCNMeshEncoderWithStartPoints) and the original fixed canonical-index lookup
(MultiCanonicalGHDReconstruct.compute_branch_conditions). Predicts, per
branch slot, where a parent vessel's cap is (a point ON the mesh surface) and
which way it points (tangent direction) — trained on
dataset/preprocess_endcaps.py's output. Downstream use: snap the predicted
endpoint to its nearest mesh vertex, cast waves from there (per sample, not
from a fixed precomputed seed — see models/mesh_plugins.py's MeshPlugins for
the version this supersedes) to open the mesh and grow a local centerline/
tangent, then feed the ORIGINAL (pre-claydoll) branch generation model.

Design, worked out in conversation — this file went through two full
redesigns before landing here, both worth recording since the reasoning is
not obvious from the final code alone:

Redesign 1 — dropped GCN+SAGPooling+cross-attention (a learned per-branch
query attending over tokens, its attention weights doubling as the
soft-argmax/localization mechanism AND feeding a shared tangent head) in
favor of two independent PER-VERTEX CLASSIFIER heads (loc_head, dir_head).
Reasoning:
  - Directly using attention weights as the coordinate-computation mechanism
    (not just for feature mixing) has real precedent — "spatial softmax"
    (Finn et al. 2016) and "integral regression" (Sun et al. 2018) both
    softmax a per-keypoint heatmap over spatial locations and take the
    expectation as the coordinate. But in BOTH of those, the heatmap is a
    plain per-keypoint output channel of a decoder — no learned query, no
    key/value dot-product, no attention module. Reusing a full
    query/key/value cross-attention block (as this file originally did,
    borrowing SlotQueryCrossAttention built for a different, unrelated
    purpose) was more machinery than the problem needed, and unlike the
    classical technique's canonical form.
  - The one real design need attention was solving — letting each branch
    slot's spatial distribution depend on WHOLE-SHAPE context, not just a
    vertex's own local feature, since "which opening is branch 0 vs. 1 vs. 2"
    is inherently a relative/global judgment a purely local feature can't
    answer — turned out to have a much simpler fix (see Redesign 2).
  - Endpoint (loc_head) and tangent (dir_head) were decoupled into
    independent heads/parameters rather than sharing one distribution: a
    patch-supervised localization distribution is pulled tight around one
    point, but tangent estimation can genuinely need a wider or differently
    shaped receptive field (a near-planar patch can be locally ambiguous
    about direction; recognizing "this is a tube, tangent points along its
    axis" needs more context than the tight cap patch alone). Verified this
    decoupling is real, not cosmetic, by checking patch_loss produces zero
    gradient into dir_head's parameters and vice versa for tangent_loss.

Redesign 2 — dropped the separate FPS-pooled "deep" branch (built purely to
produce one pooled global-context vector, concatenated onto every vertex
before the per-vertex heads) in favor of GPSConv (torch_geometric.nn.GPSConv,
"Recipe for a General, Powerful, Scalable Graph Transformer", Rampasek et al.
2022) layers on top of a local GCN "stem". Reasoning:
  - GPSConv combines a local MPNN with GLOBAL full self-attention over every
    node in the graph, in each layer. That means every vertex's own feature
    becomes genuinely whole-mesh-aware directly, which likely subsumes what a
    single compressed pooled summary vector, re-broadcast onto every vertex,
    was providing — same signal, less directly.
  - Attention's query/key similarity is only as expressive as a linear
    projection of its input. Handing GPSConv's global attention raw
    xyz(+normal) directly, with no local processing first, gives the first
    attention layer very little structure to key off. A small local GCN
    "stem" first — cheap, local-inductive-bias layers that build up
    curvature/neighborhood structure before attention runs — mirrors "Early
    Convolutions Help Transformers See Better" (Xiao et al. 2021), which
    found the same thing for vision transformers fed raw pixels.
  - FPS accordingly has no remaining role here (it was only ever used to
    build the now-removed deep branch's pooled hierarchy).

Per-vertex input features: xyz position AND per-vertex normal (pytorch3d
Meshes.verts_normals_padded, averaged from adjacent face normals — see
MultiCanonicalGHDReconstruct.to_pyg_batch's include_normals option), on by
default for this model. Local surface orientation is exactly the kind of cue
that should help both finding a cap (openings/rims have a distinctive local
normal pattern relative to a smooth dome) and estimating tangent (a tube's
cross-sectional normals rotate consistently around its axis).

Loss — endpoint_loss (masked MSE) + tangent_loss (masked 1-cosine) + a
patch_loss auxiliary term on loc_head's distribution only: naive one-hot
"nearest vertex" cross-entropy is blind to how wrong a wrong answer is (only
looks at probability mass on the true class, so an adjacent-vertex mistake
costs the same as a far-side-of-the-mesh mistake) — the fix is a soft target
with real spatial extent, dataset/preprocess_endcaps.py's in_patch (vertices
within PATCH_THRESHOLD_MM of GRAPH distance, not Euclidean, since the mesh
can fold back on itself e.g. across a thin vessel wall). patch_loss is
cross-entropy against a uniform distribution over in_patch, equivalent up to
a constant to KL(uniform-over-patch || predicted loc_head distribution).

conda activate new
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GPSConv
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_dense_batch


class RandomSO3Rotation(BaseTransform):
    """Applies an independent, uniformly random SO(3) rotation to each graph
    in a (possibly batched) PyG Data/Batch — rotates position (x[:, :3]) and,
    if present, normal (x[:, 3:6]) channels by the same matrix (no
    translation, so normals stay consistent with the rotated surface);
    edge_index/batch are untouched (rotation doesn't change connectivity).

    Without this, the network only ever sees the canonical mesh's one fixed
    orientation — nothing stops it from learning "the cap is roughly at this
    absolute xyz" instead of genuinely orientation-independent local
    geometric cues, which is exactly the kind of shortcut that would fail to
    generalize to the generated (not just real) shapes this whole redesign is
    for. One independent rotation per graph in the batch (via
    scipy.spatial.transform.Rotation.random(), not one shared rotation for
    the whole mini-batch), so a single training batch already samples diverse
    orientations rather than repeating one per step.

    Applied at the DATA level (scripts/train/train_morphoformer.py's
    prepare_batch), not inside the model: rotating only the mesh input would
    leave it inconsistent with the ground-truth endpoint/tangent targets,
    which are only ever known in the canonical (unrotated) frame — those need
    to be rotated by the exact same per-graph matrices. sample/apply_to_data/
    apply_to_points are split out (rather than doing everything in one
    forward() call) specifically so the caller can generate the rotations
    once and apply the SAME ones to both the PyG mesh batch and the separate
    endpoint/tangent label tensors, which live outside the PyG Data object
    entirely and can't be reached by a plain BaseTransform.
    """

    @staticmethod
    def sample(num_graphs, dtype, device):
        import numpy as np
        from scipy.spatial.transform import Rotation
        mats = np.stack([Rotation.random().as_matrix() for _ in range(num_graphs)])
        return torch.as_tensor(mats, dtype=dtype, device=device)   # [num_graphs, 3, 3]

    @staticmethod
    def apply_to_data(data, rot):
        """rot: [num_graphs, 3, 3], one rotation per graph in the (possibly
        batched) data. Returns a new Data/Batch; does not mutate in place."""
        batch = data.batch if data.batch is not None else torch.zeros(
            data.x.size(0), dtype=torch.long, device=data.x.device)
        R = rot[batch]                                                          # [total_N, 3, 3]

        x = data.x.clone()
        x[:, :3] = torch.einsum('nij,nj->ni', R, data.x[:, :3])
        if x.size(1) >= 6:
            x[:, 3:6] = torch.einsum('nij,nj->ni', R, data.x[:, 3:6])
        data.x = x
        return data

    @staticmethod
    def apply_to_points(rot, points):
        """rot: [B, 3, 3]; points: [B, mb, 3] (positions or directions,
        rotation has no translation component so both are valid) -> [B, mb, 3]."""
        return torch.einsum('bij,bmj->bmi', rot, points)

    def forward(self, data):
        """Standalone BaseTransform usage (mesh only, no auxiliary targets to
        keep in sync) — samples and applies in one step."""
        batch = data.batch if data.batch is not None else torch.zeros(
            data.x.size(0), dtype=torch.long, device=data.x.device)
        num_graphs = int(batch.max().item()) + 1
        rot = self.sample(num_graphs, data.x.dtype, data.x.device)
        return self.apply_to_data(data, rot)


class GCNGPSEncoder(nn.Module):
    """Local GCN 'stem' (plain message passing over the full, unpooled
    vertex graph) feeding GPSConv layers (local MPNN + global full
    self-attention per layer). See module docstring for why this ordering
    and why there's no separate pooled global-context branch.

    Input: PyG Batch with x [total_N, 3] or [total_N, 6] (xyz, optionally
    +normal), edge_index, batch. Output: (pos, feat, batch) at full input
    resolution — pos is always just the xyz sub-slice of x (never the
    normal channels), since it's later used for real spatial computations
    (soft-argmax over positions), not feature learning.
    """

    def __init__(self, in_channels=6, hidden=32, stem_layers=2, gps_layers=2, gps_heads=4):
        super().__init__()
        self.stem = nn.ModuleList()
        c_in = in_channels
        for _ in range(stem_layers):
            self.stem.append(GCNConv(c_in, hidden))
            c_in = hidden

        self.gps_layers = nn.ModuleList([
            GPSConv(hidden, conv=GCNConv(hidden, hidden), heads=gps_heads)
            for _ in range(gps_layers)
        ])

    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        pos = x[:, :3]

        h = x
        for conv in self.stem:
            h = F.relu(conv(h, ei))
        for gps in self.gps_layers:
            h = gps(h, ei, batch)   # GPSConv applies its own internal norm/activation/FFN

        return pos, h, batch


class MorphoFormer(nn.Module):
    def __init__(self, multi_recon, max_branches=3, hidden=32, stem_layers=2, gps_layers=2,
                gps_heads=4, use_normals=True, predict_dome=False, predict_phi=False,
                predict_rotation=False, phi_dim=432, rotation_dim=6):
        """predict_dome / predict_phi select which of the two models this is.

        Two variants are trained from the same class:

          uncapping   predict_dome=False  -- localises the caps so a mesh can
                      be opened. Extra heads would only add gradient noise to
                      the task that actually has to work at inference.

          descriptor  predict_dome=True, predict_phi=True -- the feature
                      extractor behind the FPD/KPD generation metrics. Dome
                      supervision forces the features to encode SAC shape
                      rather than just openings; the phi head forces them to
                      retain global geometry, since region localisation alone
                      can be satisfied without representing the whole shape
                      (and, being a 432-d continuous target, it is much harder
                      to satisfy by memorising individual meshes -- which
                      matters because this extractor is trained on the same
                      real corpus that serves as the metric's reference).
        """
        super().__init__()
        self.multi_recon = multi_recon
        self.max_branches = max_branches
        self.use_normals = use_normals
        self.predict_dome = predict_dome
        self.predict_phi = predict_phi
        self.predict_rotation = predict_rotation

        in_channels = 6 if use_normals else 3
        self.encoder = GCNGPSEncoder(in_channels=in_channels, hidden=hidden, stem_layers=stem_layers,
                                     gps_layers=gps_layers, gps_heads=gps_heads)

        # Per-vertex classifier heads, one channel per branch slot each —
        # branch identity comes from the output channel index (like a
        # per-joint heatmap CNN), not from a learned query. Independent
        # parameters, deliberately not sharing weights: see Redesign 1.
        self.loc_head = nn.Linear(hidden, max_branches)
        self.dir_head = nn.Linear(hidden, max_branches)
        self.tangent_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 3),
        )
        # Per-vertex binary segmentation: dome vs. not. One logit per vertex,
        # unlike loc/dir_head's per-branch channels -- there is exactly one dome.
        self.dome_head = nn.Linear(hidden, 1) if predict_dome else None
        # Auxiliary reconstruction from the POOLED embedding (not per-vertex),
        # so the pressure to retain shape lands on the global feature that the
        # FPD/KPD metrics actually consume.
        self.phi_head = (nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                       nn.Linear(hidden, phi_dim))
                         if predict_phi else None)
        # Separate head rather than extra channels on phi_head: the two targets
        # differ by orders of magnitude in scale, so sharing a final layer would
        # let the 432-d phi term swamp the 6-d rotation term's gradients.
        #
        # Predicts the RANDOM ROTATION applied to the input mesh, as the 6D
        # representation (first two columns of R). Two deliberate choices:
        #
        #   * The target is the augmentation rotation ALONE. It used to be
        #     Q @ R_fitted, where R_fitted is the GHD Stage-2 pose linking the
        #     warped canonical to the real patient complex. That pose never
        #     touches the mesh the network is shown -- the input is rebuilt from
        #     phi and then rotated -- so it was unrecoverable by construction,
        #     and asking for it fed pure noise into the shared embedding.
        #
        #   * 6D, not axis-angle. No 3- or 4-dimensional parameterisation of
        #     SO(3) is continuous (Zhou et al. 2019), and axis-angle breaks at
        #     pi where the axis flips sign, so no continuous network output can
        #     match it there. Measured on this corpus, 82% of the old rotation
        #     loss came from the worst 2% of samples: representation blow-up,
        #     not geometry.
        self.rotation_head = (nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                          nn.Linear(hidden, rotation_dim))
                            if predict_rotation else None)

    def forward(self, phi=None, aneurysm_type=None, pyg_batch=None):
        """Returns (endpoint [B,mb,3], tangent [B,mb,3] unit, loc_probs
        [B,mb,N] (loc_head's per-vertex distribution, for patch_loss),
        token_mask [B,N]).

        pyg_batch: pass a pre-built (and, at train time, pre-rotated —
        see RandomSO3Rotation / prepare_batch) PyG Batch directly, e.g. from
        the training loop. If omitted, one is built here from phi/
        aneurysm_type (the canonical, unrotated mesh) — the convenient path
        for eval/sanity-check call sites that just want model(phi, aneurysm_type).
        """
        if pyg_batch is None:
            assert phi is not None and aneurysm_type is not None, \
                "forward needs either pyg_batch, or both phi and aneurysm_type"
            pyg_batch = self.multi_recon.to_pyg_batch(
                phi, aneurysm_type, point_std=None, include_normals=self.use_normals)
        pos, feat, batch = self.encoder(pyg_batch)

        pos_dense, token_mask = to_dense_batch(pos, batch)      # [B, N, 3], [B, N]
        feat_dense, _ = to_dense_batch(feat, batch)              # [B, N, hidden]

        loc_logits = self.loc_head(feat_dense)                   # [B, N, mb]
        dir_logits = self.dir_head(feat_dense)                   # [B, N, mb]
        pad = ~token_mask.unsqueeze(-1)
        loc_logits = loc_logits.masked_fill(pad, float('-inf'))
        dir_logits = dir_logits.masked_fill(pad, float('-inf'))

        loc_probs = F.softmax(loc_logits, dim=1).permute(0, 2, 1)   # softmax over vertices -> [B, mb, N]
        dir_probs = F.softmax(dir_logits, dim=1).permute(0, 2, 1)   # independent distribution, [B, mb, N]

        endpoint = torch.einsum('bmn,bnc->bmc', loc_probs, pos_dense)   # soft-argmax over positions
        dir_feat = torch.einsum('bmn,bnc->bmc', dir_probs, feat_dense)  # weighted pool over features
        tangent = F.normalize(self.tangent_head(dir_feat), dim=-1)

        out = {"endpoint": endpoint, "tangent": tangent, "loc_probs": loc_probs,
               "token_mask": token_mask}
        # Masked mean over real vertices -- the global descriptor. This is the
        # vector FPD/KPD are computed on, so it is always produced, not only
        # when the auxiliary heads exist.
        m = token_mask.unsqueeze(-1).float()
        out["embedding"] = (feat_dense * m).sum(1) / m.sum(1).clamp(min=1)
        if self.dome_head is not None:
            out["dome_logits"] = self.dome_head(feat_dense).squeeze(-1)   # [B, N]
        if self.phi_head is not None:
            out["phi_pred"] = self.phi_head(out["embedding"])             # [B, phi_dim]
        if self.rotation_head is not None:
            out["rotation_pred"] = self.rotation_head(out["embedding"])       # [B, rotation_dim]
        # Tuple return kept for the existing call sites, which unpack 4 values.
        return endpoint, tangent, loc_probs, token_mask, out


    @torch.no_grad()
    def embed(self, phi=None, aneurysm_type=None, pyg_batch=None):
        """Pooled per-mesh descriptor [B, hidden] -- the FPD/KPD feature."""
        return self.forward(phi, aneurysm_type, pyg_batch)[4]["embedding"]

    def get_loss(self, endpoint_pred, tangent_pred, loc_probs, token_mask,
                endpoint_real, tangent_real, branch_mask, in_patch, node_batch):
        """loc_probs is loc_head's distribution only (see forward) — patch_loss
        below deliberately never touches dir_head's parameters, which are
        left free to shape themselves however best serves tangent_loss."""
        m = branch_mask.unsqueeze(-1).float()
        denom = branch_mask.float().sum().clamp(min=1.0)

        endpoint_loss = ((endpoint_pred - endpoint_real) ** 2 * m).sum() / (denom * 3)

        cos_sim = F.cosine_similarity(tangent_pred, tangent_real, dim=-1)
        tangent_loss = ((1.0 - cos_sim) * branch_mask.float()).sum() / denom

        # in_patch: [mb, total_N] (dataset-batched, node dim concatenated in
        # sample order) -> dense [B, mb, N] to align with loc_probs/token_mask.
        in_patch_dense, patch_mask = to_dense_batch(in_patch.t().float(), node_batch)  # [B, N, mb], [B, N]
        in_patch_dense = in_patch_dense.permute(0, 2, 1)                               # [B, mb, N]
        valid = in_patch_dense * token_mask.unsqueeze(1).float()
        patch_count = valid.sum(-1).clamp(min=1.0)                                     # [B, mb]
        log_probs = torch.log(loc_probs.clamp(min=1e-8))
        per_branch = -(valid * log_probs).sum(-1) / patch_count                        # [B, mb]
        patch_loss = (per_branch * branch_mask.float()).sum() / denom

        return endpoint_loss, tangent_loss, patch_loss
