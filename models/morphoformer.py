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

import collections
from pathlib import Path

import numpy as np
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


def region_from_probs(probs_b, frac=0.2, min_ratio=3.0, max_k=400):
    """Vertices the model actually calls this cap, from one row of loc_probs.

    Canonical copy. loc_probs is a softmax over ALL vertices, so absolute values
    are tiny and a fixed threshold is meaningless -- hence `frac` of the
    branch's own peak. That alone misbehaves when the head is untrained: the
    distribution is then near-uniform, peak ~= 1/N, and 0.2*peak selects almost
    every vertex, blanketing the mesh with something that looks like a
    prediction but carries no information.

    Two guards. `min_ratio` demands a vertex be at least that many times uniform
    probability, so an untrained head draws nothing rather than everything.
    `max_k` caps the count near a real cap's size so the region stays readable
    once the head sharpens.
    """
    if probs_b is None or probs_b.size == 0:
        return np.zeros(0, dtype=int)
    n = probs_b.size
    peak = float(probs_b.max())
    if peak <= 0:
        return np.zeros(0, dtype=int)
    thr = max(frac * peak, min_ratio / n)
    idx = np.flatnonzero(probs_b >= thr)
    if idx.size > max_k:
        idx = idx[np.argsort(probs_b[idx])[-max_k:]]
    return idx


class SensorUncapper:
    """Loads a trained morphology sensor and uses it to uncap meshes.

    Replaces the old opening-index mechanism. That design marked a fixed set of
    canonical vertices as "the opening ring" and reused them on every deformed
    mesh, which stops being true once GHD warps the template: measured against
    real centerlines, the plane normal of those stale rings is off by a mean of
    52 degrees, while the sensor's tangent is off by 10.

    THE TANGENT COMES FROM THE SENSOR, not from a plane fit to the cut. Fitting
    a plane to the rim works when the rim happens to be planar and fails when it
    is not, and on this corpus that failure is common. Grouped by rim flatness
    (ratio of the smallest to the middle singular value), error against the real
    centerline tangent:

        flatness    n     plane fit    sensor
        0.00-0.08   58       8.0 deg   8.9 deg
        0.08-0.15  210       9.1       9.2
        0.15-0.25   63      14.1      12.8
        0.25-1.00   51      34.8      10.6

    The two agree while the rim is flat, and only the plane fit degrades. The
    plane fit is still computed, but purely as a health flag -- see `rim_flatness`
    and `svd_disagreement_deg` in the returned dict -- never as the answer.
    """

    def __init__(self, model, multi_recon, device=None):
        # Frozen, permanently. Nothing downstream harvests a loss from the
        # uncapping or the merged mesh -- the sensor is a fixed measuring
        # instrument here, and a stage-2 optimiser built over model.parameters()
        # must never pick these up. eval() alone would not prevent that.
        self.model = model.eval()
        for prm in self.model.parameters():
            prm.requires_grad_(False)
        self.multi_recon = multi_recon
        self.device = device or next(model.parameters()).device
        self.is_probabilistic = any("prior_net" in n for n, _ in model.named_parameters())

    @classmethod
    def from_checkpoint(cls, ckpt_path, multi_recon, device="cpu", latent_dim=None):
        """Build from a saved sensor. Detects the probabilistic variant from the
        weights themselves rather than trusting a flag, so a checkpoint saved by
        an older script still loads correctly."""
        device = torch.device(device)
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        prob = any(k.startswith("prior_net") for k in ck["model"])
        if prob:
            from models.morphoformer_prob import ProbMorphoFormer
            ld = latent_dim or ck.get("hparams", {}).get("latent_dim") \
                 or ck["model"]["prior_net.net.2.bias"].shape[0] // 2
            model = ProbMorphoFormer(multi_recon, latent_dim=ld, **ck["args"])
        else:
            model = MorphoFormer(multi_recon, **ck["args"])
        model.load_state_dict(ck["model"])
        return cls(model.to(device), multi_recon, device)

    # ---- core ---------------------------------------------------------------

    @torch.no_grad()
    def _predict(self, phi, aneurysm_type, sample=False):
        """Sensor outputs for a same-type batch. For the probabilistic variant,
        sample=False uses the prior MEAN -- the single most likely labelling,
        which is what a deterministic pipeline needs. sample=True draws from the
        prior, giving a different plausible uncapping each call."""
        phi = torch.as_tensor(phi, dtype=torch.float32, device=self.device)
        if phi.dim() == 2:
            phi = phi[None]
        at = torch.as_tensor(aneurysm_type, dtype=torch.long, device=self.device)
        if at.dim() == 0:
            at = at.expand(phi.size(0))
        if self.is_probabilistic and not sample:
            pyg = self.multi_recon.to_pyg_batch(phi, at, point_std=None,
                                                include_normals=self.model.use_normals)
            _, feat, batch = self.model.encoder(pyg)
            fd, tok = to_dense_batch(feat, batch)
            m = tok.unsqueeze(-1).float()
            mu, _ = self.model.prior_net((fd * m).sum(1) / m.sum(1).clamp(min=1))
            return self.model(phi, at, z=mu)
        return self.model(phi, at)

    @staticmethod
    def _np(x):
        """phi may arrive as a CUDA tensor from branch_conditions."""
        if torch.is_tensor(x):
            x = x.detach().cpu()
        return np.asarray(x, dtype=np.float32)

    @staticmethod
    def push_outside(start, rim_pts, tangent, margin_frac=0.10):
        """Guarantee the branch start sits OUTSIDE the cut plane.

        The tube is lofted from the planarized rim along the generated
        centerline, and that centerline begins at this start point. If the start
        falls behind the rim plane the first ring lands inside the dome and the
        merged mesh self-intersects. Measured on synthetic shapes the sensor's
        endpoint is outside by +0.26 on average, but it lands INSIDE on 9% of
        branches, by a median of 0.13 and up to 0.79 -- so this is not rare
        enough to leave to chance.

        The margin scales with the opening's own radius, so a small branch is
        not pushed as far as a large one.
        """
        if len(rim_pts) < 3:
            return start, 0.0
        cen = rim_pts.mean(0)
        radius = float(np.linalg.norm(rim_pts - cen, axis=1).mean())
        margin = margin_frac * radius
        d = float((start - cen) @ tangent)
        if d >= margin:
            return start, 0.0
        return start + (margin - d) * tangent, margin - d

    @torch.no_grad()
    def uncap(self, phi, aneurysm_type, sample=False, region_kwargs=None,
              margin_frac=0.10):
        """Open one mesh. Returns a dict with the cut mesh and per-branch conditions.

        Faces wholly inside a predicted cap are deleted, which leaves a clean
        boundary loop; deleting every face that merely touches the cap would eat
        a ring of extra geometry and push the rim outward.
        """
        ep, tg, loc, tok, extra = self._predict(phi, aneurysm_type, sample)
        atype = int(torch.as_tensor(aneurysm_type).flatten()[0])
        nv = int(tok[0].sum())
        verts = self.multi_recon._reconstruct_verts_np(self._np(phi).reshape(-1, 3), atype)
        faces = self.multi_recon.get(atype).canonical_Meshes.faces_packed().cpu().numpy()
        n_open = len(self.multi_recon._load_openings(atype))
        rk = region_kwargs or {}

        caps, rims, starts, dirs, flat, disag = [], [], [], [], [], []
        n_islands, n_dropped, n_filled, n_pushed, ok = [], [], [], [], []
        cut = np.zeros(nv, dtype=bool)
        for b in range(n_open):
            cap = np.zeros(nv, dtype=bool)
            cap[region_from_probs(loc[0, b, :nv].cpu().numpy(), **rk)] = True
            cap, ncomp, dropped = self.largest_component(cap, faces)
            cap, nfilled = self.fill_holes(cap, faces)
            n_islands.append(ncomp - 1); n_dropped.append(dropped); n_filled.append(nfilled)
            caps.append(cap)
            cut |= cap
            rim = self._rim(cap, faces)
            rims.append(rim)
            t = tg[0, b].cpu().numpy()
            t = t / (np.linalg.norm(t) + 1e-9)
            dirs.append(t)
            st, pushed = self.push_outside(ep[0, b].cpu().numpy(), verts[rim], t, margin_frac)
            starts.append(st); n_pushed.append(pushed)
            # a rim whose longest loop misses most of the boundary is split and
            # cannot be lofted onto; flagged rather than silently swept
            nl, dmax = self.rim_topology(cap, faces)
            ok.append(bool(len(rim) >= 5 and nl == 1 and dmax == 2))
            f, d = self._plane_check(verts, rim, t, verts.mean(0))
            flat.append(f); disag.append(d)

        keep = ~cut[faces].all(1)
        return {
            "verts": verts, "faces": faces[keep], "faces_removed": int((~keep).sum()),
            "cap_masks": np.stack(caps), "rims": rims,
            "start_points": np.stack(starts), "directions": np.stack(dirs),
            "branch_mask": np.ones(n_open, dtype=bool),
            "dome": (torch.sigmoid(extra["dome_logits"][0, :nv]).cpu().numpy() > 0.5
                     if "dome_logits" in extra else None),
            # health flags, not answers
            "rim_flatness": np.array(flat), "svd_disagreement_deg": np.array(disag),
            "n_islands_dropped": np.array(n_islands), "verts_dropped": np.array(n_dropped),
            "verts_hole_filled": np.array(n_filled),
            "endpoint_pushed": np.array(n_pushed), "rim_ok": np.array(ok),
        }

    INSPECT_DIR = "runtime_train/synthetic_pool_inspect"

    def validate(self, phi, aneurysm_type, sample=False):
        """Per-branch usability of a shape's uncapping, without building tubes.

        A branch is unusable when its cap's boundary does not close into one
        loop, which happens when the predicted region wraps a thin branch and
        meets itself. Such a cap has two rims and cannot be lofted onto.
        """
        o = self.uncap(phi, aneurysm_type, sample=sample)
        ok = np.asarray(o["rim_ok"], dtype=bool)
        return {"ok": bool(ok.all()), "rim_ok": ok,
                "n_failed": int((~ok).sum()), "uncap": o}

    def quarantine(self, phi, aneurysm_type, case=None, provenance=None,
                   out_dir=None, sample=False, extra=None):
        """Set a shape aside for inspection if its uncapping fails.

        Returns the written path, or None when the shape is fine. The record
        matches the synthetic-pool schema so the inspect pool loads with the
        same tooling, plus which branches failed and why.

        Quarantined shapes must NOT feed the stage-2 direction loss and must NOT
        be merged: a split rim yields a collapsed tube, and a collapsed tube
        would teach the branch VAE to aim at geometry that does not exist.
        """
        v = self.validate(phi, aneurysm_type, sample=sample)
        if v["ok"]:
            return None
        out = Path(out_dir or self.INSPECT_DIR)
        out.mkdir(parents=True, exist_ok=True)
        case = case or f"failed_{abs(hash(self._np(phi).tobytes())) % (10**10):010d}"
        rec = {"case": case, "aneurysm_type": int(torch.as_tensor(aneurysm_type).flatten()[0]),
               "phi": self._np(phi).reshape(-1, 3),
               "is_synthetic": True, "uncap_failed": True,
               "rim_ok": v["rim_ok"], "n_failed": v["n_failed"],
               "rim_sizes": np.array([len(r) for r in v["uncap"]["rims"]]),
               "cap_sizes": v["uncap"]["cap_masks"].sum(1),
               "provenance": provenance or {}}
        if extra:
            rec.update(extra)
        path = out / f"{case}.npy"
        np.save(path, rec, allow_pickle=True)
        return path

    @torch.no_grad()
    def branch_conditions(self, phi, aneurysm_types, max_branches=3, sample=False,
                          skip_failed=False):
        """Drop-in replacement for MultiCanonicalGHDReconstruct.compute_branch_conditions.

        Same (start_points, directions, mask) signature and the same padded
        widths, so stage-2 call sites swap one for the other. Unlike the
        original it accepts MIXED types in a batch, looping per type internally.
        """
        phi = torch.as_tensor(phi, dtype=torch.float32, device=self.device)
        at = torch.as_tensor(aneurysm_types, dtype=torch.long, device=self.device)
        if at.dim() == 0:
            at = at.expand(phi.size(0))
        B = phi.size(0)
        starts = phi.new_zeros(B, max_branches, 3)
        dirs = phi.new_zeros(B, max_branches, 3)
        mask = torch.zeros(B, max_branches, dtype=torch.bool, device=self.device)
        for t in sorted({int(x) for x in at.tolist()}):
            sel = (at == t).nonzero(as_tuple=True)[0]
            ep, tg, loc, tok, _ = self._predict(phi[sel], at[sel], sample)
            n_open = min(len(self.multi_recon._load_openings(t)), max_branches)
            starts[sel, :n_open] = ep[:, :n_open]
            dirs[sel, :n_open] = F.normalize(tg[:, :n_open], dim=-1)
            mask[sel, :n_open] = True
            if skip_failed:
                # Clear the mask on branches whose rim will not close, so the
                # stage-2 direction loss (which averages over this mask) takes
                # no target from a cap that cannot be swept.
                #
                # The rim check reuses loc_probs from the batched forward above.
                # Calling uncap() per sample here instead re-ran the encoder once
                # per sample -- B+1 forward passes rather than 1 -- which
                # dominated the stage-2 step time.
                faces = self.multi_recon.get(t).canonical_Meshes.faces_packed().cpu().numpy()
                loc_np = loc[:, :n_open].detach().cpu().numpy()
                nv = int(tok[0].sum())
                for j, i in enumerate(sel.tolist()):
                    for b in range(n_open):
                        cap = np.zeros(nv, dtype=bool)
                        cap[region_from_probs(loc_np[j, b, :nv])] = True
                        cap, _, _ = self.largest_component(cap, faces)
                        cap, _ = self.fill_holes(cap, faces)
                        rim = self._rim(cap, faces)
                        nl, dmax = self.rim_topology(cap, faces)
                        if len(rim) < 5 or nl != 1 or dmax != 2:
                            mask[i, b] = False
        return starts, dirs, mask

    @torch.no_grad()
    def fused_mesh(self, phi, aneurysm_type, branch_points, branch_mask=None,
                   sample=False, **kwargs):
        """Full stage-1 + stage-2 mesh: uncap with the sensor, planarize the cut
        openings, then sweep the vessel cross-section along each generated
        centerline and merge.

        Thin wrapper over MultiCanonicalGHDReconstruct.reconstruct_fused_mesh,
        handing it the SENSOR'S openings instead of the precomputed canonical
        rings.
        """
        atype = int(torch.as_tensor(aneurysm_type).flatten()[0])
        ep, tg, loc, tok, _ = self._predict(phi, aneurysm_type, sample)
        nv = int(tok[0].sum())
        faces = self.multi_recon.get(atype).canonical_Meshes.faces_packed().cpu().numpy()
        n_open = len(self.multi_recon._load_openings(atype))

        loops, cut_masks, cut = [], [], np.zeros(nv, dtype=bool)
        for b in range(n_open):
            cap = np.zeros(nv, dtype=bool)
            cap[region_from_probs(loc[0, b, :nv].cpu().numpy())] = True
            cap, _, _ = self.largest_component(cap, faces)
            cap, _ = self.fill_holes(cap, faces)
            cut |= cap
            cut_masks.append(cap)
            loops.append(self.rim_loop(cap, faces))
        # Skip rather than sweep a broken rim: a split boundary produces a
        # collapsed tube, which is worse than no branch at all. Length alone is
        # not the test -- the split case here has loops of 27 and 17, both long
        # enough to look fine. The test is whether the LONGEST loop accounts for
        # essentially the whole boundary.
        for b, cm in enumerate(cut_masks):
            nl, dmax = self.rim_topology(cm, faces)
            if len(loops[b]) < 5 or nl != 1 or dmax != 2:
                return None
        trimmed = faces[~cut[faces].all(1)]
        return self.multi_recon.reconstruct_fused_mesh(
            self._np(phi).reshape(-1, 3), atype,
            branch_points, branch_mask,
            opening_indices=loops, trimmed_faces=trimmed,
            opening_normals=[F.normalize(tg[0, b], dim=-1).cpu().numpy()
                             for b in range(n_open)], **kwargs)

    # ---- helpers ------------------------------------------------------------

    @staticmethod
    def largest_component(cap, faces):
        """Keep only the biggest connected patch of a predicted cap.

        The loc head is a softmax over every vertex, so a confident cap usually
        comes with a few stray high-probability vertices elsewhere on the
        surface. Those islands are not part of the opening, but they make the
        cap non-simply-connected, which splits its boundary into several loops
        and leaves the tube sweeper with no single rim to loft onto. Discarding
        everything but the largest component removes them.

        Returns (kept_mask, n_components, discarded_vertex_count).
        """
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        idx = np.flatnonzero(cap)
        if idx.size == 0:
            return cap, 0, 0
        pos = -np.ones(cap.shape[0], dtype=int)
        pos[idx] = np.arange(idx.size)
        e = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            m = cap[faces[:, a]] & cap[faces[:, b]]
            if m.any():
                e.append(np.stack([pos[faces[m, a]], pos[faces[m, b]]], 1))
        if not e:
            return cap, int(idx.size), 0
        e = np.concatenate(e)
        g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(idx.size, idx.size))
        n, lab = connected_components(g, directed=False)
        if n == 1:
            return cap, 1, 0
        keep = np.bincount(lab).argmax()
        out = np.zeros_like(cap)
        out[idx[lab == keep]] = True
        return out, int(n), int(cap.sum() - out.sum())

    @staticmethod
    def fill_holes(cap, faces):
        """Absorb pockets of non-cap vertices enclosed by the cap.

        Dropping islands is not enough on its own. A cap can be one connected
        patch and still have a hole punched through it where a few vertices fell
        below the probability threshold, and that hole contributes a SECOND
        boundary loop, which splits the rim just as an island does. The mesh is
        closed and a cap is a small patch, so the non-cap region is one huge
        component plus any such pockets: keep the huge one as non-cap and flip
        everything else into the cap.
        """
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        out = ~cap
        idx = np.flatnonzero(out)
        if idx.size == 0:
            return cap, 0
        pos = -np.ones(cap.shape[0], dtype=int); pos[idx] = np.arange(idx.size)
        e = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            m = out[faces[:, a]] & out[faces[:, b]]
            if m.any():
                e.append(np.stack([pos[faces[m, a]], pos[faces[m, b]]], 1))
        if not e:
            return cap, 0
        e = np.concatenate(e)
        g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(idx.size, idx.size))
        n, lab = connected_components(g, directed=False)
        if n == 1:
            return cap, 0
        main = np.bincount(lab).argmax()
        filled = cap.copy()
        filled[idx[lab != main]] = True
        return filled, int(filled.sum() - cap.sum())

    @staticmethod
    def _boundary_edges(cap, faces):
        """Edges of the kept mesh belonging to exactly one face.

        Vectorised. A collections.Counter over ~25k half-edges is pure Python
        and, run once per branch per sample, dominated the stage-2 step time."""
        kept = faces[~cap[faces].all(1)]
        if len(kept) == 0:
            return np.zeros((0, 2), dtype=int)
        e = np.sort(np.concatenate([kept[:, [0, 1]], kept[:, [1, 2]], kept[:, [2, 0]]]), axis=1)
        # Encode each edge as one int64 key. np.unique(..., axis=0) compares rows
        # and is an order of magnitude slower than the 1-D path.
        n = int(faces.max()) + 1
        key = e[:, 0].astype(np.int64) * n + e[:, 1]
        uk, cnt = np.unique(key, return_counts=True)
        b = uk[cnt == 1]
        return np.stack([b // n, b % n], axis=1).astype(int)

    @staticmethod
    def rim_topology(cap, faces):
        """True topology of the cut boundary: (n_loops, max_vertex_degree).

        A usable rim is ONE simple cycle: a single connected component in which
        every vertex has exactly two boundary neighbours. Walking the boundary
        and comparing lengths -- which is what this used to do -- cannot tell a
        genuinely split rim from a single loop that pinches against itself at
        one vertex, because a greedy walk takes a wrong turn at the pinch and
        reports two fragments. The case that sent me chasing a "split rim" was
        exactly that: one component, 44 vertices, one of degree 4.
        """
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        be = SensorUncapper._boundary_edges(cap, faces)
        if len(be) == 0:
            return 0, 0
        vs = np.unique(be)
        pos = {int(v): i for i, v in enumerate(vs)}
        r = np.fromiter((pos[int(x)] for x in be[:, 0]), int, len(be))
        c = np.fromiter((pos[int(x)] for x in be[:, 1]), int, len(be))
        g = coo_matrix((np.ones(len(r)), (r, c)), shape=(len(vs), len(vs)))
        n_loops, _ = connected_components(g, directed=False)
        deg = np.bincount(np.concatenate([r, c]), minlength=len(vs))
        return int(n_loops), int(deg.max())

    @staticmethod
    def rim_loop(cap, faces):
        """Rim as an ORDERED cycle of vertex indices, walking the boundary.

        The tube sweeper needs the loop in order -- it lofts each centerline
        ring onto these vertices in sequence -- so an unordered set is useless
        to it. Returns the LONGEST cycle when a cap's boundary splits into
        several, which happens if the predicted region is not simply connected.
        """
        bedges = SensorUncapper._boundary_edges(cap, faces)
        if len(bedges) == 0:
            return np.zeros(0, dtype=int)
        adj = collections.defaultdict(list)
        for a, b in bedges:
            adj[a].append(b); adj[b].append(a)
        seen, loops = set(), []
        for start in adj:
            if start in seen:
                continue
            loop, cur, prev = [start], start, None
            seen.add(start)
            while True:
                nxt = next((v for v in adj[cur] if v != prev and v not in seen), None)
                if nxt is None:
                    break
                loop.append(nxt); seen.add(nxt); prev, cur = cur, nxt
            loops.append(loop)
        return np.array(max(loops, key=len), dtype=int)

    @staticmethod
    def _rim(cap, faces):
        """Vertices on the boundary loop left by deleting the cap."""
        return np.unique(SensorUncapper._boundary_edges(cap, faces))

    @staticmethod
    def _plane_check(verts, rim, tangent, centre):
        """Plane fit to the rim, reported ONLY as a health signal.

        A flat rim whose normal disagrees with the sensor is worth looking at.
        A non-flat rim explains itself: the plane fit is meaningless there and
        the sensor is the only usable answer."""
        if len(rim) < 5:
            return float("nan"), float("nan")
        P = verts[rim]; cen = P.mean(0)
        _, s, vt = np.linalg.svd(P - cen)
        n = vt[-1]
        if n @ (cen - centre) < 0:
            n = -n
        flat = float(s[2] / (s[1] + 1e-9))
        d = float(np.degrees(np.arccos(np.clip(abs(n @ tangent), -1, 1))))
        return flat, d
