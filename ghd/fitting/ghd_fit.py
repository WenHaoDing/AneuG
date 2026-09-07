"""
ghd_fit.py -- Stage 2: GHD-coefficient fitting + residual pose refinement.

Takes Stage 1's alignment output (ghd/fitting/alignment.py's final_aligned.obj)
as a FIXED target and deforms the canonical mesh, via GHD coefficients `phi`,
to match it -- jointly with a fresh residual rigid pose (rotation, scale,
translation) on top. This mirrors the old pipeline's actual mechanics,
confirmed by directly reading almaha/Aneu_GHD/fitting.py::ghd_fit() and
scripts/fit_case_V2.py: Stage 1 (there, prep_case_for_ghd.py) only coarsely
pre-aligns the target, and Stage 2 (there, fit_case_V2.py) fits the
CANONICAL's own fresh pose jointly with the deformation on top of that --
"ring-chamfer and centroid losses in GHD handle residual alignment" (that
script's own docstring). We keep the joint pose+phi optimization but drop
every loss term that needs those pre-registered ring indices.

Unlike the old pipeline (rigid-only Stage 1, so Stage 2 had to discover scale
from scratch via an "undershoot" heuristic), this repo's Stage 1
(alignment.py) already fits scale via chamfer+volume gradient descent -- so
here the residual pose initializes at identity rotation/translation (w=0,
t=0), and log_s is initialized from a calculated volume-ratio estimate
(cube-root of target/canonical volume -- see the "fittable parameters"
comment below), not an undershoot guess and not read from a persisted
Stage 1 value.

Deliberately EXCLUDES every loss term that pins the mesh to pre-registered
canonical "opening" vertex indices (ring chamfer / centroid / planarity /
lateral / normal-alignment) -- the mesh is free to warp its own opening
shape. Volume control is kept (VolumeLoss) per explicit requirement.

EXPERIMENTAL opening-index-dependent terms (added for a specific requested
experiment, all default lambda=0 / off): opening-chamfer (new, pooled
chamfer between points sampled on target vs. canonical opening/cap faces),
ring-roundness (isoperimetric circularity of each canonical opening),
ring-normal-alignment (aka "ring tangent" -- matches each opening's plane
normal/outward direction to the target's), geodesic-correspondence (chamfer
augmented with per-opening graph-geodesic-distance channels). These reverse
the "no opening-index" default and require Stage 1 to have saved per-opening
ring indices (see alignment.py::save_stage1_checkpoints) -- only meant for
this experiment, not the recommended default configuration.

All loss modules are transferred, not reimplemented, from
almaha/Aneu_GHD/losses/ into ghd/fitting/losses/ (see that package).

phi -> mesh reuses this repo's own MultiCanonicalGHDReconstruct/
GHD_Reconstruct (models/multi_canonical_ghd_reconstruct.py,
models/ghd_reconstruct.py) -- the SAME basis/canonical mesh the rest of the
repo (VAE training, branch generation) already uses, so the phi fit here is
directly compatible, not a parallel/incompatible basis.

conda activate new
python ghd/fitting/ghd_fit.py --stage1-dir runtime/alignment_test/<case-name>
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import trimesh
from pytorch3d.ops import knn_points, sample_points_from_meshes
from pytorch3d.structures import Meshes
from pytorch3d.transforms import axis_angle_to_matrix

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.multi_canonical_ghd_reconstruct import MultiCanonicalGHDReconstruct
from ghd.fitting.losses.chamfer_loss import chamfer_loss
from ghd.fitting.losses.mesh_quality_loss import (
    laplacian_loss, normal_consistency_loss, EdgeLengthLoss,
)
from ghd.fitting.losses.rigid_loss import GHDRigidLoss
from ghd.fitting.losses.mesh_thickness_loss import MeshThickness
from ghd.fitting.losses.volume_loss import VolumeLoss, mesh_volume_watertight
from ghd.fitting.losses.dvs_loss import DVSOccupancyLoss
from ghd.fitting.losses.ring_roundness_loss import RingRoundnessLoss
from ghd.fitting.losses.ring_normal_alignment_loss import (
    RingNormalAlignmentLoss, outward_reference, ring_normal_and_pairing,
)
from ghd.fitting.losses.geodesic_correspondence_loss import (
    GeodesicChamferLoss, geodesic_from_openings, raw_geodesic_distances, geodesic_mask,
)


# ── target loading (normalized to the canonical's own scale, matching the
#    old pipeline's shared-divisor convention) ──────────────────────────────

def load_target(stage1_dir, norm_canonical, n_surface_samples=20000):
    """Loads Stage 1's final_aligned.obj + landmarks.npz. Points/volume are
    divided by `norm_canonical` -- the SAME divisor used for the canonical
    mesh -- so fitting happens in one shared normalized space, matching
    fit_case_V2.py's `s_can` convention (canonical and target normalized by
    canonical's own scale, not each their own)."""
    stage1_dir = Path(stage1_dir)
    mesh = trimesh.load(stage1_dir / "final_aligned.obj", process=False)
    lm = np.load(stage1_dir / "landmarks.npz", allow_pickle=True)

    pts, face_idx = trimesh.sample.sample_surface(mesh, n_surface_samples)
    normals = mesh.face_normals[face_idx]

    target_volume_phys = float(lm["volume"])
    n_openings = int(lm["n_openings"]) if "n_openings" in lm else 0
    opening_ring_idx = [lm[f"opening_ring_idx_{k}"] for k in range(n_openings)] if n_openings else []
    cap_face_mask = lm["cap_face_mask"] if "cap_face_mask" in lm else np.zeros(len(mesh.faces), dtype=bool)
    return {
        "aneurysm_type": int(lm["aneurysm_type"]),
        "points": (np.asarray(pts) / norm_canonical).astype(np.float32),
        "normals": np.asarray(normals, dtype=np.float32),
        "volume_norm": target_volume_phys / (norm_canonical ** 3),
        "mesh_verts_norm": (np.asarray(mesh.vertices) / norm_canonical).astype(np.float32),
        "mesh_faces": np.asarray(mesh.faces),
        "n_openings": n_openings,
        "opening_ring_idx": opening_ring_idx,  # EXPERIMENTAL, see module docstring
        "cap_face_mask": cap_face_mask,        # EXPERIMENTAL, see module docstring
    }


def prepare_dvs_samples(target, device, n_dvs=20_000, n_oversamp=200_000,
                        n_surf=200_000, d_min=0.0001, d_max=0.05):
    """Ported from almaha/Aneu_GHD/fitting.py::prepare_dvs_samples, operating
    directly on Stage 1's (already normalized) aligned target points instead
    of re-deriving watertightness -- the target here is real, open, clipped
    anatomy, so it's closed with trimesh.repair.fill_holes() first (same
    fallback VolumeLoss's own mesh_volume_watertight already relies on for
    its one-off reference volume). Returns None if closing fails, so callers
    can disable occupancy gracefully instead of crashing.

    n_dvs is the size of the MASTER pool retained per class (computed once,
    up front) -- DVSOccupancyLoss then randomly sub-samples a smaller batch
    (num_sample) from this pool every iteration, so n_dvs should be well
    above num_sample for that sub-sampling to mean anything (previously both
    defaulted to 2000, i.e. no real sub-sampling was happening -- every
    point was used every iteration)."""
    V, F = target["mesh_verts_norm"], target["mesh_faces"]
    tgt_trimesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
    if not tgt_trimesh.is_watertight:
        trimesh.repair.fill_holes(tgt_trimesh)
    if not tgt_trimesh.is_watertight:
        return None

    rng = np.random.default_rng(42)
    bbox_min = V.min(axis=0) - 0.05
    bbox_max = V.max(axis=0) + 0.05
    rand_pts = bbox_min + rng.random((n_oversamp, 3)) * (bbox_max - bbox_min)

    surf_pts, face_ids = trimesh.sample.sample_surface(tgt_trimesh, n_surf)
    face_normals = tgt_trimesh.face_normals[face_ids]
    offsets = rng.uniform(d_min, d_max, size=(len(surf_pts), 1))
    signs = rng.choice([-1.0, 1.0], size=(len(surf_pts), 1))
    surf_offset_pts = surf_pts + signs * offsets * face_normals
    all_pts = np.concatenate([rand_pts, surf_offset_pts], axis=0)

    inside_mask = tgt_trimesh.contains(all_pts)
    pos_pts, neg_pts = all_pts[inside_mask], all_pts[~inside_mask]
    if len(pos_pts) < n_dvs or len(neg_pts) < n_dvs:
        return None
    rng.shuffle(pos_pts)
    rng.shuffle(neg_pts)

    target_positives = torch.tensor(pos_pts[:n_dvs], dtype=torch.float32, device=device)
    target_negatives = torch.tensor(neg_pts[:n_dvs], dtype=torch.float32, device=device)
    with torch.no_grad():
        dist_p2n = knn_points(target_positives.view(1, -1, 3), target_negatives.view(1, -1, 3), K=1)[0].view(-1)
        dist_n2p = knn_points(target_negatives.view(1, -1, 3), target_positives.view(1, -1, 3), K=1)[0].view(-1)
        dist_mean = torch.cat([dist_p2n, dist_n2p]).mean()
        sigma = 0.1 * dist_mean
        w_p2n = torch.exp(-dist_p2n / (sigma + 1e-6)); w_p2n = w_p2n / w_p2n.mean()
        w_n2p = torch.exp(-dist_n2p / (sigma + 1e-6)); w_n2p = w_n2p / w_n2p.mean()
    return target_positives, target_negatives, w_p2n, w_n2p


def _linear_decay(start, end, frac_done, decay_frac):
    if decay_frac <= 0:
        return end
    t = min(max(frac_done / decay_frac, 0.0), 1.0)
    return start + (end - start) * t


def _cosine_decay(start, end, frac_done, decay_frac):
    """Cosine-annealed decay from `start` (t=0) to `end` (t=decay_frac),
    held flat at `end` after -- same envelope as torch's CosineAnnealingLR,
    used here for the rigid-loss weight schedule (per explicit request)."""
    if decay_frac <= 0:
        return end
    t = min(max(frac_done / decay_frac, 0.0), 1.0)
    return end + (start - end) * (1 + np.cos(np.pi * t)) / 2


def plot_loss_curves(history, save_path):
    """Grid of subplots, one per loss/diagnostic term in `history`, iteration
    on the x-axis -- saved once at the end of fitting so a run's convergence
    behavior can be eyeballed without re-loading loss_history.npz."""
    terms = [k for k in history.keys() if k != "lr"] + ["lr"]
    n = len(terms)
    n_cols = 4
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.8 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    for i, term in enumerate(terms):
        ax = axes[i]
        vals = history[term]
        ax.plot(vals, linewidth=1)
        ax.set_title(term, fontsize=10)
        ax.set_xlabel("iter", fontsize=8)
        ax.tick_params(labelsize=7)
        if term not in ("lr", "lambda_rigid_w", "volume_target_frac", "lambda_volume_w",
                        "lambda_consistency_w", "lambda_edge_w",
                        "rot_norm", "scale", "trans_norm") and len(vals) and max(vals) > 0:
            ax.set_yscale("log")
    for i in range(n, len(axes)):
        axes[i].axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)
    print(f"  Loss curves saved: {save_path}")


def render_fit_sanity(save_path, fitted_verts_phys, faces, target_verts_phys, n_angles=6):
    """One combined multi-angle grid image, fitted mesh (blue) vs. target
    mesh (grey wireframe-ish points) -- same visual pattern as
    alignment.py::render_sanity_images, but for a single posed mesh pair
    instead of point-cloud branch segments."""
    # DISPLAY merely being SET makes pv.system_supports_plotting() true, so over
    # an `ssh -X` forward with no GLX the old guard never fired and VTK called
    # abort() -- uncatchable. Always render offscreen on xvfb instead.
    os.environ.pop("DISPLAY", None)
    import pyvista as pv

    try:
        pv.start_xvfb()
    except Exception:
        pass

    fitted_pd = pv.PolyData(fitted_verts_phys, faces=np.concatenate(
        [np.full((faces.shape[0], 1), 3), faces], axis=1))
    all_pts = np.vstack([fitted_verts_phys, target_verts_phys])
    focal = all_pts.mean(axis=0)
    diag = np.linalg.norm(all_pts.max(0) - all_pts.min(0))
    cam_dist = diag * 1.6

    n_cols = min(3, n_angles)
    n_rows = int(np.ceil(n_angles / n_cols))
    plotter = pv.Plotter(off_screen=True, shape=(n_rows, n_cols),
                         window_size=(480 * n_cols, 480 * n_rows), border=True)

    for i in range(n_angles):
        row, col = divmod(i, n_cols)
        plotter.subplot(row, col)
        angle_deg = round(360 * i / n_angles)
        angle_rad = np.deg2rad(angle_deg)
        cam_pos = focal + cam_dist * np.array([np.cos(angle_rad), np.sin(angle_rad), 0.3])

        plotter.add_points(target_verts_phys, color="lightgreen", point_size=3, opacity=0.5, label="target")
        plotter.add_mesh(fitted_pd, color="royalblue", opacity=0.6, show_edges=True,
                         edge_color="navy", label="fitted" if i == 0 else None)

        plotter.add_text(f"{angle_deg} deg", font_size=10, position="upper_edge")
        if i == 0:
            plotter.add_legend(size=(0.35, 0.3), bcolor="white")
        plotter.set_background("white")
        plotter.camera.position = cam_pos
        plotter.camera.focal_point = focal
        plotter.camera.up = (0.0, 0.0, 1.0)

    plotter.screenshot(str(save_path))
    plotter.close()


# ── EXPERIMENTAL opening-index-dependent loss setup ─────────────────────────
# See module docstring. All of this is one-time setup (numpy/CPU), building
# fixed torch buffers reused every iteration -- canonical topology/rest-pose
# never changes during fitting, and the target is fixed throughout.

def _canonical_opening_data(mc, atype, canonical_root, device):
    """Per-opening canonical data: rank-ordered vertex indices (from
    canonical_topology.npy, already rank-matched to case branch order -- see
    alignment.py's module docstring), walk order, rest-pose plane
    (normal/centroid), outward-oriented pairing/sign for normal-alignment,
    and the pooled cap-face mask (mesh.obj faces not in mesh_trimmed.obj,
    i.e. the opening/cap regions -- same position-matching MultiCanonicalGHDReconstruct
    ._trimmed_faces already does for uncapping, just inverted)."""
    can_type_dir = Path(canonical_root) / ("Bifurcated" if atype == 0 else "Sidewall")
    topo = np.load(can_type_dir / "canonical_topology.npy", allow_pickle=True).item()
    recon = mc.get(atype)
    V0_np = recon.canonical_Meshes.verts_packed().detach().cpu().numpy()
    F_can_np = recon.canonical_Meshes.faces_packed().detach().cpu().numpy()
    can_vnormals = trimesh.Trimesh(vertices=V0_np, faces=F_can_np, process=False).vertex_normals

    opening_idx = [np.asarray(o["indices"], dtype=np.int64) for o in topo["openings"]]

    walk_order, planes, pairings, signs = [], [], [], []
    for idxs in opening_idx:
        # canonical_topology.npy's opening indices aren't a mathematically
        # exact single-edge boundary loop (a handful of vertices have 3
        # ring-neighbours instead of 2 -- extract_boundary_loop_order's
        # strict graph walk fails on them), so order cyclically by angle
        # around the ring's own best-fit plane instead -- robust to that
        # irregularity, and well-justified since these openings are
        # anatomically expected to be close to planar/circular anyway (the
        # same premise RingRoundnessLoss itself relies on).
        pts0 = V0_np[idxs]
        c = pts0.mean(axis=0)
        _, _, Vt = np.linalg.svd(pts0 - c, full_matrices=False)
        normal = Vt[-1]
        tmp = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) <= 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = np.cross(normal, tmp); e1 /= np.linalg.norm(e1)
        e2 = np.cross(normal, e1)
        angles = np.arctan2((pts0 - c) @ e2, (pts0 - c) @ e1)
        order = idxs[np.argsort(angles)]
        walk_order.append(order)
        pts = V0_np[order]
        planes.append((normal, c))
        normal_ref = outward_reference(pts, V0_np, can_vnormals)
        _, pairing, sign = ring_normal_and_pairing(pts, normal_ref=normal_ref)
        pairings.append(pairing)
        signs.append(sign)

    trimmed = mc._trimmed_faces(atype)
    trimmed_set = {tuple(sorted(f)) for f in trimmed}
    cap_face_mask = np.array([tuple(sorted(f)) not in trimmed_set for f in F_can_np])

    return {
        "V0_np": V0_np, "F_can_np": F_can_np,
        "opening_idx": opening_idx, "walk_order": walk_order,
        "planes": planes, "pairings": pairings, "signs": signs,
        "cap_face_mask": cap_face_mask,
    }


def setup_experimental_losses(mc, atype, canonical_root, target, device,
                              n_opening_chamfer_pts=3000, geodesic_weight=1.0, geodesic_mask_eps=0.05):
    """Builds the four EXPERIMENTAL opening-index-dependent loss modules
    (see module docstring), returning a dict with whichever ones the
    available data supports -- None for terms that can't be built (e.g. no
    opening rings saved by Stage 1, or fewer than 2 openings on one side).
    Returns {} entirely if target has no opening data at all."""
    result = {}
    if target["n_openings"] == 0:
        return result

    can = _canonical_opening_data(mc, atype, canonical_root, device)
    recon = mc.get(atype)
    F_can = recon.canonical_Meshes.faces_packed()

    n_shared = min(len(can["opening_idx"]), target["n_openings"])
    tgt_mesh_np = target["mesh_verts_norm"]
    tgt_faces_np = target["mesh_faces"]
    tgt_vnormals = trimesh.Trimesh(vertices=tgt_mesh_np, faces=tgt_faces_np, process=False).vertex_normals

    # -- ring roundness (canonical-only, its own rest-pose plane) --
    ring_specs_round = [(can["walk_order"][k], can["planes"][k][0], can["planes"][k][1])
                        for k in range(len(can["opening_idx"]))]
    result["roundness"] = RingRoundnessLoss(ring_specs_round).to(device)

    # -- ring normal-alignment ("ring tangent"): canonical pairing/sign vs. target normal --
    specs_normal = []
    for k in range(n_shared):
        tgt_idx = target["opening_ring_idx"][k]
        tgt_pts = tgt_mesh_np[tgt_idx]
        tgt_normal_ref = outward_reference(tgt_pts, tgt_mesh_np, tgt_vnormals)
        normal_tgt, _, _ = ring_normal_and_pairing(tgt_pts, normal_ref=tgt_normal_ref)
        specs_normal.append((can["opening_idx"][k], can["pairings"][k], can["signs"][k], normal_tgt))
    result["normal_alignment"] = RingNormalAlignmentLoss(specs_normal).to(device)

    # -- geodesic correspondence --
    raw_can = raw_geodesic_distances(can["V0_np"], can["F_can_np"], can["opening_idx"])
    geo_can = raw_can / raw_can.max(axis=0, keepdims=True)
    mask_can = geodesic_mask(raw_can, geodesic_mask_eps)
    tgt_opening_idx = target["opening_ring_idx"][:n_shared]
    raw_tgt = raw_geodesic_distances(tgt_mesh_np, tgt_faces_np, tgt_opening_idx)
    geo_tgt = raw_tgt / raw_tgt.max(axis=0, keepdims=True)
    mask_tgt = geodesic_mask(raw_tgt, geodesic_mask_eps)
    # geo_can has one channel per CANONICAL opening; only keep the channels
    # with a matching target opening (n_shared), consistent with geo_tgt.
    result["geodesic"] = GeodesicChamferLoss(
        geo_can[:, :n_shared], mask_can, tgt_mesh_np, geo_tgt, mask_tgt, geo_weight=geodesic_weight,
    ).to(device)

    # -- opening chamfer (new, pooled target-cap-points vs. live canonical-cap-points) --
    can_cap_face_idx = torch.as_tensor(np.where(can["cap_face_mask"])[0], dtype=torch.long, device=device)
    tgt_cap_mask = target["cap_face_mask"]
    if can_cap_face_idx.numel() > 0 and tgt_cap_mask.any():
        tgt_cap_mesh = trimesh.Trimesh(vertices=tgt_mesh_np, faces=tgt_faces_np[tgt_cap_mask], process=False)
        tgt_cap_pts, _ = trimesh.sample.sample_surface(tgt_cap_mesh, n_opening_chamfer_pts)
        result["opening_chamfer"] = {
            "can_cap_face_idx": can_cap_face_idx,
            "tgt_cap_pts": torch.as_tensor(np.asarray(tgt_cap_pts), dtype=torch.float32, device=device).unsqueeze(0),
        }

    return result


# ── fitting driver ───────────────────────────────────────────────────────────

def ghd_fit(stage1_dir, canonical_root, out_dir, aneurysm_type=None, case_dir=None,
           n_iter=10000, lr=1e-3, eta_min=1e-4, device="cuda",
           lambda_chamfer_n1=0.8, lambda_laplacian=1e-2,
           lambda_rigid_start=2.0, lambda_rigid_end=0.001, rigid_decay_frac=0.80,
           rigid_warmup_iters=0, rigid_warmup_start=5.0,
           lambda_volume=1.0, volume_ceiling_ratio=1.2,
           volume_target_frac_start=1.0, volume_target_frac_end=1.0,
           volume_target_decay_frac=0.5, volume_ramp_frac=0.5,
           lambda_thickness=2.0, thickness_r=0.2,
           lambda_consistency_start=0.3, lambda_consistency_end=0.3, consistency_decay_frac=0.80,
           lambda_edge_start=0.1, lambda_edge_end=0.1, edge_decay_frac=0.80,
           lambda_occupancy=1.0, dvs_surf_d_min=0.0001, dvs_surf_d_max=0.05,
           lambda_opening_chamfer=0.0, n_opening_chamfer_pts=3000,
           lambda_roundness=0.0, lambda_normal_alignment=0.0,
           lambda_geodesic=0.0, geodesic_weight=1.0, geodesic_mask_eps=0.05,
           n_surface_samples=20000, n_rigid_checkpoints=10,
           extra_rigid_checkpoint_values=(0.1, 0.05, 0.01),
           rigid_pose_freeze_frac=0.0,
           log_every=200):
    """
    Volume control, relaxed per explicit request: the LIVE target is now a
    constant 100% of true volume (volume_target_frac_start/end both default
    1.0 -- no more deliberate 75% undershoot; the mesh is allowed to expand
    toward the real size without an artificial ceiling pulling it down
    early). Instead, the WEIGHT is what's scheduled now: lambda_volume is
    the FINAL weight, reached via a cosine RAMP-UP (reusing _cosine_decay
    with start=0, end=lambda_volume -- mathematically identical shape to the
    rigid schedule, just increasing instead of decreasing) over the first
    volume_ramp_frac (default 0.5) of iterations, so volume is completely
    unpenalized at the very start (free exploration of pose/shape) and only
    progressively enforced as training proceeds. volume_target_frac_*/
    volume_target_decay_frac are kept (default to a no-op constant 1.0) in
    case a target-fraction schedule is ever wanted again independently of
    the weight ramp. volume_ceiling_ratio (default 1.2) is the separate,
    constant-throughout-training hard ceiling on live/target volume ratio
    (see VolumeLoss's own docstring) -- independent of both schedules.

    lambda_consistency_{start,end} / lambda_edge_{start,end}: same cosine-
    decay pattern as the rigid weight (start -> end over
    consistency_decay_frac / edge_decay_frac of iterations, then held
    flat). Default to start==end (i.e. a no-op constant weight, matching
    the old single-value behavior) -- pass a lower/zero *_end to relax
    these regularizers late in training, e.g. for an ablation testing
    whether they're over-constraining the fit once the shape has mostly
    converged.

    rigid_pose_freeze_frac: the residual pose (w/log_s/t, fit fresh here,
    jointly with phi -- see module docstring) has NO rigidity penalty of its
    own, unlike phi's deformation (lambda_rigid). Early in training, when
    lambda_rigid is high and bending phi is expensive, the optimizer's
    cheapest way to reduce chamfer can be to just rotate the whole mesh via
    w instead of actually deforming -- which can undo a correct branch
    correspondence Stage 1's alignment specifically established, before phi
    ever gets a chance to use it. Setting this > 0 (a fraction of n_iter,
    e.g. 0.5) freezes w (rotation only -- log_s/t stay free) via
    requires_grad_(False) for that leading fraction, forcing the optimizer
    to make progress by deforming phi instead; w unfreezes automatically
    afterward and both are optimized jointly as before. 0.0 (default) = no
    freeze, unchanged behavior.

    n_rigid_checkpoints: checkpoints (+ sanity images) are snapshotted at
    n_rigid_checkpoints evenly-spaced (in weight-value) points along the
    rigid-loss WEIGHT's own decay schedule (lambda_rigid_start ->
    lambda_rigid_end, cosine-annealed over rigid_decay_frac of iterations,
    then held flat) -- NOT the raw rigid-loss VALUE, which
    is noisy/non-monotonic and can cross several value-thresholds within a
    handful of iterations (tried first, empirically produced clusters of
    near-duplicate checkpoints seconds apart -- unusable for monitoring).
    The WEIGHT schedule is deterministic and monotonic, so this instead
    gives evenly-spaced snapshots across the whole decay, each labeled by
    the weight value active when it was taken -- letting you pick an
    earlier (higher-rigid-weight = less warped) fallback if the final
    (lowest-weight, best-chamfer) result is too warped.

    extra_rigid_checkpoint_values: guaranteed additional thresholds, merged
    with the n_rigid_checkpoints evenly-spaced ones -- default (0.1, 0.05,
    0.01) exists because the evenly-LINEAR-spaced default list (2.0 down to
    0.001) barely samples that low-weight tail (only its very last point is
    near there), even though that's exactly where the mesh is most free to
    warp and most worth inspecting closely.
    """
    device = torch.device(device)
    stage1_dir, out_dir = Path(stage1_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "rigid_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sanity_dir = out_dir / "sanity"
    sanity_dir.mkdir(parents=True, exist_ok=True)

    mc = MultiCanonicalGHDReconstruct(canonical_root, device=device)

    lm = np.load(stage1_dir / "landmarks.npz", allow_pickle=True)
    atype = int(lm["aneurysm_type"]) if aneurysm_type is None else aneurysm_type
    recon = mc.get(atype)
    norm_canonical = recon.norm_canonical
    V0 = recon.canonical_Meshes.verts_packed()          # [N, 3] normalized-space canonical rest verts
    F_can = recon.canonical_Meshes.faces_packed()        # [F, 3]
    U = recon.GHD_eigvec                                  # [N, num_coeffs]
    num_coeffs = U.shape[1]

    target = load_target(stage1_dir, norm_canonical, n_surface_samples=n_surface_samples)
    tgt_pts = torch.as_tensor(target["points"], device=device).unsqueeze(0)
    tgt_nrm = torch.as_tensor(target["normals"], device=device).unsqueeze(0)
    target_volume = target["volume_norm"]
    target_verts_phys = target["mesh_verts_norm"] * norm_canonical  # for sanity renders

    dvs_samples = None
    if lambda_occupancy > 0:
        dvs_samples = prepare_dvs_samples(target, device, d_min=dvs_surf_d_min, d_max=dvs_surf_d_max)
        if dvs_samples is None:
            print("  WARNING: could not close target mesh for DVS occupancy sampling "
                  "-- disabling occupancy loss (lambda_occupancy forced to 0).")
            lambda_occupancy = 0.0
        else:
            # FIXED subset of 4000/class, chosen once here and reused every
            # iteration (not resampled) -- per explicit request: bigger
            # pools didn't cost meaningful GPU time/memory, but a
            # per-iteration-reshuffled target set isn't wanted either.
            # prepare_dvs_samples already shuffles before returning, so a
            # plain prefix slice is already a random (but now fixed) subset.
            n_occ = 4000
            dvs_samples = tuple(t[:n_occ] for t in dvs_samples)

    experimental = setup_experimental_losses(
        mc, atype, canonical_root, target, device,
        n_opening_chamfer_pts=n_opening_chamfer_pts,
        geodesic_weight=geodesic_weight, geodesic_mask_eps=geodesic_mask_eps,
    ) if (lambda_opening_chamfer > 0 or lambda_roundness > 0
          or lambda_normal_alignment > 0 or lambda_geodesic > 0) else {}
    if lambda_opening_chamfer > 0 and "opening_chamfer" not in experimental:
        print("  WARNING: no usable opening/cap data for opening-chamfer -- disabling (lambda forced to 0).")
        lambda_opening_chamfer = 0.0
    if lambda_roundness > 0 and "roundness" not in experimental:
        lambda_roundness = 0.0
    if lambda_normal_alignment > 0 and "normal_alignment" not in experimental:
        lambda_normal_alignment = 0.0
    if lambda_geodesic > 0 and "geodesic" not in experimental:
        lambda_geodesic = 0.0

    # ── fittable parameters: phi (shape) + fresh residual pose (rotation,
    #    scale, translation), jointly optimized -- see module docstring.
    # log_s's init is CALCULATED directly from geometry -- not read from a
    # persisted Stage 1 value (Stage 1's landmarks.npz deliberately doesn't
    # save one, matching the old prep_case_for_ghd.py format). The old
    # pipeline's own heuristic (max-vertex-norm ratio, "_size_match") was
    # tried here first and empirically made the initial fit WORSE (it's a
    # noisy proxy -- a single extremal point, not representative of overall
    # size), whereas volume already matches almost exactly at scale=1 (Stage
    # 1 already fits scale via chamfer+volume). So instead: cube-root of the
    # volume ratio, V(s*X) = s^3 * V(X) -- the same identity alignment.py's
    # own volume loss uses -- which is both a better-behaved estimator and
    # directly tied to VolumeLoss, the term explicitly required to be kept.
    V0_faces_np = F_can.detach().cpu().numpy()
    V0_volume = float(trimesh.Trimesh(vertices=V0.detach().cpu().numpy(), faces=V0_faces_np, process=False).volume)
    size_match = (target_volume / max(V0_volume, 1e-12)) ** (1.0 / 3.0)
    phi = nn.Parameter(torch.zeros(num_coeffs, 3, device=device))
    w = nn.Parameter(torch.zeros(3, device=device))
    log_s = nn.Parameter(torch.tensor(float(np.log(max(size_match, 1e-6))), device=device))
    t = nn.Parameter(torch.zeros(3, device=device))

    optimizer = torch.optim.Adam([phi, w, log_s, t], lr=lr)
    # The warm-up is a PRELUDE, not a slice taken out of n_iter: n_iter always
    # means the length of the normal fitting process, so a 1000-iteration
    # warm-up plus n_iter=10000 runs 11000 in total. Every other schedule
    # (lr, volume, consistency, edge) is timed over the n_iter part alone and
    # simply sits at its starting value while the warm-up runs.
    total_iters = n_iter + max(rigid_warmup_iters, 0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iters,
                                                           eta_min=eta_min)

    edge_loss_fn = EdgeLengthLoss(recon.canonical_Meshes)
    rigid_loss_fn = GHDRigidLoss(V0, F_can)
    thickness_fn = MeshThickness(r=thickness_r)
    volume_loss_fn = VolumeLoss(target_volume=target_volume, ceiling_ratio=volume_ceiling_ratio)
    occupancy_fn = DVSOccupancyLoss(num_sample=4000) if lambda_occupancy > 0 else None

    def render():
        offset = torch.einsum('nm,mc->nc', U, phi)
        V_deformed = V0 + offset
        R = axis_angle_to_matrix(w.unsqueeze(0)).squeeze(0)
        V_rendered = (V_deformed @ R.T) * torch.exp(log_s) + t
        return Meshes(verts=[V_rendered], faces=[F_can])

    def save_rigid_checkpoint(mesh_, iteration, rigid_weight, rigid_value):
        """Snapshot phi/w/log_s/t + fitted mesh + a target-overlay sanity
        image, tagged by the rigid-loss WEIGHT active at the time (the
        deterministic decay schedule, not the noisy raw loss value) -- see
        n_rigid_checkpoints docstring above."""
        tag = f"w{rigid_weight:.3f}"
        with torch.no_grad():
            verts_phys = (mesh_.verts_packed() * norm_canonical).detach().cpu().numpy()
        faces_np = F_can.detach().cpu().numpy()
        trimesh.Trimesh(vertices=verts_phys, faces=faces_np, process=False).export(
            ckpt_dir / f"ghd_fitted_{tag}.obj")
        np.savez(ckpt_dir / f"ghd_coefficients_{tag}.npz",
                 phi=phi.detach().cpu().numpy(), w_rot=w.detach().cpu().numpy(),
                 log_scale=np.array(log_s.detach().cpu().item()), t_vec=t.detach().cpu().numpy(),
                 iteration=np.array(iteration), lambda_rigid_w=np.array(rigid_weight),
                 rigid_loss=np.array(rigid_value))
        render_fit_sanity(sanity_dir / f"sanity_{tag}.png", verts_phys, faces_np, target_verts_phys)
        print(f"  rigid-checkpoint saved: lambda_rigid_w<={rigid_weight:.3f} @ iter {iteration} "
              f"(rigid_loss={rigid_value:.4f}) -> {tag}", flush=True)

    # Evenly spaced along the WEIGHT's own decay (descending, start -> end),
    # plus any extra guaranteed values (see extra_rigid_checkpoint_values
    # docstring) that fall within [end, start] -- merged, deduplicated,
    # sorted descending so the trigger loop below still sees a strictly
    # decreasing sequence.
    auto_thresholds = np.linspace(lambda_rigid_start, lambda_rigid_end, n_rigid_checkpoints)
    extra_thresholds = [v for v in extra_rigid_checkpoint_values
                        if min(lambda_rigid_start, lambda_rigid_end) <= v <= max(lambda_rigid_start, lambda_rigid_end)]
    # warm-up levels get their own checkpoints: the whole point of passing
    # through 5/4/3 is being able to look at the mesh at each.
    warmup_thresholds = ([v for v in (5.0, 4.0, 3.0)
                          if rigid_warmup_iters > 0
                          and lambda_rigid_start <= v <= rigid_warmup_start])
    rigid_weight_thresholds = sorted(set(np.round(np.concatenate(
        [auto_thresholds, extra_thresholds, warmup_thresholds]), 6)), reverse=True)
    next_ckpt_idx = 0

    history = {k: [] for k in [
        "chamfer", "chamfer_n1", "laplacian", "consistency", "edge", "rigid",
        "volume", "thickness", "occupancy", "opening_chamfer", "roundness",
        "normal_alignment", "geodesic", "total", "lambda_rigid_w",
        "volume_target_frac", "lambda_volume_w", "lambda_consistency_w", "lambda_edge_w",
        "lr", "rot_norm", "scale", "trans_norm",
    ]}
    best_chamfer = float("inf")
    best_state = None

    for it in range(total_iters):
        # 0 throughout the warm-up, then 0->1 across the n_iter normal phase
        frac = max(0, it - rigid_warmup_iters) / max(n_iter - 1, 1)
        if rigid_pose_freeze_frac > 0:
            w.requires_grad_(frac >= rigid_pose_freeze_frac)
        if rigid_warmup_iters > 0 and it < rigid_warmup_iters:
            # High-rigid warm-up: hold the mesh close to the canonical while the
            # pose and the low-order shape settle, then hand over to the normal
            # schedule at exactly lambda_rigid_start so the two join without a
            # step. Linear (not cosine) so the weight spends even time at each
            # level -- the point is to pass through 5/4/3 and checkpoint there.
            wf = it / max(rigid_warmup_iters, 1)
            lam_rigid = rigid_warmup_start + (lambda_rigid_start - rigid_warmup_start) * wf
        else:
            # own timebase -- `frac` still drives the volume/consistency/edge
            # schedules and must keep counting from iteration 0, or adding a
            # warm-up would silently retime every other decay too.
            lam_rigid = _cosine_decay(lambda_rigid_start, lambda_rigid_end,
                                      frac, rigid_decay_frac)
        vol_frac = _linear_decay(volume_target_frac_start, volume_target_frac_end, frac, volume_target_decay_frac)
        lam_volume = _cosine_decay(0.0, lambda_volume, frac, volume_ramp_frac)
        lam_consistency = _cosine_decay(lambda_consistency_start, lambda_consistency_end, frac, consistency_decay_frac)
        lam_edge = _cosine_decay(lambda_edge_start, lambda_edge_end, frac, edge_decay_frac)

        mesh = render()
        src_pts, src_nrm = sample_points_from_meshes(mesh, n_surface_samples, return_normals=True)

        loss_chamfer, loss_chamfer_n1 = chamfer_loss(src_pts, tgt_pts, src_nrm, tgt_nrm)
        loss_laplacian = laplacian_loss(mesh)
        loss_consistency = normal_consistency_loss(mesh)
        loss_edge = edge_loss_fn(mesh)
        loss_rigid = rigid_loss_fn(mesh.verts_packed())
        loss_volume = volume_loss_fn(mesh, target_frac=vol_frac)

        # Checkpoint by rigid-loss WEIGHT (the deterministic decay schedule),
        # not iteration count or the noisy raw loss value -- snapshots the
        # state active BEFORE this iteration's gradient step.
        while (next_ckpt_idx < len(rigid_weight_thresholds)
               and lam_rigid <= rigid_weight_thresholds[next_ckpt_idx]):
            save_rigid_checkpoint(mesh, it, rigid_weight_thresholds[next_ckpt_idx], loss_rigid.item())
            next_ckpt_idx += 1

        # MeshThickness.forward returns (dist, dist_v_norm, closed_indx, sign);
        # old pipeline names these (thickness_dict, thickness, _, sign) --
        # mask/first-relu-term use dist_v_norm ("thickness"), second-relu-term
        # uses dist ("thickness_dict"). Keep that exact mapping.
        dist, dist_v_norm, _, sign = thickness_fn(mesh)
        mask = (dist_v_norm.abs() > 0.1).logical_not().float()
        signed = torch.sign(sign)
        loss_thickness = (torch.relu(0.04 - dist_v_norm * signed) + torch.relu(0.01 - dist * signed)) * mask
        loss_thickness = loss_thickness.mean() + (1e-4 / (sign ** 2 + 1e-6) * mask).mean()

        loss_occupancy = torch.zeros((), device=device)
        if occupancy_fn is not None:
            pos, neg, wp2n, wn2p = dvs_samples
            loss_occupancy = occupancy_fn(mesh, pos, neg, wp2n, wn2p)

        verts_live = mesh.verts_packed()
        loss_opening_chamfer = torch.zeros((), device=device)
        if lambda_opening_chamfer > 0:
            oc = experimental["opening_chamfer"]
            can_cap_mesh = Meshes(verts=[verts_live], faces=[F_can[oc["can_cap_face_idx"]]])
            can_cap_pts = sample_points_from_meshes(can_cap_mesh, n_opening_chamfer_pts)
            loss_opening_chamfer, _ = chamfer_loss(can_cap_pts, oc["tgt_cap_pts"])

        loss_roundness = torch.zeros((), device=device)
        if lambda_roundness > 0:
            loss_roundness = experimental["roundness"](verts_live)

        loss_normal_alignment = torch.zeros((), device=device)
        if lambda_normal_alignment > 0:
            loss_normal_alignment = experimental["normal_alignment"](verts_live)

        loss_geodesic = torch.zeros((), device=device)
        if lambda_geodesic > 0:
            loss_geodesic = experimental["geodesic"](verts_live)

        total = (loss_chamfer + lambda_chamfer_n1 * loss_chamfer_n1
                + lambda_laplacian * loss_laplacian
                + lam_consistency * loss_consistency
                + lam_edge * loss_edge
                + lam_rigid * loss_rigid
                + lam_volume * loss_volume
                + lambda_thickness * loss_thickness
                + lambda_occupancy * loss_occupancy
                + lambda_opening_chamfer * loss_opening_chamfer
                + lambda_roundness * loss_roundness
                + lambda_normal_alignment * loss_normal_alignment
                + lambda_geodesic * loss_geodesic)

        optimizer.zero_grad()
        total.backward()
        optimizer.step()
        scheduler.step()

        cur_chamfer = loss_chamfer.item()
        if cur_chamfer < best_chamfer:
            best_chamfer = cur_chamfer
            best_state = {
                "phi": phi.detach().clone(), "w": w.detach().clone(),
                "log_s": log_s.detach().clone(), "t": t.detach().clone(),
                "iter": it,
            }

        history["chamfer"].append(cur_chamfer)
        history["chamfer_n1"].append(loss_chamfer_n1.item())
        history["laplacian"].append(loss_laplacian.item())
        history["consistency"].append(loss_consistency.item())
        history["edge"].append(loss_edge.item())
        history["rigid"].append(loss_rigid.item())
        history["volume"].append(loss_volume.item())
        history["thickness"].append(loss_thickness.item())
        history["occupancy"].append(loss_occupancy.item())
        history["opening_chamfer"].append(loss_opening_chamfer.item())
        history["roundness"].append(loss_roundness.item())
        history["normal_alignment"].append(loss_normal_alignment.item())
        history["geodesic"].append(loss_geodesic.item())
        history["total"].append(total.item())
        history["lambda_rigid_w"].append(lam_rigid)
        history["volume_target_frac"].append(vol_frac)
        history["lambda_volume_w"].append(lam_volume)
        history["lambda_consistency_w"].append(lam_consistency)
        history["lambda_edge_w"].append(lam_edge)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        history["rot_norm"].append(torch.norm(w).item())
        history["scale"].append(torch.exp(log_s).item())
        history["trans_norm"].append(torch.norm(t).item())

        if it % log_every == 0 or it == n_iter - 1:
            print(f"  [{it:6d}/{total_iters}] chamfer={cur_chamfer:.5f} rigid={loss_rigid.item():.4f} "
                  f"volume={loss_volume.item():.4f} (w={lam_volume:.3f}) thickness={loss_thickness.item():.4f} "
                  f"edge={loss_edge.item():.4f} (w={lam_edge:.3f}) consistency={loss_consistency.item():.4f} (w={lam_consistency:.3f}) "
                  f"occ={loss_occupancy.item():.4f} opench={loss_opening_chamfer.item():.4f} "
                  f"round={loss_roundness.item():.4f} normal={loss_normal_alignment.item():.4f} "
                  f"geo={loss_geodesic.item():.4f} total={total.item():.4f}", flush=True)

    phi.data.copy_(best_state["phi"]); w.data.copy_(best_state["w"])
    log_s.data.copy_(best_state["log_s"]); t.data.copy_(best_state["t"])
    with torch.no_grad():
        final_mesh = render()
        final_pts = sample_points_from_meshes(final_mesh, n_surface_samples)
        chamfer_final, _ = chamfer_loss(final_pts, tgt_pts)

    verts_phys = (final_mesh.verts_packed() * norm_canonical).detach().cpu().numpy()
    faces_np = F_can.detach().cpu().numpy()
    trimesh.Trimesh(vertices=verts_phys, faces=faces_np, process=False).export(out_dir / "ghd_fitted.obj")
    render_fit_sanity(sanity_dir / "sanity_final.png", verts_phys, faces_np, target_verts_phys)

    np.savez(out_dir / "ghd_coefficients.npz",
             phi=best_state["phi"].cpu().numpy(),
             w_rot=best_state["w"].cpu().numpy(),
             log_scale=np.array(best_state["log_s"].cpu().item()),
             t_vec=best_state["t"].cpu().numpy())

    metrics = {
        "chamfer_final": float(chamfer_final.item()),
        "chamfer_best": best_chamfer,
        "best_iter": best_state["iter"],
        "aneurysm_type": atype,
        # canonical-only normalization divisor (mm) -- same formula/role as the
        # old pipeline's "s_can" (fit_case_V2.py), constant per canonical type,
        # NOT the same thing as ghd_coefficients.npz's per-case "log_scale".
        # Kept here (not in ghd_coefficients.npz, matching the old pipeline's
        # own split) so phi/w_rot/log_scale/t_vec can be converted back to
        # physical units without re-deriving it from the canonical mesh.
        "s_can": float(norm_canonical),
        # source geometry that produced this fit -- results are filed by config
        # (runtime/<config>/<dataset>/<case>/) and the geometry lives on a
        # different disk, so without this the output can't say what it came from.
        "case_dir": str(case_dir) if case_dir else None,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    np.savez(out_dir / "loss_history.npz", **{k: np.asarray(v) for k, v in history.items()})
    plot_loss_curves(history, out_dir / "loss_curves.png")

    print(f"Stage 2 done. chamfer_best={best_chamfer:.5f} @ iter {best_state['iter']} "
          f"-> {out_dir}")
    return metrics


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage1-dir", required=True, help="Dir containing Stage 1's final_aligned.obj + landmarks.npz")
    parser.add_argument("--canonical-root", default=str(ROOT / "dataset" / "canonical"))
    parser.add_argument("--out", default=None, help="Default: same as --stage1-dir")
    parser.add_argument("--aneurysm-type", type=int, default=None, help="Override auto-detected type (0=bifurcated,1/2=sidewall)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rigid-warmup-iters", type=int, default=0,
                        help="Run this many iterations of HIGH rigid weight first, decreasing "
                             "linearly from --rigid-warmup-start down to --lambda-rigid-start, "
                             "before the normal schedule begins. 0 = off (default).")
    parser.add_argument("--rigid-warmup-start", type=float, default=5.0,
                        help="Rigid weight at the very first iteration of the warm-up.")
    parser.add_argument("--n-iter", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eta-min", type=float, default=1e-4)
    parser.add_argument("--lambda-chamfer-n1", type=float, default=0.8)
    parser.add_argument("--lambda-laplacian", type=float, default=1e-2)
    parser.add_argument("--lambda-rigid-start", type=float, default=2.0)
    parser.add_argument("--lambda-rigid-end", type=float, default=0.001)
    parser.add_argument("--rigid-decay-frac", type=float, default=0.80)
    parser.add_argument("--lambda-volume", type=float, default=1.0,
                         help="FINAL weight, reached via cosine ramp-up over --volume-ramp-frac of "
                              "iterations (0 at start -- unpenalized early free exploration).")
    parser.add_argument("--volume-ramp-frac", type=float, default=0.5,
                         help="Fraction of iterations over which the volume loss weight ramps 0 -> "
                              "--lambda-volume (cosine), then holds at full weight.")
    parser.add_argument("--volume-ceiling-ratio", type=float, default=1.2,
                         help="Hard ceiling on live/target volume ratio, constant throughout training -- "
                              "loss ramps up sharply past this fraction (see VolumeLoss).")
    parser.add_argument("--volume-target-frac-start", type=float, default=1.0,
                         help="Live volume target as a fraction of true target volume (default 1.0 -- "
                              "no artificial undershoot; the mesh may expand toward the real size).")
    parser.add_argument("--volume-target-frac-end", type=float, default=1.0)
    parser.add_argument("--volume-target-decay-frac", type=float, default=0.5)
    parser.add_argument("--lambda-thickness", type=float, default=2.0)
    parser.add_argument("--thickness-r", type=float, default=0.2)
    parser.add_argument("--lambda-consistency-start", type=float, default=0.3)
    parser.add_argument("--lambda-consistency-end", type=float, default=0.3,
                         help="Default equals start (no decay). Cosine-annealed like the rigid weight.")
    parser.add_argument("--consistency-decay-frac", type=float, default=0.80)
    parser.add_argument("--lambda-edge-start", type=float, default=0.1)
    parser.add_argument("--lambda-edge-end", type=float, default=0.1,
                         help="Default equals start (no decay). Cosine-annealed like the rigid weight.")
    parser.add_argument("--edge-decay-frac", type=float, default=0.80)
    parser.add_argument("--lambda-occupancy", type=float, default=1.0, help="On by default -- auto-disabled with a warning if the target mesh can't be closed for sampling; see prepare_dvs_samples. Was 2.0 (the old reference pipeline's weight) until it was lowered to 1.0 as the better default.")
    parser.add_argument("--dvs-surf-d-min", type=float, default=0.0001)
    parser.add_argument("--dvs-surf-d-max", type=float, default=0.05)
    parser.add_argument("--lambda-opening-chamfer", type=float, default=0.0,
                         help="EXPERIMENTAL, opening-index-dependent (0=disabled default). Pooled chamfer "
                              "between points sampled on target vs. canonical opening/cap faces.")
    parser.add_argument("--n-opening-chamfer-pts", type=int, default=3000)
    parser.add_argument("--lambda-roundness", type=float, default=0.0,
                         help="EXPERIMENTAL, opening-index-dependent (0=disabled default). Isoperimetric "
                              "roundness of each canonical opening.")
    parser.add_argument("--lambda-normal-alignment", type=float, default=0.0,
                         help="EXPERIMENTAL, opening-index-dependent (0=disabled default), aka 'ring tangent' "
                              "-- matches each opening's plane-normal direction to the target's.")
    parser.add_argument("--lambda-geodesic", type=float, default=0.0,
                         help="EXPERIMENTAL, opening-index-dependent (0=disabled default). Chamfer augmented "
                              "with per-opening graph-geodesic-distance channels.")
    parser.add_argument("--geodesic-weight", type=float, default=1.0)
    parser.add_argument("--geodesic-mask-eps", type=float, default=0.05)
    parser.add_argument("--n-surface-samples", type=int, default=20000)
    parser.add_argument("--n-rigid-checkpoints", type=int, default=10,
                         help="Snapshot checkpoints/sanity images at this many evenly-spaced points along "
                              "the rigid-loss WEIGHT's own decay (checkpoints -> rigid_checkpoints/, "
                              "images -> sanity/, both in --out), instead of a fixed iteration cadence.")
    parser.add_argument("--extra-rigid-checkpoint-values", type=str, default="0.1,0.05,0.01",
                         help="Comma-separated additional guaranteed rigid-weight checkpoint values, merged "
                              "with the evenly-spaced --n-rigid-checkpoints ones (values outside "
                              "[lambda-rigid-end, lambda-rigid-start] are ignored).")
    parser.add_argument("--rigid-pose-freeze-frac", type=float, default=0.0,
                         help="Freeze the residual pose's ROTATION (w only, not log_s/t) via "
                              "requires_grad_(False) for this leading fraction of n_iter (e.g. "
                              "0.5), so the optimizer can't cheaply rotate away a correct branch "
                              "correspondence instead of deforming phi while lambda_rigid is "
                              "still high. 0.0 (default) = no freeze.")
    parser.add_argument("--log-every", type=int, default=200)
    args = parser.parse_args()

    out_dir = args.out or args.stage1_dir
    extra_rigid_checkpoint_values = tuple(float(v) for v in args.extra_rigid_checkpoint_values.split(",") if v.strip())
    ghd_fit(
        stage1_dir=args.stage1_dir, canonical_root=args.canonical_root, out_dir=out_dir,
        aneurysm_type=args.aneurysm_type, n_iter=args.n_iter, lr=args.lr, eta_min=args.eta_min,
        device=args.device, lambda_chamfer_n1=args.lambda_chamfer_n1, lambda_laplacian=args.lambda_laplacian,
        lambda_rigid_start=args.lambda_rigid_start, lambda_rigid_end=args.lambda_rigid_end,
        rigid_warmup_iters=args.rigid_warmup_iters,
        rigid_warmup_start=args.rigid_warmup_start,
        rigid_decay_frac=args.rigid_decay_frac, lambda_volume=args.lambda_volume,
        volume_ceiling_ratio=args.volume_ceiling_ratio,
        volume_target_frac_start=args.volume_target_frac_start, volume_target_frac_end=args.volume_target_frac_end,
        volume_target_decay_frac=args.volume_target_decay_frac, volume_ramp_frac=args.volume_ramp_frac,
        lambda_thickness=args.lambda_thickness, thickness_r=args.thickness_r,
        lambda_consistency_start=args.lambda_consistency_start, lambda_consistency_end=args.lambda_consistency_end,
        consistency_decay_frac=args.consistency_decay_frac,
        lambda_edge_start=args.lambda_edge_start, lambda_edge_end=args.lambda_edge_end,
        edge_decay_frac=args.edge_decay_frac,
        lambda_occupancy=args.lambda_occupancy, dvs_surf_d_min=args.dvs_surf_d_min, dvs_surf_d_max=args.dvs_surf_d_max,
        lambda_opening_chamfer=args.lambda_opening_chamfer, n_opening_chamfer_pts=args.n_opening_chamfer_pts,
        lambda_roundness=args.lambda_roundness, lambda_normal_alignment=args.lambda_normal_alignment,
        lambda_geodesic=args.lambda_geodesic, geodesic_weight=args.geodesic_weight, geodesic_mask_eps=args.geodesic_mask_eps,
        n_surface_samples=args.n_surface_samples, n_rigid_checkpoints=args.n_rigid_checkpoints,
        extra_rigid_checkpoint_values=extra_rigid_checkpoint_values,
        rigid_pose_freeze_frac=args.rigid_pose_freeze_frac,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
