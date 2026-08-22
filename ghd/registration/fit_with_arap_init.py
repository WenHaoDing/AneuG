"""fit_with_arap_init.py -- standalone experiment: identical two-stage
gradient-based fitting to fit_with_tps_init.py, but the Stage A warm-start
target is built with ARAP (arap_register.py) instead of TPS.

WHY: TPS (a global affine + unbounded-RBF interpolant) forces the whole
canonical mesh -- including the unconstrained dome -- to inflate in order
to reach stretched branch landmarks (measured ~1.4x volume ratio on the
test case). ARAP with per-branch RIGID RING handles (see arap_register.py
-- each branch's opening ring moves as one rigid body: minimal/no-twist
rotation aligning canonical->target tangent, translate centroid to target
tip) came out at ~1.04x volume ratio on the same case, with a clean,
watertight, crease-free mesh -- no dome anchor needed at all.

  Stage A (new): starting from the TRUE native rest pose (phi=0, w=0, t=0),
    fit phi + residual pose toward the ARAP-warped canonical mesh via a
    DIRECT per-vertex loss -- exact correspondence (canonical vertex i
    always maps to ARAP-warp output i), so this is a well-posed
    regression, not the ambiguous chamfer/occupancy matching Stage B
    relies on.

  Stage B (mirrors ghd_fit.py's own default config exactly): continues
    optimizing the SAME phi/w/log_s/t (not reset) against the REAL target
    via chamfer/occupancy/volume/etc, same as the production pipeline.

Entirely SEPARATE from ghd/fitting/ghd_fit.py and fit_with_tps_init.py --
duplicates the training loop, so nothing about the production pipeline or
the TPS experiment is modified or even imported-with-side-effects.

conda activate new
python ghd/registration/fit_with_arap_init.py --case-dir /path/to/case
"""

import argparse
import json
import sys
from pathlib import Path

import igl
import numpy as np
import torch
import torch.nn as nn
import trimesh
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from pytorch3d.transforms import axis_angle_to_matrix

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.multi_canonical_ghd_reconstruct import MultiCanonicalGHDReconstruct
from ghd.fitting.losses.chamfer_loss import chamfer_loss
from ghd.fitting.losses.mesh_quality_loss import laplacian_loss, normal_consistency_loss, EdgeLengthLoss
from ghd.fitting.losses.rigid_loss import GHDRigidLoss
from ghd.fitting.losses.mesh_thickness_loss import MeshThickness
from ghd.fitting.losses.volume_loss import VolumeLoss
from ghd.fitting.losses.dvs_loss import DVSOccupancyLoss
from ghd.fitting.losses.geodesic_correspondence_loss import raw_geodesic_distances
from ghd.fitting.ghd_fit import (
    load_target, prepare_dvs_samples, render_fit_sanity, plot_loss_curves,
    _cosine_decay, _linear_decay, setup_experimental_losses,
)
from ghd.fitting.alignment import (
    save_stage1_checkpoints, render_sanity_images,
    load_case_data, load_canonical_data, fit_alignment, DOME_LOADERS,
)
from ghd.registration.tps_register import render_warped_vs_target
from ghd.registration.arap_register import build_ring_handles
from ghd.registration.self_intersection import find_self_intersections
from ghd.fitting.skeleton_alignment import fit_skeleton_alignment


def compute_branch_proximity_weights(V0_np, F_can_np, opening_idx_sets, decay_length=0.3, min_weight=0.1):
    """Per-vertex weight for Stage A's vertex-matching loss: 1.0 AT an
    opening/branch base, decaying exponentially (by graph geodesic distance
    ON THE CANONICAL MESH to the NEAREST opening) down to min_weight deep in
    the dome interior. Same rationale as fit_with_tps_init.py's copy of this
    function: Stage A's job is bending the branches, not matching the dome
    exactly (which, with ARAP, isn't even anchored to anything in
    particular -- see arap_register.py's decision to drop dome handles)."""
    raw = raw_geodesic_distances(V0_np, F_can_np, opening_idx_sets)  # (N, K)
    dist_to_nearest = raw.min(axis=1)
    return min_weight + (1.0 - min_weight) * np.exp(-dist_to_nearest / decay_length)


def fit_with_arap_init(case_dir, canonical_root=None, out_dir=None,
                       n_iter=7500, lr=1e-3, eta_min=1e-4, device="cuda",
                       align_epochs=800, log_every=200,
                       n_iter_stage_a=3000, lr_stage_a=1e-2, eta_min_stage_a=1e-4,
                       lambda_rigid_stage_a=0.05, lambda_laplacian_stage_a=1e-3,
                       lambda_consistency_stage_a=0.03, lambda_edge_stage_a=0.01,
                       branch_weight_decay_length=0.3, branch_weight_min=0.1,
                       lambda_chamfer_n1=0.8, lambda_laplacian=1e-2,
                       lambda_rigid_start=2.0, lambda_rigid_end=0.001, rigid_decay_frac=0.80,
                       lambda_volume=1.0, volume_ceiling_ratio=1.2,
                       volume_target_frac_start=1.0, volume_target_frac_end=1.0, volume_target_decay_frac=0.5,
                       volume_ramp_frac=0.5, lambda_thickness=2.0, thickness_r=0.2,
                       lambda_consistency_start=0.3, lambda_consistency_end=0.3, consistency_decay_frac=0.80,
                       lambda_edge_start=0.1, lambda_edge_end=0.1, edge_decay_frac=0.80,
                       lambda_occupancy=2.0, dvs_surf_d_min=0.0001, dvs_surf_d_max=0.05,
                       lambda_opening_chamfer=0.0, n_opening_chamfer_pts=3000,
                       use_ring_normal=False, ring_normal_blend=1.0,
                       stage_a_intersection_check_min_iter=500,
                       alignment_mode="chamfer",
                       n_surface_samples=20000, skip_stage_a=False):
    """See module docstring. skip_stage_a: same checkpoint/resume mechanism
    as fit_with_tps_init.py -- reuses out_dir/stage_a_checkpoint.npz +
    final_aligned.obj + landmarks.npz from a prior run if present."""
    device = torch.device(device)
    case_dir = Path(case_dir)
    canonical_root = Path(canonical_root) if canonical_root else ROOT / "dataset" / "canonical"
    out_dir = Path(out_dir) if out_dir else ROOT / "runtime" / "arap_init_test" / case_dir.name
    ckpt_dir = out_dir / "rigid_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sanity_dir = out_dir / "sanity"
    sanity_dir.mkdir(parents=True, exist_ok=True)
    sanity_dir_a = out_dir / "sanity_stage_a"
    sanity_dir_a.mkdir(parents=True, exist_ok=True)

    stage_a_ckpt_path = out_dir / "stage_a_checkpoint.npz"
    reuse_stage_a = (skip_stage_a and stage_a_ckpt_path.exists()
                     and (out_dir / "final_aligned.obj").exists()
                     and (out_dir / "landmarks.npz").exists())

    if reuse_stage_a:
        print(f"=== Reusing cached alignment + ARAP + Stage A from {stage_a_ckpt_path} ===", flush=True)
        lm = np.load(out_dir / "landmarks.npz", allow_pickle=True)
        atype = int(lm["aneurysm_type"])
        stage_a_ckpt = np.load(stage_a_ckpt_path)
        warped_verts_phys = stage_a_ckpt["warped_verts_phys"]
        n_arap_handles = int(stage_a_ckpt["n_arap_handles"])
        stage_a_final_vertex_mse = float(stage_a_ckpt["final_vertex_mse"])
    else:
        # Alignment + ARAP registration: one fresh default alignment fit
        # serves both the usual Stage-1 checkpoint export AND the ARAP
        # ring-handle construction (build_ring_handles needs the same
        # case/canonical/transform).
        print("=== Alignment + ARAP registration ===", flush=True)
        case = load_case_data(case_dir, dome_points_loader=DOME_LOADERS["nrrd"])
        atype = case["aneurysm_type"]
        canonical_dir = canonical_root / ("Bifurcated" if atype == 0 else "Sidewall")
        canonical = load_canonical_data(canonical_dir)
        if alignment_mode == "skeleton":
            transform, _ = fit_skeleton_alignment(case, canonical, device=device)
        else:
            transform, _ = fit_alignment(case, canonical, epochs=align_epochs, device=device, log_every=align_epochs)
        save_stage1_checkpoints(out_dir, transform, case, canonical)
        render_sanity_images(out_dir, transform, case, canonical)

        _canon_mesh_tmp = trimesh.load(canonical_dir / "mesh.obj", process=False)
        V_canon_tmp = np.asarray(_canon_mesh_tmp.vertices, dtype=np.float64)
        F_canon_tmp = np.asarray(_canon_mesh_tmp.faces, dtype=np.int64)

        b_handles, bc_targets = build_ring_handles(case, canonical, transform, device, canonical_dir, V_canon_tmp,
                                                   use_ring_normal=use_ring_normal, out_dir=out_dir,
                                                   ring_normal_blend=ring_normal_blend)
        n_arap_handles = int(len(b_handles))
        print(f"  ARAP: {n_arap_handles} ring handle vertices across "
              f"{len(canonical['branch_endpoints'])} branches, no dome anchor", flush=True)

        arap_data = igl.ARAPData()
        igl.arap_precomputation(V_canon_tmp, F_canon_tmp, 3, b_handles.astype(np.int32), arap_data)
        warped_verts_phys = igl.arap_solve(bc_targets, arap_data, V_canon_tmp.copy())

        _vol_ratio = (trimesh.Trimesh(vertices=warped_verts_phys, faces=F_canon_tmp, process=False).volume
                     / _canon_mesh_tmp.volume)
        print(f"  ARAP-warped volume ratio (warped/canonical) = {_vol_ratio:.4f}", flush=True)
        trimesh.Trimesh(vertices=warped_verts_phys, faces=F_canon_tmp, process=False).export(
            out_dir / "canonical_arap_warped.obj")

    canonical_dir = canonical_root / ("Bifurcated" if atype == 0 else "Sidewall")
    canon_mesh = trimesh.load(canonical_dir / "mesh.obj", process=False)

    mc = MultiCanonicalGHDReconstruct(canonical_root, device=device)
    recon = mc.get(atype)
    norm_canonical = recon.norm_canonical
    V0 = recon.canonical_Meshes.verts_packed()
    F_can = recon.canonical_Meshes.faces_packed()
    U = recon.GHD_eigvec
    num_coeffs = U.shape[1]

    warp_target_norm = torch.as_tensor(warped_verts_phys / norm_canonical, dtype=torch.float32, device=device)

    target = load_target(out_dir, norm_canonical, n_surface_samples=n_surface_samples)
    tgt_pts = torch.as_tensor(target["points"], device=device).unsqueeze(0)
    tgt_nrm = torch.as_tensor(target["normals"], device=device).unsqueeze(0)
    target_volume = target["volume_norm"]
    target_verts_phys = target["mesh_verts_norm"] * norm_canonical

    # Raw ARAP output vs. the real target, BEFORE any GHD-basis projection or
    # training at all -- the ceiling on what Stage A could possibly achieve.
    render_warped_vs_target(sanity_dir_a / "arap_raw_vs_real_target.png", warped_verts_phys,
                            canon_mesh.faces, target_verts_phys)

    edge_loss_fn = EdgeLengthLoss(recon.canonical_Meshes)
    rigid_loss_fn = GHDRigidLoss(V0, F_can)

    if reuse_stage_a:
        phi = nn.Parameter(torch.as_tensor(stage_a_ckpt["phi"], dtype=torch.float32, device=device))
        w = nn.Parameter(torch.as_tensor(stage_a_ckpt["w"], dtype=torch.float32, device=device))
        log_s = nn.Parameter(torch.tensor(float(stage_a_ckpt["log_s"]), device=device))
        t = nn.Parameter(torch.as_tensor(stage_a_ckpt["t"], dtype=torch.float32, device=device))
        print(f"  Loaded phi/w/log_s/t from Stage A checkpoint "
              f"(final_vertex_mse={stage_a_final_vertex_mse:.6f})", flush=True)
    else:
        V0_faces_np = F_can.detach().cpu().numpy()
        V0_volume = float(trimesh.Trimesh(vertices=V0.detach().cpu().numpy(), faces=V0_faces_np, process=False).volume)
        size_match = (target_volume / max(V0_volume, 1e-12)) ** (1.0 / 3.0)

        topo = np.load(canonical_dir / "canonical_topology.npy", allow_pickle=True).item()
        opening_idx_sets = [np.asarray(o["indices"], dtype=np.int64) for o in topo["openings"]]
        branch_weight_np = compute_branch_proximity_weights(
            V0.detach().cpu().numpy(), V0_faces_np, opening_idx_sets,
            decay_length=branch_weight_decay_length, min_weight=branch_weight_min)
        branch_weight = torch.as_tensor(branch_weight_np, dtype=torch.float32, device=device)
        print(f"  branch-proximity weight: mean={branch_weight_np.mean():.3f} "
              f"min={branch_weight_np.min():.3f} max={branch_weight_np.max():.3f}", flush=True)
        phi = nn.Parameter(torch.zeros(num_coeffs, 3, device=device))
        w = nn.Parameter(torch.zeros(3, device=device))
        log_s = nn.Parameter(torch.tensor(float(np.log(max(size_match, 1e-6))), device=device))
        t = nn.Parameter(torch.zeros(3, device=device))

    def render():
        offset = torch.einsum('nm,mc->nc', U, phi)
        V_deformed = V0 + offset
        R = axis_angle_to_matrix(w.unsqueeze(0)).squeeze(0)
        V_rendered = (V_deformed @ R.T) * torch.exp(log_s) + t
        return Meshes(verts=[V_rendered], faces=[F_can])

    if not reuse_stage_a:
        with torch.no_grad():
            rest_verts_phys = (render().verts_packed() * norm_canonical).cpu().numpy()
        render_fit_sanity(sanity_dir_a / "iter_before_rest_pose_vs_arap_target.png", rest_verts_phys,
                          F_can.cpu().numpy(), warped_verts_phys)

        # ══════════════════════════════════════════════════════════════════
        # Stage A: fit phi + pose toward the ARAP-warped canonical mesh
        # (exact per-vertex correspondence -- no sampling/chamfer needed).
        # ══════════════════════════════════════════════════════════════════
        print(f"=== Stage A (fit to ARAP-warped mesh, {n_iter_stage_a} iters) ===", flush=True)
        opt_a = torch.optim.Adam([phi, w, log_s, t], lr=lr_stage_a)
        sched_a = torch.optim.lr_scheduler.CosineAnnealingLR(opt_a, T_max=n_iter_stage_a, eta_min=eta_min_stage_a)
        history_a = {k: [] for k in ["vertex_mse", "laplacian", "consistency", "edge", "rigid", "total", "lr"]}

        # Self-intersection can't be prevented during Stage A's gradient
        # descent (no term in the loss checks for it -- see conversation),
        # only DETECTED after the fact. So instead of trying to fix it in
        # the loss, checkpoint phi/w/log_s/t at every log_every snapshot
        # (once past stage_a_intersection_check_min_iter, since early
        # iterations are essentially guaranteed to still be tangled) and,
        # once training finishes, roll back to the LATEST checkpoint that
        # was actually clean -- rather than trusting whatever the final
        # iteration happened to land on.
        last_clean_checkpoint = None  # (iter, phi, w, log_s, t)

        for it in range(n_iter_stage_a):
            mesh = render()
            loss_vertex = (((mesh.verts_packed() - warp_target_norm) ** 2).sum(-1) * branch_weight).mean()
            loss_laplacian = laplacian_loss(mesh)
            loss_consistency = normal_consistency_loss(mesh)
            loss_edge = edge_loss_fn(mesh)
            loss_rigid = rigid_loss_fn(mesh.verts_packed())

            total_a = (loss_vertex
                      + lambda_laplacian_stage_a * loss_laplacian
                      + lambda_consistency_stage_a * loss_consistency
                      + lambda_edge_stage_a * loss_edge
                      + lambda_rigid_stage_a * loss_rigid)

            opt_a.zero_grad()
            total_a.backward()
            opt_a.step()
            sched_a.step()

            history_a["vertex_mse"].append(loss_vertex.item())
            history_a["laplacian"].append(loss_laplacian.item())
            history_a["consistency"].append(loss_consistency.item())
            history_a["edge"].append(loss_edge.item())
            history_a["rigid"].append(loss_rigid.item())
            history_a["total"].append(total_a.item())
            history_a["lr"].append(opt_a.param_groups[0]["lr"])

            if it % log_every == 0 or it == n_iter_stage_a - 1:
                print(f"  [A {it:5d}/{n_iter_stage_a}] vertex_mse={loss_vertex.item():.6f} "
                      f"rigid={loss_rigid.item():.4f} laplacian={loss_laplacian.item():.5f} "
                      f"total={total_a.item():.5f}", flush=True)
                with torch.no_grad():
                    snap_verts_phys = (mesh.verts_packed() * norm_canonical).cpu().numpy()
                render_fit_sanity(sanity_dir_a / f"iter_{it:05d}.png", snap_verts_phys,
                                  F_can.cpu().numpy(), warped_verts_phys)

                if it >= stage_a_intersection_check_min_iter:
                    snap_mesh = trimesh.Trimesh(vertices=snap_verts_phys, faces=F_can.cpu().numpy(), process=False)
                    pairs = find_self_intersections(snap_mesh)
                    print(f"    self-intersection check @ iter {it}: {len(pairs)} pair(s)", flush=True)
                    if len(pairs) == 0:
                        last_clean_checkpoint = (it, phi.detach().clone(), w.detach().clone(),
                                                 log_s.detach().clone(), t.detach().clone())

        if last_clean_checkpoint is not None and last_clean_checkpoint[0] != n_iter_stage_a - 1:
            ckpt_it, ckpt_phi, ckpt_w, ckpt_log_s, ckpt_t = last_clean_checkpoint
            print(f"  Final Stage A state is self-intersecting (or unchecked) -- rolling back "
                  f"to last known-clean checkpoint @ iter {ckpt_it}.", flush=True)
            phi.data.copy_(ckpt_phi); w.data.copy_(ckpt_w)
            log_s.data.copy_(ckpt_log_s); t.data.copy_(ckpt_t)
            stage_a_final_vertex_mse = history_a["vertex_mse"][ckpt_it]
        elif last_clean_checkpoint is None:
            print(f"  WARNING: no self-intersection-free checkpoint found at/after iter "
                  f"{stage_a_intersection_check_min_iter} -- keeping final iteration's state as-is.",
                  flush=True)
            stage_a_final_vertex_mse = history_a["vertex_mse"][-1]
        else:
            stage_a_final_vertex_mse = history_a["vertex_mse"][-1]
        np.savez(out_dir / "stage_a_loss_history.npz", **{k: np.asarray(v) for k, v in history_a.items()})
        plot_loss_curves(history_a, sanity_dir_a / "stage_a_loss_curves.png")
        with torch.no_grad():
            stage_a_verts_phys = (render().verts_packed() * norm_canonical).cpu().numpy()
        render_fit_sanity(sanity_dir_a / "final_vs_real_target.png", stage_a_verts_phys, F_can.cpu().numpy(), target_verts_phys)
        render_fit_sanity(sanity_dir_a / "final_vs_arap_target.png", stage_a_verts_phys,
                          F_can.cpu().numpy(), warped_verts_phys)
        print(f"  Stage A done -> {sanity_dir_a / 'final_vs_real_target.png'} "
              f"(vs. real target), {sanity_dir_a / 'final_vs_arap_target.png'} "
              f"(vs. ARAP target -- the actual correctness check), "
              f"per-iteration snapshots in {sanity_dir_a}/", flush=True)

        np.savez(stage_a_ckpt_path,
                phi=phi.detach().cpu().numpy(), w=w.detach().cpu().numpy(),
                log_s=np.array(log_s.detach().cpu().item()), t=t.detach().cpu().numpy(),
                warped_verts_phys=warped_verts_phys, n_arap_handles=np.array(n_arap_handles),
                final_vertex_mse=stage_a_final_vertex_mse)
        print(f"  Stage A checkpoint saved -> {stage_a_ckpt_path}", flush=True)

    # ══════════════════════════════════════════════════════════════════════
    # Stage B: continue optimizing the SAME phi/w/log_s/t toward the real
    # target -- mirrors ghd_fit.py's own default-config loop exactly
    # (DUPLICATED, not imported -- see module docstring for why).
    # ══════════════════════════════════════════════════════════════════════
    print(f"=== Stage B (fit to real target, {n_iter} iters) ===", flush=True)
    dvs_samples = None
    if lambda_occupancy > 0:
        dvs_samples = prepare_dvs_samples(target, device, d_min=dvs_surf_d_min, d_max=dvs_surf_d_max)
        if dvs_samples is None:
            print("  WARNING: could not close target mesh for occupancy -- disabling "
                  "(lambda_occupancy forced to 0).", flush=True)
            lambda_occupancy = 0.0
        else:
            dvs_samples = tuple(t_[:4000] for t_ in dvs_samples)

    experimental = setup_experimental_losses(
        mc, atype, canonical_root, target, device, n_opening_chamfer_pts=n_opening_chamfer_pts,
    ) if lambda_opening_chamfer > 0 else {}
    if lambda_opening_chamfer > 0 and "opening_chamfer" not in experimental:
        print("  WARNING: no usable opening/cap data for opening-chamfer -- disabling "
              "(lambda forced to 0).", flush=True)
        lambda_opening_chamfer = 0.0

    optimizer = torch.optim.Adam([phi, w, log_s, t], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iter, eta_min=eta_min)
    thickness_fn = MeshThickness(r=thickness_r)
    volume_loss_fn = VolumeLoss(target_volume=target_volume, ceiling_ratio=volume_ceiling_ratio)
    occupancy_fn = DVSOccupancyLoss(num_sample=4000) if lambda_occupancy > 0 else None

    auto_thresholds = np.linspace(lambda_rigid_start, lambda_rigid_end, 10)
    extra_thresholds = [v for v in (0.1, 0.05, 0.01)
                       if min(lambda_rigid_start, lambda_rigid_end) <= v <= max(lambda_rigid_start, lambda_rigid_end)]
    rigid_weight_thresholds = sorted(set(np.round(np.concatenate([auto_thresholds, extra_thresholds]), 6)), reverse=True)
    next_ckpt_idx = 0

    def save_rigid_checkpoint(mesh_, iteration, rigid_weight, rigid_value):
        tag = f"w{rigid_weight:.3f}"
        with torch.no_grad():
            verts_phys_ = (mesh_.verts_packed() * norm_canonical).detach().cpu().numpy()
        faces_np_ = F_can.detach().cpu().numpy()
        trimesh.Trimesh(vertices=verts_phys_, faces=faces_np_, process=False).export(ckpt_dir / f"ghd_fitted_{tag}.obj")
        np.savez(ckpt_dir / f"ghd_coefficients_{tag}.npz",
                phi=phi.detach().cpu().numpy(), w_rot=w.detach().cpu().numpy(),
                log_scale=np.array(log_s.detach().cpu().item()), t_vec=t.detach().cpu().numpy(),
                iteration=np.array(iteration), lambda_rigid_w=np.array(rigid_weight), rigid_loss=np.array(rigid_value))
        render_fit_sanity(sanity_dir / f"sanity_{tag}.png", verts_phys_, faces_np_, target_verts_phys)
        print(f"  rigid-checkpoint saved: lambda_rigid_w<={rigid_weight:.3f} @ iter {iteration} "
              f"(rigid_loss={rigid_value:.4f}) -> {tag}", flush=True)

    history = {k: [] for k in [
        "chamfer", "chamfer_n1", "laplacian", "consistency", "edge", "rigid",
        "volume", "thickness", "occupancy", "opening_chamfer", "total", "lambda_rigid_w",
        "volume_target_frac", "lambda_volume_w", "lambda_consistency_w", "lambda_edge_w",
        "lr", "rot_norm", "scale", "trans_norm",
    ]}
    best_chamfer = float("inf")
    best_state = None

    for it in range(n_iter):
        frac = it / max(n_iter - 1, 1)
        lam_rigid = _cosine_decay(lambda_rigid_start, lambda_rigid_end, frac, rigid_decay_frac)
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

        while (next_ckpt_idx < len(rigid_weight_thresholds)
               and lam_rigid <= rigid_weight_thresholds[next_ckpt_idx]):
            save_rigid_checkpoint(mesh, it, rigid_weight_thresholds[next_ckpt_idx], loss_rigid.item())
            next_ckpt_idx += 1

        dist, dist_v_norm, _, sign = thickness_fn(mesh)
        mask = (dist_v_norm.abs() > 0.1).logical_not().float()
        signed = torch.sign(sign)
        loss_thickness = (torch.relu(0.04 - dist_v_norm * signed) + torch.relu(0.01 - dist * signed)) * mask
        loss_thickness = loss_thickness.mean() + (1e-4 / (sign ** 2 + 1e-6) * mask).mean()

        loss_occupancy = torch.zeros((), device=device)
        if occupancy_fn is not None:
            pos, neg, wp2n, wn2p = dvs_samples
            loss_occupancy = occupancy_fn(mesh, pos, neg, wp2n, wn2p)

        loss_opening_chamfer = torch.zeros((), device=device)
        if lambda_opening_chamfer > 0:
            oc = experimental["opening_chamfer"]
            can_cap_mesh = Meshes(verts=[mesh.verts_packed()], faces=[F_can[oc["can_cap_face_idx"]]])
            can_cap_pts = sample_points_from_meshes(can_cap_mesh, n_opening_chamfer_pts)
            loss_opening_chamfer, _ = chamfer_loss(can_cap_pts, oc["tgt_cap_pts"])

        total = (loss_chamfer + lambda_chamfer_n1 * loss_chamfer_n1
                + lambda_laplacian * loss_laplacian
                + lam_consistency * loss_consistency
                + lam_edge * loss_edge
                + lam_rigid * loss_rigid
                + lam_volume * loss_volume
                + lambda_thickness * loss_thickness
                + lambda_occupancy * loss_occupancy
                + lambda_opening_chamfer * loss_opening_chamfer)

        optimizer.zero_grad()
        total.backward()
        optimizer.step()
        scheduler.step()

        cur_chamfer = loss_chamfer.item()
        if cur_chamfer < best_chamfer:
            best_chamfer = cur_chamfer
            best_state = {"phi": phi.detach().clone(), "w": w.detach().clone(),
                         "log_s": log_s.detach().clone(), "t": t.detach().clone(), "iter": it}

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
            print(f"  [B {it:6d}/{n_iter}] chamfer={cur_chamfer:.5f} rigid={loss_rigid.item():.4f} "
                  f"volume={loss_volume.item():.4f} (w={lam_volume:.3f}) thickness={loss_thickness.item():.4f} "
                  f"edge={loss_edge.item():.4f} consistency={loss_consistency.item():.4f} "
                  f"occ={loss_occupancy.item():.4f} total={total.item():.4f}", flush=True)

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
             phi=best_state["phi"].cpu().numpy(), w_rot=best_state["w"].cpu().numpy(),
             log_scale=np.array(best_state["log_s"].cpu().item()), t_vec=best_state["t"].cpu().numpy())

    metrics = {
        "chamfer_final": float(chamfer_final.item()), "chamfer_best": best_chamfer,
        "best_iter": best_state["iter"], "aneurysm_type": atype, "s_can": float(norm_canonical),
        "n_arap_handles": n_arap_handles,
        "stage_a_final_vertex_mse": stage_a_final_vertex_mse,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    np.savez(out_dir / "loss_history.npz", **{k: np.asarray(v) for k, v in history.items()})
    plot_loss_curves(history, out_dir / "loss_curves.png")

    print(f"Done. chamfer_best={best_chamfer:.5f} @ iter {best_state['iter']} -> {out_dir}", flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--canonical-root", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--n-iter", type=int, default=7500, help="Stage B iterations.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Stage B base learning rate.")
    parser.add_argument("--n-iter-stage-a", type=int, default=3000)
    parser.add_argument("--lr-stage-a", type=float, default=1e-2)
    parser.add_argument("--align-epochs", type=int, default=800)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eta-min", type=float, default=1e-4)
    parser.add_argument("--eta-min-stage-a", type=float, default=1e-4)
    parser.add_argument("--lambda-occupancy", type=float, default=2.0)
    parser.add_argument("--lambda-opening-chamfer", type=float, default=0.0,
                        help="Weak chamfer loss between the mesh's own opening/cap faces and "
                             "the target's sampled cap points -- see ghd_fit.py's "
                             "setup_experimental_losses. 0.0 = disabled (default).")
    parser.add_argument("--n-opening-chamfer-pts", type=int, default=3000)
    parser.add_argument("--use-ring-normal", action="store_true",
                        help="Orient ring handles by each ring's own outward normal instead of "
                             "the coarse centerline-tangent estimate -- see arap_register.py.")
    parser.add_argument("--ring-normal-blend", type=float, default=1.0,
                        help="SLERP blend factor for --use-ring-normal, 0=pure tangent, "
                             "1=pure ring-normal. 0.5 found to avoid the cap-region "
                             "self-intersection full ring-normal (1.0) introduced.")
    parser.add_argument("--stage-a-intersection-check-min-iter", type=int, default=500,
                        help="Only checkpoint/check Stage A snapshots for self-intersection at "
                             "or after this iteration -- earlier ones are essentially guaranteed "
                             "still tangled. Stage A rolls back to the latest clean checkpoint "
                             "if the final iteration isn't clean.")
    parser.add_argument("--alignment-mode", choices=["chamfer", "skeleton"], default="chamfer",
                        help="'chamfer' (default): fit_alignment's closed-form init + 800-epoch "
                             "gradient-descent refinement against surface/centerline/volume/dome "
                             "losses. 'skeleton': skeleton_alignment.py's closed-form-only fit "
                             "(branch endpoints + dome/neck centroid), no refinement, no mesh "
                             "surface used at all -- ported from AneuSeg/prep_case_for_ghd.py's "
                             "approach, as a test of whether the gradient refinement sometimes "
                             "makes things worse.")
    parser.add_argument("--lambda-rigid-stage-a", type=float, default=0.05)
    parser.add_argument("--lambda-laplacian-stage-a", type=float, default=1e-3)
    parser.add_argument("--lambda-consistency-stage-a", type=float, default=0.03)
    parser.add_argument("--lambda-edge-stage-a", type=float, default=0.01)
    parser.add_argument("--branch-weight-decay-length", type=float, default=0.3)
    parser.add_argument("--branch-weight-min", type=float, default=0.1)
    parser.add_argument("--skip-stage-a", action="store_true")
    args = parser.parse_args()
    fit_with_arap_init(args.case_dir, args.canonical_root, args.out_dir, n_iter=args.n_iter,
                       lr=args.lr, eta_min=args.eta_min, device=args.device,
                       align_epochs=args.align_epochs, lambda_occupancy=args.lambda_occupancy,
                       lambda_opening_chamfer=args.lambda_opening_chamfer,
                       n_opening_chamfer_pts=args.n_opening_chamfer_pts,
                       use_ring_normal=args.use_ring_normal, ring_normal_blend=args.ring_normal_blend,
                       stage_a_intersection_check_min_iter=args.stage_a_intersection_check_min_iter,
                       alignment_mode=args.alignment_mode,
                       n_iter_stage_a=args.n_iter_stage_a, lambda_rigid_stage_a=args.lambda_rigid_stage_a,
                       lr_stage_a=args.lr_stage_a, eta_min_stage_a=args.eta_min_stage_a,
                       lambda_laplacian_stage_a=args.lambda_laplacian_stage_a,
                       lambda_consistency_stage_a=args.lambda_consistency_stage_a,
                       lambda_edge_stage_a=args.lambda_edge_stage_a,
                       branch_weight_decay_length=args.branch_weight_decay_length,
                       branch_weight_min=args.branch_weight_min, skip_stage_a=args.skip_stage_a)


if __name__ == "__main__":
    main()
