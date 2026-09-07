"""fit_with_tps_init.py -- standalone experiment: TWO sequential gradient-
based fitting stages sharing the same phi/pose parameters, instead of
ghd_fit.py's single stage starting phi at zero:

  Stage A (new): starting from the TRUE native rest pose (phi=0, w=0, t=0),
    fit phi + residual pose toward the TPS-warped canonical mesh (see
    tps_register.py) via a DIRECT per-vertex loss -- exact correspondence
    (canonical vertex i always maps to TPS-warp output i), so this is a
    well-posed regression, not the ambiguous chamfer/occupancy matching
    Stage B relies on. Regularized the same way Stage B is (laplacian,
    edge, normal-consistency, rigid) so the result is still a well-formed,
    properly-regularized mesh -- NOT a raw least-squares projection onto
    the eigenbasis (tried first, rejected: it ignores mesh regularity
    entirely and produces something that was never actually fit to
    anything, just algebraically closest in a truncated basis).

  Stage B (mirrors ghd_fit.py's own default config exactly): continues
    optimizing the SAME phi/w/log_s/t (not reset) against the REAL target
    via chamfer/occupancy/volume/etc, same as the production pipeline --
    the only difference from the standard zero-init baseline is where it
    starts from.

WHY THIS MATTERS BEYOND THIS ONE CASE: the actual deliverable of Stage B is
phi itself -- it's meant to become training data for a VAE-based generative
model over the GHD coefficient space, not just a rendered mesh. That's why
Stage A has to be a real regularized fit, not a shortcut: phi needs to be a
genuine, well-formed point in that space, consistently produced the same
way across every case, since inconsistent or degenerate phi extraction
would corrupt whatever the VAE later learns from it.

TPS and the GHD basis are UNRELATED math, worth being explicit about: TPS
is an extrinsic landmark warp in raw 3D space; phi is a coefficient vector
in the canonical mesh's own intrinsic graph-Laplacian eigenbasis. Stage A
is the bridge between them -- an actual optimization, not an algebraic
identity.

Entirely SEPARATE from ghd/fitting/ghd_fit.py -- imports its loss/data-
loading utilities and tps_register.py's TPS fit, but duplicates the
training loop itself, so nothing about the production pipeline (ghd_fit.py,
run_case.py) is modified or even imported-with-side-effects. Safe to
delete this whole file without affecting anything else.

conda activate new
python ghd/registration/fit_with_tps_init.py --case-dir /path/to/case
"""

import argparse
import json
import sys
from pathlib import Path

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
    _cosine_decay, _linear_decay,
)
from ghd.fitting.alignment import save_stage1_checkpoints, render_sanity_images
from ghd.registration.tps_register import build_landmarks, fit_tps, render_warped_vs_target


def compute_branch_proximity_weights(V0_np, F_can_np, opening_idx_sets, decay_length=0.3, min_weight=0.1):
    """Per-vertex weight for Stage A's vertex-matching loss: 1.0 AT an
    opening/branch base, decaying exponentially (by graph geodesic distance
    ON THE CANONICAL MESH to the NEAREST opening -- reusing
    raw_geodesic_distances, the same building block the experimental
    geodesic-correspondence loss already uses) down to min_weight deep in
    the dome interior.

    Stage A's actual job is bending the branches to the right location
    (see module docstring) -- the dome interior has no landmark
    constraints of its own in the TPS fit and can come out badly
    distorted there (see the registration diagnosis this whole experiment
    grew out of), so fighting equally hard to match it exactly just
    competes with the rigid/laplacian regularizers for no real benefit.
    Downweighting it lets the optimizer spend its effort where the
    correspondence is actually trustworthy: near the branches."""
    raw = raw_geodesic_distances(V0_np, F_can_np, opening_idx_sets)  # (N, K)
    dist_to_nearest = raw.min(axis=1)
    return min_weight + (1.0 - min_weight) * np.exp(-dist_to_nearest / decay_length)


def fit_with_tps_init(case_dir, canonical_root=None, out_dir=None,
                      n_iter=7500, lr=1e-3, eta_min=1e-4, device="cuda",
                      n_resample=64, align_epochs=800, log_every=200,
                      # Stage A (fit toward the TPS-warped canonical mesh) --
                      # all four regularizers relaxed to ~10% of Stage B's
                      # steady-state defaults: Stage A's job is WARPING
                      # performance (matching the TPS mesh), not mesh
                      # quality -- kept nonzero so the mesh doesn't
                      # degenerate, but not so large they fight the warp.
                      n_iter_stage_a=3000, lr_stage_a=1e-2, eta_min_stage_a=1e-4,
                      lambda_rigid_stage_a=0.05, lambda_laplacian_stage_a=1e-3,
                      lambda_consistency_stage_a=0.03, lambda_edge_stage_a=0.01,
                      branch_weight_decay_length=0.3, branch_weight_min=0.1,
                      # Stage B (fit toward the real target -- matches
                      # ghd_fit.py's own default config exactly)
                      lambda_chamfer_n1=0.8, lambda_laplacian=1e-2,
                      lambda_rigid_start=2.0, lambda_rigid_end=0.001, rigid_decay_frac=0.80,
                      lambda_volume=1.0, volume_ceiling_ratio=1.2,
                      volume_target_frac_start=1.0, volume_target_frac_end=1.0, volume_target_decay_frac=0.5,
                      volume_ramp_frac=0.5, lambda_thickness=2.0, thickness_r=0.2,
                      lambda_consistency_start=0.3, lambda_consistency_end=0.3, consistency_decay_frac=0.80,
                      lambda_edge_start=0.1, lambda_edge_end=0.1, edge_decay_frac=0.80,
                      lambda_occupancy=1.0, dvs_surf_d_min=0.0001, dvs_surf_d_max=0.05,
                      n_surface_samples=20000, skip_stage_a=False):
    """See module docstring for the two-stage design. Stage B's lambda_*
    defaults match ghd_fit.py's own "default" config exactly -- the only
    difference from the standard zero-init baseline (runtime/<dataset>/
    <case>/ for the same case) is that phi/w/log_s/t start whatever Stage A
    converged to, not zeros.

    skip_stage_a: if True and out_dir/stage_a_checkpoint.npz (+ the usual
    final_aligned.obj/landmarks.npz) already exist from a PRIOR run, skips
    alignment + TPS registration + Stage A entirely and loads phi/w/log_s/t
    straight from that checkpoint -- lets you iterate on Stage B's own
    hyperparameters (lr, rigid schedule, etc.) without redoing the
    expensive, deterministic-anyway upstream work every time. Silently
    falls back to running everything fresh if the checkpoint is missing."""
    device = torch.device(device)
    case_dir = Path(case_dir)
    canonical_root = Path(canonical_root) if canonical_root else ROOT / "dataset" / "canonical"
    out_dir = Path(out_dir) if out_dir else ROOT / "runtime" / "tps_init_test" / case_dir.name
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
        print(f"=== Reusing cached alignment + TPS + Stage A from {stage_a_ckpt_path} ===", flush=True)
        lm = np.load(out_dir / "landmarks.npz", allow_pickle=True)
        atype = int(lm["aneurysm_type"])
        stage_a_ckpt = np.load(stage_a_ckpt_path)
        warped_verts_phys = stage_a_ckpt["warped_verts_phys"]
        tps_rbf_weight_norm = float(stage_a_ckpt["rbf_weight_norm"])
        stage_a_final_vertex_mse = float(stage_a_ckpt["final_vertex_mse"])
    else:
        # Alignment + TPS registration: one fresh default alignment fit
        # serves both the usual Stage-1 checkpoint export AND the TPS
        # landmark warp (build_landmarks already does the alignment
        # internally).
        print("=== Alignment + TPS registration ===", flush=True)
        canonical_landmarks, case_landmarks, case, canonical, transform = build_landmarks(
            case_dir, canonical_root=canonical_root, align_epochs=align_epochs, device=device,
            n_resample=n_resample)
        save_stage1_checkpoints(out_dir, transform, case, canonical)
        render_sanity_images(out_dir, transform, case, canonical)

        warp = fit_tps(canonical_landmarks, case_landmarks)
        print(f"  TPS: rbf_weight_norm={warp.rbf_weight_norm:.4f}  affine_norm={warp.affine_norm:.4f}", flush=True)
        tps_rbf_weight_norm = warp.rbf_weight_norm

        atype = case["aneurysm_type"]
        _canonical_dir_tmp = canonical_root / ("Bifurcated" if atype == 0 else "Sidewall")
        _canon_mesh_tmp = trimesh.load(_canonical_dir_tmp / "mesh.obj", process=False)
        warped_verts_phys = warp(np.asarray(_canon_mesh_tmp.vertices, dtype=np.float64))

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

    # Raw TPS output vs. the real target, BEFORE any GHD-basis projection or
    # training at all -- the ceiling on what Stage A could possibly achieve
    # (Stage A can only get as good as this, since it's fitting phi to
    # approximate this exact mesh, not exceed it).
    render_warped_vs_target(sanity_dir_a / "tps_raw_vs_real_target.png", warped_verts_phys,
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
        # ── TRUE native rest pose (phi=0, w=0, t=0), only log_s gets a non-
        # trivial init (cube-root volume-ratio, same estimator ghd_fit.py
        # itself uses) since scale is a reasonable, cheap guess to start
        # from and isn't specific to either stage's target. ──
        V0_faces_np = F_can.detach().cpu().numpy()
        V0_volume = float(trimesh.Trimesh(vertices=V0.detach().cpu().numpy(), faces=V0_faces_np, process=False).volume)
        size_match = (target_volume / max(V0_volume, 1e-12)) ** (1.0 / 3.0)

        # Per-vertex weight for Stage A: 1.0 at the branches, decaying toward
        # branch_weight_min deep in the dome -- see compute_branch_proximity_weights.
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
        render_fit_sanity(sanity_dir_a / "iter_before_rest_pose_vs_tps_target.png", rest_verts_phys,
                          F_can.cpu().numpy(), warped_verts_phys)

        # ══════════════════════════════════════════════════════════════════
        # Stage A: fit phi + pose toward the TPS-warped canonical mesh (exact
        # per-vertex correspondence -- no sampling/chamfer needed).
        # ══════════════════════════════════════════════════════════════════
        print(f"=== Stage A (fit to TPS-warped mesh, {n_iter_stage_a} iters) ===", flush=True)
        opt_a = torch.optim.Adam([phi, w, log_s, t], lr=lr_stage_a)
        sched_a = torch.optim.lr_scheduler.CosineAnnealingLR(opt_a, T_max=n_iter_stage_a, eta_min=eta_min_stage_a)
        history_a = {k: [] for k in ["vertex_mse", "laplacian", "consistency", "edge", "rigid", "total", "lr"]}

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
                # Snapshot vs. the TPS target (what Stage A is ACTUALLY trying
                # to match) at the same cadence as the printed log -- lets you
                # watch the warp-toward-TPS-mesh progress directly, not just
                # the loss number, in its own folder separate from Stage B's
                # sanity/.
                with torch.no_grad():
                    snap_verts_phys = (mesh.verts_packed() * norm_canonical).cpu().numpy()
                render_fit_sanity(sanity_dir_a / f"iter_{it:05d}.png", snap_verts_phys,
                                  F_can.cpu().numpy(), warped_verts_phys)

        stage_a_final_vertex_mse = history_a["vertex_mse"][-1]
        np.savez(out_dir / "stage_a_loss_history.npz", **{k: np.asarray(v) for k, v in history_a.items()})
        plot_loss_curves(history_a, sanity_dir_a / "stage_a_loss_curves.png")
        with torch.no_grad():
            stage_a_verts_phys = (render().verts_packed() * norm_canonical).cpu().numpy()
        render_fit_sanity(sanity_dir_a / "final_vs_real_target.png", stage_a_verts_phys, F_can.cpu().numpy(), target_verts_phys)
        # THE actual correctness check for Stage A: its result vs. the TPS-
        # warped mesh it was actually trained to match (the previous line
        # compares against the real target instead, useful for progress-
        # tracking but not for verifying Stage A itself converged to what
        # it was supposed to).
        render_fit_sanity(sanity_dir_a / "final_vs_tps_target.png", stage_a_verts_phys,
                          F_can.cpu().numpy(), warped_verts_phys)
        print(f"  Stage A done -> {sanity_dir_a / 'final_vs_real_target.png'} "
              f"(vs. real target), {sanity_dir_a / 'final_vs_tps_target.png'} "
              f"(vs. TPS target -- the actual correctness check), "
              f"per-iteration snapshots in {sanity_dir_a}/", flush=True)

        # Checkpoint phi/w/log_s/t (+ everything Stage B needs to be rebuilt
        # cheaply) so a later run with --skip-stage-a can jump straight to
        # Stage B without redoing alignment + TPS + this training loop.
        np.savez(stage_a_ckpt_path,
                phi=phi.detach().cpu().numpy(), w=w.detach().cpu().numpy(),
                log_s=np.array(log_s.detach().cpu().item()), t=t.detach().cpu().numpy(),
                warped_verts_phys=warped_verts_phys, rbf_weight_norm=tps_rbf_weight_norm,
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

    # Fresh optimizer/scheduler for Stage B -- Adam's momentum from fitting
    # a completely different objective (Stage A's exact vertex targets)
    # isn't meaningful carried into chamfer/occupancy fitting; only the
    # PARAMETER VALUES (phi/w/log_s/t) carry over, which is the whole point.
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
        "volume", "thickness", "occupancy", "total", "lambda_rigid_w",
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

        total = (loss_chamfer + lambda_chamfer_n1 * loss_chamfer_n1
                + lambda_laplacian * loss_laplacian
                + lam_consistency * loss_consistency
                + lam_edge * loss_edge
                + lam_rigid * loss_rigid
                + lam_volume * loss_volume
                + lambda_thickness * loss_thickness
                + lambda_occupancy * loss_occupancy)

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
        "tps_rbf_weight_norm": tps_rbf_weight_norm,
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
    parser.add_argument("--lr-stage-a", type=float, default=1e-2,
                        help="Higher than Stage B's lr -- Stage A is an easy, exact-"
                             "correspondence regression, not the ambiguous chamfer/occupancy "
                             "matching Stage B does, so it can tolerate a much bigger step size.")
    parser.add_argument("--align-epochs", type=int, default=800)
    parser.add_argument("--n-resample", type=int, default=64,
                        help="TPS landmarks per branch, see tps_register.py.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eta-min", type=float, default=1e-4)
    parser.add_argument("--eta-min-stage-a", type=float, default=1e-4)
    parser.add_argument("--lambda-occupancy", type=float, default=1.0)
    parser.add_argument("--lambda-rigid-stage-a", type=float, default=0.05,
                        help="Relaxed to ~10%% of Stage B's rigid start (2.0) -- Stage A's "
                             "job is warping performance, not mesh quality, see module docstring.")
    parser.add_argument("--lambda-laplacian-stage-a", type=float, default=1e-3)
    parser.add_argument("--lambda-consistency-stage-a", type=float, default=0.03)
    parser.add_argument("--lambda-edge-stage-a", type=float, default=0.01)
    parser.add_argument("--branch-weight-decay-length", type=float, default=0.3,
                        help="Geodesic decay length (normalized-space units) for the per-"
                             "vertex branch-proximity weight in Stage A's loss -- see "
                             "compute_branch_proximity_weights.")
    parser.add_argument("--branch-weight-min", type=float, default=0.1,
                        help="Floor weight for vertices far from any branch (dome interior).")
    parser.add_argument("--skip-stage-a", action="store_true",
                        help="Reuse a cached alignment + TPS + Stage A result from a prior run "
                             "in the same --out-dir (out_dir/stage_a_checkpoint.npz) instead of "
                             "redoing it -- lets you iterate on Stage B's own hyperparameters "
                             "quickly. Falls back to running everything fresh if no checkpoint "
                             "is found.")
    args = parser.parse_args()
    fit_with_tps_init(args.case_dir, args.canonical_root, args.out_dir, n_iter=args.n_iter,
                      lr=args.lr, eta_min=args.eta_min, device=args.device, n_resample=args.n_resample,
                      align_epochs=args.align_epochs, lambda_occupancy=args.lambda_occupancy,
                      n_iter_stage_a=args.n_iter_stage_a, lambda_rigid_stage_a=args.lambda_rigid_stage_a,
                      lr_stage_a=args.lr_stage_a, eta_min_stage_a=args.eta_min_stage_a,
                      lambda_laplacian_stage_a=args.lambda_laplacian_stage_a,
                      lambda_consistency_stage_a=args.lambda_consistency_stage_a,
                      lambda_edge_stage_a=args.lambda_edge_stage_a,
                      branch_weight_decay_length=args.branch_weight_decay_length,
                      branch_weight_min=args.branch_weight_min, skip_stage_a=args.skip_stage_a)


if __name__ == "__main__":
    main()
