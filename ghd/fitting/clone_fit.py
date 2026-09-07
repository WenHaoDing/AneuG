"""clone_fit.py -- "fitting cloning": TWO-STAGE fit that warm-starts phi
from an OLD project iteration's already-fitted result, then continues
fitting to THIS case's REAL target using the standard pipeline's default
setup.

WHY: the old phi (ghd_coefficients.npz's "phi") can't be reused directly
-- it's expressed in the OLD canonical's eigenbasis/scale (a different
mesh resolution and a different norm_canonical), so the numbers are
meaningless against today's basis. BUT the old run's ghd_fitted.obj is
just a MESH -- real 3D geometry, basis-independent. So instead of trying
to transplant phi, treat that old mesh as a warm-start TARGET and re-fit
the CURRENT canonical's phi/pose toward it first (Stage A), before
switching to the case's actual real target (Stage B).

  Stage 1 (alignment) runs FIRST -- it DEFINES the frame everything else
    lives in. The clone mesh arrives in the OLD pipeline's own rigid frame
    (neck at origin, R_frame from ring geometry, normalised by its own
    s_can = ||V_can||max * 1.10 * 2.5), which is unrelated to this repo's
    similarity frame -- on a sample case, 53 degrees and 0.88x apart. So the
    clone is reframed onto this case's real target by a surface-chamfer
    similarity fit (ghd/registration/clone_reframe.py) BEFORE Stage A runs.
    Without this the warm start is actively harmful: Stage A converges onto a
    pose in the foreign frame, then Stage B swaps in a target tens of degrees
    away while keeping that pose. The reframing is metadata-free and its
    scale absorbs s_can, so the old *1.10*2.5 never has to be reproduced.

  Stage A (warm start toward the OLD fitted mesh): starting from the
    native rest pose (phi=0, w=0, t=0), fit phi + pose toward the old
    ghd_fitted.obj via the same loss battery ghd_fit.py uses (chamfer +
    laplacian + edge + consistency + volume + thickness + occupancy),
    PLUS a node-to-node MSE. NOTE: an earlier version of this file claimed
    the old and current canonicals have different topology and that only a
    nearest-neighbour match was possible. That is FALSE -- they are
    byte-identical (V=4143, F=8282, same face array), so vertex i of the
    clone IS vertex i of the canonical and an exact indexed correspondence
    exists. KNN was measurably the wrong operator: at phi=0 it reported
    0.0174 (0.71 mm) against a true indexed error of 0.1373 (1.99 mm), a
    7.9x under-report, reaching only 1860 of 4143 clone vertices -- 55% of
    the target exerted no pull at all, and 99.5% of vertices matched to
    something other than their true correspondent. Stage A consequently
    barely deformed the mesh. Now uses the indexed correspondence when the
    topology matches (auto-detected), falling back to KNN when it doesn't. The rigid-loss weight is kept HIGH throughout (NOT
    decayed down to near-zero the way Stage A does in fit_with_arap_init.py)
    -- per explicit instruction: this stage should nudge phi into a good
    neighbourhood, not aggressively warp the mesh. The old fit is a very
    high-fidelity target (its own chamfer_best was often ~1e-5), so a
    gentle pull is enough to land close.

  Stage B (fit to the REAL target): standard Stage-1 alignment
    (fit_alignment, exactly like run_case.py) against this case's actual
    geometry, then continues optimizing the SAME phi/w/log_s/t (not
    reset) against the real target using ghd_fit.py's DEFAULT loss
    weights/schedule -- i.e. this stage is a faithful duplicate of
    ghd_fit.py's own default configuration, just starting from Stage A's
    state instead of zero.

Entirely separate from ghd_fit.py/run_case.py -- imports their loaders/
loss modules/utility functions, duplicates the training loops themselves.

conda activate new
python ghd/fitting/clone_fit.py --old-case-dir "/media/.../Fitting_Results_Final/ImperialNHS/<case>" \
    --case-dir "/path/to/this/case's/real/geometry"
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import trimesh
from pytorch3d.ops import sample_points_from_meshes, knn_points
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
from ghd.fitting.ghd_fit import (
    load_target, prepare_dvs_samples, render_fit_sanity, plot_loss_curves, _cosine_decay, _linear_decay,
    _canonical_opening_data,
)
from ghd.fitting.alignment import (
    load_case_data, load_canonical_data, fit_alignment,
    save_stage1_checkpoints, render_sanity_images, DOME_LOADERS,
)
from ghd.registration.clone_reframe import reframe_clone_to_target

ANEU_TYPE_TO_CANONICAL = {"bifurcation": 0, "sidewall": 1}


def clone_fit(old_case_dir, case_dir, canonical_root=None, out_dir=None, device="cuda",
              # Stage A (warm start toward the OLD fitted mesh)
              n_iter_stage_a=3000, lr_stage_a=1e-3, eta_min_stage_a=1e-4,
              lambda_chamfer_stage_a=1.0, mesh_health_scale_stage_a=1.0,
              lambda_rigid_stage_a=2.0, stage_a_rigid_end_frac=0.25,
              stage_a_rigid_decay_frac=0.8, lambda_laplacian_stage_a=1e-2,
              lambda_consistency_stage_a=0.3, lambda_edge_stage_a=0.1,
              lambda_node_mse=1.0, lambda_volume_stage_a=1.0, lambda_thickness_stage_a=2.0,
              lambda_occupancy_stage_a=1.0, lambda_opening_chamfer_stage_a=0.0,
              n_opening_chamfer_pts=3000,
              # Stage B (fit to the REAL target -- ghd_fit.py's own defaults)
              n_iter=10000, lr=1e-3, eta_min=1e-4,
              align_epochs=800, dome_source="nrrd",
              lambda_chamfer_n1=0.8, lambda_laplacian=1e-2,
              lambda_rigid_start=2.0, lambda_rigid_end=0.001, rigid_decay_frac=0.80,
              lambda_volume=1.0, volume_ceiling_ratio=1.2,
              volume_target_frac_start=1.0, volume_target_frac_end=1.0, volume_target_decay_frac=0.5,
              volume_ramp_frac=0.5, lambda_thickness=2.0, thickness_r=0.2,
              lambda_consistency_start=0.3, lambda_consistency_end=0.3, consistency_decay_frac=0.80,
              lambda_edge_start=0.1, lambda_edge_end=0.1, edge_decay_frac=0.80,
              lambda_occupancy=1.0, dvs_surf_d_min=0.0001, dvs_surf_d_max=0.05,
              n_surface_samples=20000, log_every=200, skip_stage_b=False,
              skip_stage_a=False, fit_target="real"):
    device = torch.device(device)
    old_case_dir = Path(old_case_dir)
    case_dir = Path(case_dir)
    canonical_root = Path(canonical_root) if canonical_root else ROOT / "dataset" / "canonical"
    case_name = old_case_dir.name
    dataset_name = old_case_dir.parent.name
    out_dir = Path(out_dir) if out_dir else ROOT / "runtime" / "fitting" / "cloned" / dataset_name / case_name
    sanity_dir = out_dir / "sanity"
    sanity_dir.mkdir(parents=True, exist_ok=True)
    sanity_dir_a = out_dir / "sanity_stage_a"
    sanity_dir_a.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "rigid_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # MESH HEALTH scale: one knob on the whole group (rigid, laplacian,
    # normal-consistency, edge, thickness), holding the node-MSE weight fixed.
    # Sweeping this rather than the MSE weight keeps the knob identical to the
    # one fit_with_arap_init.py exposes, so the two pipelines' MSE:mesh-health
    # ratios are directly comparable.
    if mesh_health_scale_stage_a != 1.0:
        lambda_rigid_stage_a *= mesh_health_scale_stage_a
        lambda_laplacian_stage_a *= mesh_health_scale_stage_a
        lambda_consistency_stage_a *= mesh_health_scale_stage_a
        lambda_edge_stage_a *= mesh_health_scale_stage_a
        lambda_thickness_stage_a *= mesh_health_scale_stage_a

    old_mesh = trimesh.load(old_case_dir / "ghd_fitted.obj", process=False)
    old_lm = np.load(old_case_dir / "landmarks.npz", allow_pickle=True)
    atype_str = str(old_lm["aneu_type"])
    atype = ANEU_TYPE_TO_CANONICAL[atype_str]
    print(f"=== Clone-fitting {dataset_name}/{case_name} (old aneu_type={atype_str} -> "
          f"canonical type {atype}) ===", flush=True)

    # NOTE: ghd_fitted.obj is in the OLD pipeline's own normalised space AND its
    # own rigid alignment frame -- both unrelated to this repo's. It is NOT
    # rescaled by the old s_can here: the reframing step below fits a full
    # SIMILARITY transform onto this case's real target, and that transform's
    # scale absorbs s_can (including the old pipeline's hand-embedded
    # *1.10*2.5). See ghd/registration/clone_reframe.py.
    old_metrics = json.loads((old_case_dir / "metrics.json").read_text())
    old_uncapped_path = old_case_dir / "ghd_fitted_uncapped.obj"
    old_uncapped = (trimesh.load(old_uncapped_path, process=False)
                    if old_uncapped_path.exists() else None)

    mc = MultiCanonicalGHDReconstruct(canonical_root, device=device)
    recon = mc.get(atype)
    norm_canonical = recon.norm_canonical
    V0 = recon.canonical_Meshes.verts_packed()
    F_can = recon.canonical_Meshes.faces_packed()
    U = recon.GHD_eigvec
    num_coeffs = U.shape[1]
    V0_faces_np = F_can.detach().cpu().numpy()

    edge_loss_fn = EdgeLengthLoss(recon.canonical_Meshes)
    rigid_loss_fn = GHDRigidLoss(V0, F_can)
    thickness_fn = MeshThickness(r=thickness_r)

    def render(phi, w, log_s, t):
        offset = torch.einsum('nm,mc->nc', U, phi)
        V_deformed = V0 + offset
        R = axis_angle_to_matrix(w.unsqueeze(0)).squeeze(0)
        V_rendered = (V_deformed @ R.T) * torch.exp(log_s) + t
        return Meshes(verts=[V_rendered], faces=[F_can])

    # ══════════════════════════════════════════════════════════════════════
    # Stage 1 (alignment) -- runs FIRST, because it DEFINES the frame. The
    # clone target must be moved into this frame before Stage A, otherwise
    # Stage A converges onto a pose in the old pipeline's frame and Stage B
    # then swaps in a target tens of degrees away while keeping that pose.
    # ══════════════════════════════════════════════════════════════════════
    print(f"=== Stage 1 (alignment): {case_dir.name} ===", flush=True)
    case = load_case_data(case_dir, dome_points_loader=DOME_LOADERS[dome_source])
    if case["aneurysm_type"] != atype and not (case["aneurysm_type"] in (1, 2) and atype in (1, 2)):
        raise ValueError(f"Real case's aneurysm_type={case['aneurysm_type']} doesn't match "
                         f"clone target's canonical type={atype}.")
    canonical_dir = canonical_root / ("Bifurcated" if atype == 0 else "Sidewall")
    canonical = load_canonical_data(canonical_dir)
    transform, align_result = fit_alignment(case, canonical, epochs=align_epochs, device=device,
                                            log_every=align_epochs)
    save_stage1_checkpoints(out_dir, transform, case, canonical)
    render_sanity_images(out_dir, transform, case, canonical)
    # save_stage1_checkpoints drops the similarity's SCALE from landmarks.npz
    # (legacy-format compatibility), which makes that frame non-invertible.
    # Persist the full (R, s, t) so any later analysis of this run can recover
    # the frame without needing a separate run_case.py run of the same case.
    np.save(out_dir / "alignment_result.npy",
            {"R": transform.rotation_matrix().detach().cpu().numpy(),
             "s": transform.scale().detach().cpu().item(),
             "t": transform.t.detach().cpu().numpy()}, allow_pickle=True)

    target = load_target(out_dir, norm_canonical, n_surface_samples=n_surface_samples)
    tgt_pts = torch.as_tensor(target["points"], device=device).unsqueeze(0)
    tgt_nrm = torch.as_tensor(target["normals"], device=device).unsqueeze(0)
    target_volume = target["volume_norm"]
    target_verts_phys = target["mesh_verts_norm"] * norm_canonical
    target_mesh_phys = trimesh.Trimesh(vertices=target_verts_phys,
                                       faces=target["mesh_faces"], process=False)

    # ══════════════════════════════════════════════════════════════════════
    # Reframe the clone into canonical space (surface chamfer, metadata-free).
    #
    # Two routes, because the two configs need different things from it:
    #
    #  fit_target="real" (the "clone" config): the reframed clone is only Stage
    #    A's warm-start target -- Stage B then continues the same phi/pose onto
    #    the REAL target, so a small placement error is a starting error that
    #    gets optimized away. Registers straight onto final_aligned.obj, which
    #    is already in canonical space.
    #
    #  fit_target="clone" (the "clone_target" config): the clone IS the fitting
    #    target, so the reframe residual is a permanent floor on accuracy --
    #    nothing downstream corrects it. Registering onto final_aligned.obj
    #    would be circular: that mesh is the output of the gradient-descent
    #    alignment, so a bad alignment would corrupt the clone's placement
    #    twice (once through the transform baked into final_aligned.obj, again
    #    through the chamfer chasing it). Instead register against the case's
    #    WORLD-space mesh and then apply the SAME transform the real target
    #    got. The clone then lands exactly where the target would, and the
    #    chamfer fit only has to absorb s_can rather than s_can * s_align.
    # ══════════════════════════════════════════════════════════════════════
    if fit_target == "clone":
        print("=== Reframing clone: world-space registration, then the case's "
              "own alignment transform ===", flush=True)
        _R, _s, _t, clone_world, reframe_info = reframe_clone_to_target(
            old_mesh, case["closed_mesh"], clone_uncapped=old_uncapped,
            target_branch_endpoints=case["branch_endpoints"], device=str(device))
        clone_world.export(out_dir / "clone_target_world.obj")
        # the world-space mesh the clone was registered ONTO -- exported so the
        # world-space leg of the reframe can be eyeballed on its own
        case["closed_mesh"].export(out_dir / "case_closed_world.obj")
        with torch.no_grad():
            verts_canon = transform(
                torch.as_tensor(np.asarray(clone_world.vertices), dtype=torch.float32,
                                device=device)).cpu().numpy()
        old_mesh = trimesh.Trimesh(vertices=verts_canon, faces=clone_world.faces, process=False)
    else:
        print("=== Reframing clone target into canonical space ===", flush=True)
        _R, _s, _t, old_mesh, reframe_info = reframe_clone_to_target(
            old_mesh, target_mesh_phys, clone_uncapped=old_uncapped,
            target_branch_endpoints=np.load(out_dir / "landmarks.npz",
                                            allow_pickle=True)["branch_endpoints"],
            device=str(device))
    old_mesh.export(out_dir / "clone_target.obj")
    np.savez(out_dir / "clone_reframe.npz", R=_R, s=_s, t=_t,
             **{k: v for k, v in reframe_info.items() if not isinstance(v, str)})
    if not reframe_info["ok"]:
        print(f"  WARNING: clone reframe residual {reframe_info['mean_surface_dist']:.4f} mm "
              f"({reframe_info['rel_error']*100:.2f}% of target RMS) exceeds the 3% gate -- "
              f"the warm start for this case may be unreliable.", flush=True)

    # ══════════════════════════════════════════════════════════════════════
    # Stage A: warm-start toward the OLD fitted mesh (clone target). Rigid
    # weight kept HIGH throughout -- see module docstring -- so this stage
    # nudges phi into a good neighbourhood without aggressively warping.
    # ══════════════════════════════════════════════════════════════════════
    # ── EXPERIMENTAL Stage-A opening chamfer (default off) ─────────────────
    # The old repo's canonical is byte-identical to this one (V=4143, F=8282,
    # same face array), so the clone mesh shares the canonical's vertex
    # indexing and canonical_topology.npy's cap-face mask applies to it
    # directly -- no cap detection needed on the clone side.
    # Pulls the canonical's openings onto the clone's openings during the warm
    # start, so branch ends are matched by the branches rather than being
    # absorbed by whatever surface happens to be nearest (the false-dome-stub
    # failure mode). Keep the weight SMALL -- this is a nudge, not a driver.
    opening_cap_face_idx = clone_cap_pts = None
    if lambda_opening_chamfer_stage_a > 0:
        _can_open = _canonical_opening_data(mc, atype, canonical_root, device)
        _cap_idx = np.where(_can_open["cap_face_mask"])[0]
        if _cap_idx.size == 0:
            print("  WARNING: canonical has no cap faces -- Stage-A opening chamfer disabled.", flush=True)
            lambda_opening_chamfer_stage_a = 0.0
        else:
            opening_cap_face_idx = torch.as_tensor(_cap_idx, dtype=torch.long, device=device)
            _clone_cap = trimesh.Trimesh(vertices=np.asarray(old_mesh.vertices),
                                         faces=np.asarray(old_mesh.faces)[_cap_idx], process=False)
            _pts, _ = trimesh.sample.sample_surface(_clone_cap, n_opening_chamfer_pts)
            clone_cap_pts = torch.as_tensor(np.asarray(_pts, dtype=np.float32) / norm_canonical,
                                            device=device).unsqueeze(0)
            print(f"  Stage-A opening chamfer ON (lambda={lambda_opening_chamfer_stage_a}, "
                  f"{_cap_idx.size} cap faces shared with the canonical)", flush=True)

    if not skip_stage_a:
        print(f"=== Stage A (warm start toward clone target, {n_iter_stage_a} iters) ===", flush=True)
    clone_target_verts_norm_np = np.asarray(old_mesh.vertices, dtype=np.float32) / norm_canonical
    clone_target_faces_np = np.asarray(old_mesh.faces)
    clone_target_volume_norm = float(old_mesh.volume) / (norm_canonical ** 3)
    ctgt_pts_raw, ctgt_face_idx = trimesh.sample.sample_surface(old_mesh, n_surface_samples)
    ctgt_nrm_raw = old_mesh.face_normals[ctgt_face_idx]
    ctgt_pts = torch.as_tensor(np.asarray(ctgt_pts_raw, dtype=np.float32) / norm_canonical,
                               device=device).unsqueeze(0)
    ctgt_nrm = torch.as_tensor(np.asarray(ctgt_nrm_raw, dtype=np.float32), device=device).unsqueeze(0)
    clone_target_verts_norm_t = torch.as_tensor(clone_target_verts_norm_np, device=device).unsqueeze(0)
    # Exact indexed correspondence when the clone shares the canonical's
    # topology (it does -- same repo canonical), else fall back to KNN.
    node_mse_indexed = (clone_target_verts_norm_np.shape[0] == V0.shape[0]
                        and clone_target_faces_np.shape == V0_faces_np.shape
                        and bool((clone_target_faces_np == V0_faces_np).all()))
    print(f"  Stage-A node MSE: {'INDEXED (exact correspondence)' if node_mse_indexed else 'KNN (topology differs)'}",
          flush=True)
    clone_target_verts_phys = clone_target_verts_norm_np * norm_canonical

    clone_target_dict = {"mesh_verts_norm": clone_target_verts_norm_np, "mesh_faces": clone_target_faces_np}
    dvs_samples_a = None
    if lambda_occupancy_stage_a > 0:
        dvs_samples_a = prepare_dvs_samples(clone_target_dict, device, d_min=dvs_surf_d_min, d_max=dvs_surf_d_max)
        if dvs_samples_a is None:
            print("  WARNING: could not close clone target for occupancy -- disabling for Stage A.", flush=True)
            lambda_occupancy_stage_a = 0.0
        else:
            dvs_samples_a = tuple(t_[:4000] for t_ in dvs_samples_a)
    occupancy_fn_a = DVSOccupancyLoss(num_sample=4000) if lambda_occupancy_stage_a > 0 else None
    volume_loss_fn_a = VolumeLoss(target_volume=clone_target_volume_norm, ceiling_ratio=volume_ceiling_ratio)

    V0_volume = float(trimesh.Trimesh(vertices=V0.detach().cpu().numpy(), faces=V0_faces_np, process=False).volume)
    size_match = (clone_target_volume_norm / max(V0_volume, 1e-12)) ** (1.0 / 3.0)

    phi = nn.Parameter(torch.zeros(num_coeffs, 3, device=device))
    w = nn.Parameter(torch.zeros(3, device=device))
    log_s = nn.Parameter(torch.tensor(float(np.log(max(size_match, 1e-6))), device=device))
    t = nn.Parameter(torch.zeros(3, device=device))

    if skip_stage_a:
        print("=== Stage A SKIPPED (Stage B starts from the native rest pose) ===", flush=True)

    opt_a = torch.optim.Adam([phi, w, log_s, t], lr=lr_stage_a)
    sched_a = torch.optim.lr_scheduler.CosineAnnealingLR(opt_a, T_max=n_iter_stage_a, eta_min=eta_min_stage_a)
    history_a = {k: [] for k in ["chamfer", "node_mse", "rigid", "laplacian", "opening_chamfer",
                                 "mesh_health", "lambda_rigid_w", "total", "lr"]}

    # Stage-A rigid schedule: decay from lambda_rigid_stage_a down to
    # stage_a_rigid_end_frac of it (default 25%) over stage_a_rigid_decay_frac
    # of the run. Holding it high the whole way keeps the mesh near its rest
    # pose and starves the data terms late in Stage A -- especially once
    # chamfer/occupancy are off and node-MSE is the only thing pulling.
    lam_rigid_a_start = lambda_rigid_stage_a
    lam_rigid_a_end = lambda_rigid_stage_a * stage_a_rigid_end_frac
    print(f"  Stage-A rigid schedule: {lam_rigid_a_start:.4f} -> {lam_rigid_a_end:.4f} "
          f"over {stage_a_rigid_decay_frac:.0%} of {n_iter_stage_a} iters", flush=True)

    for it in range(n_iter_stage_a if skip_stage_a else 0, n_iter_stage_a):
        frac_a = it / max(n_iter_stage_a - 1, 1)
        lam_rigid_a = _cosine_decay(lam_rigid_a_start, lam_rigid_a_end, frac_a, stage_a_rigid_decay_frac)
        mesh = render(phi, w, log_s, t)
        src_pts, src_nrm = sample_points_from_meshes(mesh, n_surface_samples, return_normals=True)
        loss_chamfer, loss_chamfer_n1 = chamfer_loss(src_pts, ctgt_pts, src_nrm, ctgt_nrm)
        loss_laplacian = laplacian_loss(mesh)
        loss_consistency = normal_consistency_loss(mesh)
        loss_edge = edge_loss_fn(mesh)
        loss_rigid = rigid_loss_fn(mesh.verts_packed())
        loss_volume = volume_loss_fn_a(mesh, target_frac=1.0)
        dist, dist_v_norm, _, sign = thickness_fn(mesh)
        mask = (dist_v_norm.abs() > 0.1).logical_not().float()
        signed = torch.sign(sign)
        loss_thickness = (torch.relu(0.04 - dist_v_norm * signed) + torch.relu(0.01 - dist * signed)) * mask
        loss_thickness = loss_thickness.mean() + (1e-4 / (sign ** 2 + 1e-6) * mask).mean()
        loss_occupancy = torch.zeros((), device=device)
        if occupancy_fn_a is not None:
            pos, neg, wp2n, wn2p = dvs_samples_a
            loss_occupancy = occupancy_fn_a(mesh, pos, neg, wp2n, wn2p)
        if node_mse_indexed:
            loss_node_mse = ((mesh.verts_packed() - clone_target_verts_norm_t[0]) ** 2).sum(-1).mean()
        else:
            knn = knn_points(mesh.verts_packed().unsqueeze(0), clone_target_verts_norm_t, K=1)
            loss_node_mse = knn.dists[..., 0].mean()

        loss_opening_chamfer_a = torch.zeros((), device=device)
        if lambda_opening_chamfer_stage_a > 0:
            can_cap_mesh = Meshes(verts=[mesh.verts_packed()], faces=[F_can[opening_cap_face_idx]])
            can_cap_pts = sample_points_from_meshes(can_cap_mesh, n_opening_chamfer_pts)
            loss_opening_chamfer_a, _ = chamfer_loss(can_cap_pts, clone_cap_pts)

        # MESH HEALTH group: rigid + laplacian + normal-consistency + edge +
        # thickness. These constrain the mesh's own quality rather than its
        # agreement with the target, so they stay on in every Stage-A variant.
        loss_mesh_health = (lambda_laplacian_stage_a * loss_laplacian
                           + lambda_consistency_stage_a * loss_consistency
                           + lambda_edge_stage_a * loss_edge
                           + lam_rigid_a * loss_rigid
                           + lambda_thickness_stage_a * loss_thickness)
        # DATA group: what pulls the mesh onto the clone target. chamfer and
        # occupancy are individually switchable (lambda -> 0) so Stage A can be
        # run on node-MSE + opening alone.
        loss_data = (lambda_chamfer_stage_a * (loss_chamfer + lambda_chamfer_n1 * loss_chamfer_n1)
                    + lambda_volume_stage_a * loss_volume
                    + lambda_occupancy_stage_a * loss_occupancy
                    + lambda_node_mse * loss_node_mse
                    + lambda_opening_chamfer_stage_a * loss_opening_chamfer_a)
        total_a = loss_data + loss_mesh_health

        opt_a.zero_grad()
        total_a.backward()
        opt_a.step()
        sched_a.step()

        history_a["chamfer"].append(loss_chamfer.item())
        history_a["node_mse"].append(loss_node_mse.item())
        history_a["rigid"].append(loss_rigid.item())
        history_a["laplacian"].append(loss_laplacian.item())
        history_a["opening_chamfer"].append(loss_opening_chamfer_a.item())
        history_a["mesh_health"].append(loss_mesh_health.item())
        history_a["lambda_rigid_w"].append(lam_rigid_a)
        history_a["total"].append(total_a.item())
        history_a["lr"].append(opt_a.param_groups[0]["lr"])

        if it % log_every == 0 or it == n_iter_stage_a - 1:
            print(f"  [A {it:5d}/{n_iter_stage_a}] chamfer={loss_chamfer.item():.6f} "
                  f"node_mse={loss_node_mse.item():.6f} rigid={loss_rigid.item():.4f} "
                  f"(w={lam_rigid_a:.3f}) "
                  f"opench={loss_opening_chamfer_a.item():.6f} "
                  f"health={loss_mesh_health.item():.5f} total={total_a.item():.5f}", flush=True)
            with torch.no_grad():
                snap_verts_phys = (mesh.verts_packed() * norm_canonical).cpu().numpy()
            render_fit_sanity(sanity_dir_a / f"iter_{it:05d}.png", snap_verts_phys,
                              V0_faces_np, clone_target_verts_phys)

    with torch.no_grad():
        stage_a_verts_phys = (render(phi, w, log_s, t).verts_packed() * norm_canonical).cpu().numpy()
    if not skip_stage_a:
        np.savez(out_dir / "stage_a_loss_history.npz", **{k: np.asarray(v) for k, v in history_a.items()})
        plot_loss_curves(history_a, sanity_dir_a / "stage_a_loss_curves.png")
        render_fit_sanity(sanity_dir_a / "final_vs_clone_target.png", stage_a_verts_phys,
                          V0_faces_np, clone_target_verts_phys)
        print(f"  Stage A done -> {sanity_dir_a / 'final_vs_clone_target.png'}", flush=True)

    if skip_stage_b:
        # Stage-A-only mode: export Stage A's mesh as the result and stop.
        # Used when only the warm start needs eyeballing (e.g. weight sweeps),
        # so 7500 Stage-B iterations aren't spent to answer a Stage-A question.
        with torch.no_grad():
            mesh_a = render(phi, w, log_s, t)
            chamfer_vs_real, _ = chamfer_loss(
                sample_points_from_meshes(mesh_a, n_surface_samples), tgt_pts)
            node_mse_final = (((mesh_a.verts_packed() - clone_target_verts_norm_t[0]) ** 2).sum(-1).mean()
                              if node_mse_indexed else torch.zeros(()))
        trimesh.Trimesh(vertices=stage_a_verts_phys, faces=V0_faces_np,
                        process=False).export(out_dir / "ghd_fitted.obj")
        render_fit_sanity(sanity_dir / "sanity_final.png", stage_a_verts_phys,
                          V0_faces_np, target_verts_phys)
        np.savez(out_dir / "ghd_coefficients.npz",
                 phi=phi.detach().cpu().numpy(), w_rot=w.detach().cpu().numpy(),
                 log_scale=np.array(log_s.detach().cpu().item()), t_vec=t.detach().cpu().numpy())
        metrics = {
            "stage_a_only": True,
            "chamfer_best": float(chamfer_vs_real.item()),
            "chamfer_final": float(chamfer_vs_real.item()),
            "best_iter": n_iter_stage_a,
            "stage_a_node_mse": float(node_mse_final.item()),
            "stage_a_chamfer_vs_clone": history_a["chamfer"][-1],
            "stage_a_rigid": history_a["rigid"][-1],
            "stage_a_opening_chamfer": history_a["opening_chamfer"][-1],
            "stage_a_mesh_health": history_a["mesh_health"][-1],
            "aneurysm_type": atype, "s_can": float(norm_canonical),
            "case_dir": str(case_dir),
            "old_case": f"{dataset_name}/{case_name}", "old_aneu_type": atype_str,
            "reframe_mean_surface_dist": reframe_info["mean_surface_dist"],
            "reframe_rel_error": reframe_info["rel_error"],
            "lambda_rigid_stage_a": float(lambda_rigid_stage_a),
            "stage_a_rigid_end_frac": float(stage_a_rigid_end_frac),
            "mesh_health_scale_stage_a": float(mesh_health_scale_stage_a),
            "lambda_node_mse": float(lambda_node_mse),
            "lambda_opening_chamfer_stage_a": float(lambda_opening_chamfer_stage_a),
            "lambda_chamfer_stage_a": float(lambda_chamfer_stage_a),
            "lambda_occupancy_stage_a": float(lambda_occupancy_stage_a),
            "node_mse_mode": "indexed" if node_mse_indexed else "knn",
            "skip_stage_a": bool(skip_stage_a), "fit_target": fit_target,
        }
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Stage-A only. node_mse={node_mse_final.item():.6f} "
              f"chamfer_vs_real={chamfer_vs_real.item():.5f} -> {out_dir}", flush=True)
        return metrics

    if fit_target == "clone":
        # Fit the REFRAMED CLONE MESH instead of the real target. The old
        # project's fitted mesh is clean and watertight; the real clipped
        # geometry is not. This re-derives phi in the current basis against
        # the better-conditioned surface. The clone is already in canonical
        # space (reframed above), so only what Stage B's chamfer/volume/
        # occupancy terms point at changes.
        print("=== Stage B target: REFRAMED CLONE MESH (not the real target) ===", flush=True)
        target = clone_target_dict
        tgt_pts, tgt_nrm = ctgt_pts, ctgt_nrm
        target_volume = clone_target_volume_norm
        target_verts_phys = clone_target_verts_phys

    # ══════════════════════════════════════════════════════════════════════
    # Stage B: continue optimizing the SAME phi/w/log_s/t toward the REAL
    # target using ghd_fit.py's own default loss weights/schedule. The pose
    # carried over from Stage A is already in this frame, because Stage A's
    # clone target was reframed into it before Stage A ran.
    # ══════════════════════════════════════════════════════════════════════
    _tgt_name = "reframed clone" if fit_target == "clone" else "real target"
    print(f"=== Stage B (fit to {_tgt_name}, default setup, {n_iter} iters) ===", flush=True)
    dvs_samples = None
    if lambda_occupancy > 0:
        dvs_samples = prepare_dvs_samples(target, device, d_min=dvs_surf_d_min, d_max=dvs_surf_d_max)
        if dvs_samples is None:
            print("  WARNING: could not close target mesh for occupancy -- disabling.", flush=True)
            lambda_occupancy = 0.0
        else:
            dvs_samples = tuple(t_[:4000] for t_ in dvs_samples)

    optimizer = torch.optim.Adam([phi, w, log_s, t], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iter, eta_min=eta_min)
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
        trimesh.Trimesh(vertices=verts_phys_, faces=V0_faces_np, process=False).export(ckpt_dir / f"ghd_fitted_{tag}.obj")
        np.savez(ckpt_dir / f"ghd_coefficients_{tag}.npz",
                phi=phi.detach().cpu().numpy(), w_rot=w.detach().cpu().numpy(),
                log_scale=np.array(log_s.detach().cpu().item()), t_vec=t.detach().cpu().numpy(),
                iteration=np.array(iteration), lambda_rigid_w=np.array(rigid_weight), rigid_loss=np.array(rigid_value))
        render_fit_sanity(sanity_dir / f"sanity_{tag}.png", verts_phys_, V0_faces_np, target_verts_phys)
        print(f"  rigid-checkpoint saved: lambda_rigid_w<={rigid_weight:.3f} @ iter {iteration} "
              f"(rigid_loss={rigid_value:.4f}) -> {tag}", flush=True)

    history = {k: [] for k in [
        "chamfer", "chamfer_n1", "laplacian", "consistency", "edge", "rigid",
        "volume", "thickness", "occupancy", "total", "lambda_rigid_w",
        "volume_target_frac", "lambda_volume_w", "lambda_consistency_w", "lambda_edge_w", "lr",
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

        mesh = render(phi, w, log_s, t)
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

        if it % log_every == 0 or it == n_iter - 1:
            print(f"  [B {it:6d}/{n_iter}] chamfer={cur_chamfer:.5f} rigid={loss_rigid.item():.4f} "
                  f"volume={loss_volume.item():.4f} (w={lam_volume:.3f}) thickness={loss_thickness.item():.4f} "
                  f"edge={loss_edge.item():.4f} consistency={loss_consistency.item():.4f} "
                  f"occ={loss_occupancy.item():.4f} total={total.item():.4f}", flush=True)

    phi.data.copy_(best_state["phi"]); w.data.copy_(best_state["w"])
    log_s.data.copy_(best_state["log_s"]); t.data.copy_(best_state["t"])
    with torch.no_grad():
        final_mesh = render(phi, w, log_s, t)
        final_pts = sample_points_from_meshes(final_mesh, n_surface_samples)
        chamfer_final, _ = chamfer_loss(final_pts, tgt_pts)

    verts_phys = (final_mesh.verts_packed() * norm_canonical).detach().cpu().numpy()
    trimesh.Trimesh(vertices=verts_phys, faces=V0_faces_np, process=False).export(out_dir / "ghd_fitted.obj")
    render_fit_sanity(sanity_dir / "sanity_final.png", verts_phys, V0_faces_np, target_verts_phys)

    np.savez(out_dir / "ghd_coefficients.npz",
             phi=best_state["phi"].cpu().numpy(), w_rot=best_state["w"].cpu().numpy(),
             log_scale=np.array(best_state["log_s"].cpu().item()), t_vec=best_state["t"].cpu().numpy())

    metrics = {
        "chamfer_final": float(chamfer_final.item()), "chamfer_best": best_chamfer,
        "best_iter": best_state["iter"], "aneurysm_type": atype, "s_can": float(norm_canonical),
        "case_dir": str(case_dir),
        "old_case": f"{dataset_name}/{case_name}", "old_aneu_type": atype_str,
        "reframe_mean_surface_dist": reframe_info["mean_surface_dist"],
        "reframe_rel_error": reframe_info["rel_error"],
        "reframe_scale": reframe_info["scale"],
        "reframe_route": "world+transform" if fit_target == "clone" else "canonical",
        "skip_stage_a": bool(skip_stage_a), "fit_target": fit_target,
        "reframe_init": reframe_info["init"],
        "reframe_ok": reframe_info["ok"],
        "lambda_rigid_stage_a": float(lambda_rigid_stage_a),
        "lambda_opening_chamfer_stage_a": float(lambda_opening_chamfer_stage_a),
        "lambda_chamfer_stage_a": float(lambda_chamfer_stage_a),
        "lambda_node_mse": float(lambda_node_mse),
        "node_mse_mode": "indexed" if node_mse_indexed else "knn",
        "lambda_occupancy_stage_a": float(lambda_occupancy_stage_a),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    np.savez(out_dir / "loss_history.npz", **{k: np.asarray(v) for k, v in history.items()})
    plot_loss_curves(history, out_dir / "loss_curves.png")

    print(f"Done. chamfer_best={best_chamfer:.5f} @ iter {best_state['iter']} -> {out_dir}", flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--old-case-dir", required=True,
                        help="e.g. '/media/.../Fitting_Results_Final/ImperialNHS/<case_name>'")
    parser.add_argument("--case-dir", required=True,
                        help="This case's REAL geometry directory (for Stage B's actual target).")
    parser.add_argument("--canonical-root", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dome-source", choices=list(DOME_LOADERS), default="nrrd")
    parser.add_argument("--align-epochs", type=int, default=800)
    parser.add_argument("--n-iter-stage-a", type=int, default=3000)
    parser.add_argument("--lr-stage-a", type=float, default=1e-3)
    parser.add_argument("--lambda-rigid-stage-a", type=float, default=2.0,
                        help="Kept HIGH (not decayed) throughout Stage A -- don't want the mesh "
                             "to warp too much toward the clone target, just nudge phi into a "
                             "good neighbourhood.")
    parser.add_argument("--lambda-node-mse", type=float, default=1.0)
    parser.add_argument("--mesh-health-scale-stage-a", type=float, default=1.0,
                        help="Scales the WHOLE Stage-A mesh-health group (rigid, laplacian, "
                             "consistency, edge, thickness) while the node-MSE weight stays fixed. "
                             "Base rigid is 2.0, so scale 0.025 gives ARAP's validated 20:1 "
                             "MSE:rigid ratio.")
    parser.add_argument("--stage-a-rigid-end-frac", type=float, default=0.25,
                        help="Stage-A rigid weight decays to this FRACTION of its initial value.")
    parser.add_argument("--stage-a-rigid-decay-frac", type=float, default=0.8,
                        help="Fraction of Stage A over which the rigid decay completes.")
    parser.add_argument("--lambda-chamfer-stage-a", type=float, default=1.0,
                        help="Scales BOTH Stage-A chamfer terms (point + normal). 0 disables "
                             "surface chamfer in Stage A, leaving node-MSE/opening/volume as the "
                             "only data terms. Chamfer is still COMPUTED and logged either way.")
    parser.add_argument("--lambda-occupancy-stage-a", type=float, default=1.0,
                        help="Stage-A DVS occupancy weight. 0 disables it.")
    parser.add_argument("--lambda-opening-chamfer-stage-a", type=float, default=0.0,
                        help="EXPERIMENTAL, default off. Pulls the canonical's opening caps onto "
                             "the clone's during Stage A. Keep SMALL (~0.1) -- a nudge, not a driver.")
    parser.add_argument("--n-opening-chamfer-pts", type=int, default=3000)
    parser.add_argument("--n-iter", type=int, default=10000, help="Stage B iterations.")
    parser.add_argument("--skip-stage-a", action="store_true",
                        help="Skip the warm start; Stage B begins at the native rest pose.")
    parser.add_argument("--fit-target", choices=["real", "clone"], default="real",
                        help="What Stage B fits. 'real' = this case's own geometry (default). "
                             "'clone' = the REFRAMED clone mesh, i.e. re-derive phi against the "
                             "old project's clean fitted surface instead of the noisy real clip.")
    parser.add_argument("--skip-stage-b", action="store_true",
                        help="Stop after Stage A and export its mesh as the result. For answering Stage-A questions (weight sweeps) without paying for Stage B.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Stage B base learning rate.")
    args = parser.parse_args()
    clone_fit(args.old_case_dir, args.case_dir, args.canonical_root, args.out_dir, args.device,
              n_iter_stage_a=args.n_iter_stage_a, lr_stage_a=args.lr_stage_a,
              lambda_rigid_stage_a=args.lambda_rigid_stage_a, lambda_node_mse=args.lambda_node_mse,
              lambda_chamfer_stage_a=args.lambda_chamfer_stage_a,
              mesh_health_scale_stage_a=args.mesh_health_scale_stage_a,
              stage_a_rigid_end_frac=args.stage_a_rigid_end_frac,
              stage_a_rigid_decay_frac=args.stage_a_rigid_decay_frac,
              lambda_occupancy_stage_a=args.lambda_occupancy_stage_a,
              lambda_opening_chamfer_stage_a=args.lambda_opening_chamfer_stage_a,
              n_opening_chamfer_pts=args.n_opening_chamfer_pts,
              n_iter=args.n_iter, lr=args.lr, align_epochs=args.align_epochs, dome_source=args.dome_source, skip_stage_b=args.skip_stage_b,
              skip_stage_a=args.skip_stage_a, fit_target=args.fit_target)


if __name__ == "__main__":
    main()
