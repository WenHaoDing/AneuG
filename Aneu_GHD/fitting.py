"""
GHD Fitting pipeline for aneurysm meshes.

Public API
----------
FitConfig   — dataclass of all hyperparameters and loss weights
FitResult   — dataclass returned by ghd_fit()
ghd_fit()   — two-stage alignment + GHD optimisation loop
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.optim as optim
import trimesh
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import knn_points, sample_points_from_meshes
from pytorch3d.structures import Meshes

from .losses import (
    DVSOccupancyLoss,
    EdgeLengthLoss,
    GHDRigidLoss,
    LandmarkLoss,
    laplacian_loss,
    normal_consistency_loss,
    winding_occupancy,
)


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class FitConfig:
    """
    All hyperparameters for one GHD fitting run.

    Instantiate with defaults and override only what you need:
        cfg = FitConfig(n_iter=8000, lambda_landmark=0.0)
    """
    # ── Optimiser ─────────────────────────────────────────────────────────────
    n_iter:        int   = 12_000
    lr:            float = 1e-3
    eta_min:       float = 1e-6   # cosine annealing floor

    # ── Affine DOF ────────────────────────────────────────────────────────────
    fit_R: bool = True
    fit_s: bool = True
    fit_T: bool = True

    # ── Point-cloud sampling ──────────────────────────────────────────────────
    num_samples:     int = 5_000   # fixed target samples for chamfer
    n_dvs:           int = 10_000  # DVS samples per class (pos / neg)
    n_dvs_oversamp:  int = 60_000  # oversample before filtering
    num_dvs_sample:  int = 12_000  # per-iteration DVS draw
    NP_ratio:        float = 1.0

    # ── Loss weights ──────────────────────────────────────────────────────────
    lambda_chamfer:      float = 1.0
    lambda_chamfer_n1:   float = 0.5
    lambda_occupancy:    float = 1.0
    lambda_laplacian:    float = 1e-2
    lambda_consistency:  float = 0.3
    lambda_rigid_start:  float = 3.0
    lambda_rigid_end:    float = 0.01
    rigid_decay_frac:    float = 0.7   # decay over first N% of iterations
    lambda_edge:         float = 0.1
    lambda_landmark:     float = 0.1
    lm_taper_frac:       float = 0.3   # taper landmark loss to 0 over first N%

    # ── Logging / checkpointing ───────────────────────────────────────────────
    log_every:         int  = 200
    save_every:        int  = 1000    # save intermediate .obj (0 = disabled)
    out_dir:           str  = "ghd_debug_outputs"

    # ── Evaluation ────────────────────────────────────────────────────────────
    converge_threshold: float = 0.95   # converged if chamfer_final < init * thr
    device:            str   = "cuda"  # "cuda" or "cpu"


# ── Results ───────────────────────────────────────────────────────────────────

@dataclass
class FitResult:
    """
    Output of ghd_fit().  All mesh data is in normalised space.

    Metrics
    -------
    chamfer_final  : final chamfer distance (surface accuracy, lower = better)
    chamfer_best   : best chamfer seen during fitting
    best_iter      : iteration that achieved chamfer_best
    dice           : volumetric Dice score vs target (0–1, higher = better)
    gar            : Good Angle Ratio — fraction of triangles with all
                     angles in [30°, 120°]  (0–1, higher = better)
    converged      : chamfer_final < chamfer_init * FitConfig.converge_threshold
    """
    verts:          np.ndarray          # (N, 3) fitted vertices, normalised space
    loss_history:   dict                # {loss_name: [float]}
    chamfer_final:  float
    chamfer_best:   float
    best_iter:      int
    dice:           float
    gar:            float
    converged:      bool


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_device(cfg: FitConfig) -> torch.device:
    if cfg.device.startswith("cuda") and torch.cuda.is_available():
        return torch.device(cfg.device)   # respects "cuda", "cuda:0", "cuda:2", etc.
    return torch.device("cpu")


def _so3_exp_map(w: torch.Tensor) -> torch.Tensor:
    """Rodrigues exponential map: axis-angle vector → SO(3) matrix."""
    theta = torch.norm(w) + 1e-12
    k = w / theta
    K = torch.stack([
        torch.stack([torch.tensor(0., device=w.device),  -k[2],  k[1]]),
        torch.stack([ k[2],  torch.tensor(0., device=w.device), -k[0]]),
        torch.stack([-k[1],   k[0], torch.tensor(0., device=w.device)]),
    ])
    I = torch.eye(3, device=w.device, dtype=w.dtype)
    return I + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)


def _ghd_deform(
    V0:    torch.Tensor,   # (N, 3)
    U_t:   torch.Tensor,   # (N, p)
    phi:   torch.Tensor,   # (p, 3)
) -> torch.Tensor:
    """Linear GHD deformation: V(phi) = V0 + U @ phi."""
    return V0 + (U_t @ phi)


def _render_ghd(
    V0:      torch.Tensor,
    U_t:     torch.Tensor,
    phi:     torch.Tensor,
    w_aff:   torch.Tensor,
    log_s:   torch.Tensor,
    t_vec:   torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Non-rigid GHD deformation + affine transform (rotation, scale, translation)."""
    V_nonrigid = _ghd_deform(V0, U_t, phi)
    R = _so3_exp_map(w_aff)
    s = torch.exp(log_s)
    V_rendered = (V_nonrigid @ R.T) * s + t_vec
    return V_nonrigid, V_rendered


def _good_angle_ratio(verts: np.ndarray, faces: np.ndarray) -> float:
    """
    Fraction of triangles with all angles in [30°, 120°] (vectorised).
    GHD paper Eq. (18): measures mesh quality preservation.
    """
    fc = verts[faces]                          # (F, 3, 3)
    v0 = fc[:, 0]; v1 = fc[:, 1]; v2 = fc[:, 2]

    def _angle(a, b, c):
        u = b - a;  v = c - a
        cos = (u * v).sum(-1) / (
            np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-12
        )
        return np.degrees(np.arccos(np.clip(cos, -1, 1)))

    a0 = _angle(v0, v1, v2)
    a1 = _angle(v1, v2, v0)
    a2 = _angle(v2, v0, v1)

    good = (
        (a0 >= 30) & (a0 <= 120) &
        (a1 >= 30) & (a1 <= 120) &
        (a2 >= 30) & (a2 <= 120)
    )
    return float(good.mean())


def _compute_dice(
    fitted_mesh: Meshes,
    pos_pts:     torch.Tensor,
    neg_pts:     torch.Tensor,
) -> float:
    """Volumetric Dice between fitted mesh and DVS ground-truth occupancy."""
    pts    = torch.cat([pos_pts, neg_pts], dim=0)
    gt     = torch.cat([
        torch.ones(pos_pts.shape[0],  device=pos_pts.device),
        torch.zeros(neg_pts.shape[0], device=neg_pts.device),
    ])
    with torch.no_grad():
        occ   = winding_occupancy(fitted_mesh, pts)
        pred  = (occ > 0.5).float()
    inter = (pred * gt).sum()
    dice  = (2 * inter / (pred.sum() + gt.sum() + 1e-6)).item()
    return float(dice)


# ── DVS sample preparation ────────────────────────────────────────────────────

def prepare_dvs_samples(
    V_tgt: np.ndarray,
    F_tgt: np.ndarray,
    cfg:   FitConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample interior and exterior points from the target mesh for DVS loss.

    Uses trimesh.contains() (ray casting) to classify random bounding-box
    points as inside or outside, then computes distance-based weights.

    Returns
    -------
    target_positives : (n_dvs, 3)  interior points
    target_negatives : (n_dvs, 3)  exterior points
    dist_weights_p2n : (n_dvs,)    distance weights for positives
    dist_weights_n2p : (n_dvs,)    distance weights for negatives
    """
    tgt_trimesh = trimesh.Trimesh(vertices=V_tgt, faces=F_tgt, process=False)

    rng      = np.random.default_rng(42)
    bbox_min = V_tgt.min(axis=0) - 0.05
    bbox_max = V_tgt.max(axis=0) + 0.05
    rand_pts = bbox_min + rng.random((cfg.n_dvs_oversamp, 3)) * (bbox_max - bbox_min)

    _t0 = time.time()
    inside_mask = tgt_trimesh.contains(rand_pts)
    print(f"  trimesh.contains: {time.time() - _t0:.1f}s")
    pos_pts     = rand_pts[inside_mask]
    neg_pts     = rand_pts[~inside_mask]

    print(f"  DVS: {len(pos_pts)} interior pts, {len(neg_pts)} exterior pts")
    if len(pos_pts) < cfg.n_dvs or len(neg_pts) < cfg.n_dvs:
        raise RuntimeError(
            f"Not enough DVS samples: need {cfg.n_dvs} per class, "
            f"got {len(pos_pts)} pos / {len(neg_pts)} neg. "
            "Increase n_dvs_oversamp in FitConfig."
        )

    target_positives = torch.tensor(pos_pts[:cfg.n_dvs], dtype=torch.float32, device=device)
    target_negatives = torch.tensor(neg_pts[:cfg.n_dvs], dtype=torch.float32, device=device)

    with torch.no_grad():
        dist_p2n  = knn_points(
            target_positives.view(1, -1, 3),
            target_negatives.view(1, -1, 3), K=1,
        )[0].view(-1)
        dist_n2p  = knn_points(
            target_negatives.view(1, -1, 3),
            target_positives.view(1, -1, 3), K=1,
        )[0].view(-1)
        dist_mean = torch.cat([dist_p2n, dist_n2p]).mean()
        dist_weights_p2n = 1 - torch.exp(-dist_p2n ** 2 / (dist_mean ** 2 + 1e-6))
        dist_weights_p2n = dist_weights_p2n / dist_weights_p2n.mean()
        dist_weights_n2p = 1 - torch.exp(-dist_n2p ** 2 / (dist_mean ** 2 + 1e-6))
        dist_weights_n2p = dist_weights_n2p / dist_weights_n2p.mean()

    return target_positives, target_negatives, dist_weights_p2n, dist_weights_n2p


# ── Main fitting function ─────────────────────────────────────────────────────

def ghd_fit(
    V_init:      np.ndarray,               # (N, 3) CPD-aligned canonical (float32)
    F:           np.ndarray,               # (M, 3) canonical faces (int64)
    V_tgt:       np.ndarray,               # (K, 3) normalised target vertices
    F_tgt:       np.ndarray,               # (Mt, 3) target faces
    U:           np.ndarray,               # (N, p) pre-computed eigenvector basis
    cfg:         FitConfig | None = None,
    lm_can_s:    dict | None = None,       # normalised canonical landmarks
    lm_tgt_s:    dict | None = None,       # normalised target    landmarks
    idx_lm_dome: int | None  = None,       # canonical vertex index for dome
    cap_up_idxs: np.ndarray | None = None, # k canonical indices for upstream cap
    cap_dn_idxs: np.ndarray | None = None, # k canonical indices for downstream cap
) -> FitResult:
    """
    Run GHD fitting and return a FitResult with metrics and fitted vertices.

    Parameters
    ----------
    V_init      : (N, 3) anatomy+CPD aligned canonical vertices (normalised space)
    F           : (M, 3) canonical face indices
    V_tgt       : (K, 3) target vertices (normalised space)
    F_tgt       : (Mt, 3) target face indices
    U           : (N, p) Laplacian eigenvector basis of the canonical mesh.
                  Pre-compute once; reuse for every target case.
    cfg         : FitConfig (uses defaults if None)
    lm_can_s    : normalised canonical landmarks — required for landmark loss
    lm_tgt_s    : normalised target landmarks    — required for landmark loss
    idx_lm_dome : dome vertex index  ─┐ required if cfg.lambda_landmark > 0
    cap_up_idxs : cap_up indices     ─┤
    cap_dn_idxs : cap_dn indices     ─┘

    Returns
    -------
    FitResult
    """
    if cfg is None:
        cfg = FitConfig()

    device = _get_device(cfg)
    use_lm = (
        cfg.lambda_landmark > 0
        and idx_lm_dome  is not None
        and cap_up_idxs  is not None
        and cap_dn_idxs  is not None
        and lm_tgt_s     is not None
    )

    # ── Tensors ───────────────────────────────────────────────────────────────
    V0   = torch.tensor(V_init, dtype=torch.float32, device=device)
    F0   = torch.tensor(F,      dtype=torch.int64,   device=device)
    Vt   = torch.tensor(V_tgt,  dtype=torch.float32, device=device)
    Ft   = torch.tensor(F_tgt,  dtype=torch.int64,   device=device)
    U_t  = torch.tensor(U,      dtype=torch.float32, device=device)

    # ── Learnable parameters (affine starts from identity) ────────────────────
    p_basis   = U_t.shape[1]
    phi       = torch.nn.Parameter(torch.zeros((p_basis, 3), device=device))
    w_ghd     = torch.nn.Parameter(torch.zeros(3,            device=device))
    log_s_ghd = torch.nn.Parameter(torch.tensor(0.0,         device=device))
    t_vec_ghd = torch.nn.Parameter(torch.zeros(3,            device=device))

    optim_params = [phi]
    if cfg.fit_R: optim_params.append(w_ghd)
    if cfg.fit_s: optim_params.append(log_s_ghd)
    if cfg.fit_T: optim_params.append(t_vec_ghd)

    optimizer = optim.Adam(optim_params, lr=cfg.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.n_iter, eta_min=cfg.eta_min,
    )

    # ── Fixed target samples (reused every iteration) ─────────────────────────
    tgt_mesh = Meshes(verts=[Vt], faces=[Ft])
    with torch.no_grad():
        Y_fixed, Y_normals_fixed = sample_points_from_meshes(
            tgt_mesh, num_samples=cfg.num_samples, return_normals=True,
        )

    # ── DVS samples ───────────────────────────────────────────────────────────
    print("Preparing DVS samples...")
    target_positives, target_negatives, dist_weights_p2n, dist_weights_n2p = \
        prepare_dvs_samples(V_tgt, F_tgt, cfg, device)

    # ── Loss modules ──────────────────────────────────────────────────────────
    src_mesh_init = Meshes(verts=[V0], faces=[F0])
    rigid_losser  = GHDRigidLoss(V0, F0).to(device)
    edge_losser   = EdgeLengthLoss(src_mesh_init).to(device)
    dvs_losser    = DVSOccupancyLoss(
        num_sample=cfg.num_dvs_sample, NP_ratio=cfg.NP_ratio,
    ).to(device)

    lm_losser = None
    if use_lm:
        lm_tgt_dome_t   = torch.tensor(
            lm_tgt_s["dome"].astype(np.float32),   device=device)
        lm_tgt_cap_up_t = torch.tensor(
            lm_tgt_s["cap_up"].astype(np.float32), device=device)
        lm_tgt_cap_dn_t = torch.tensor(
            lm_tgt_s["cap_down"].astype(np.float32), device=device)
        lm_losser = LandmarkLoss(idx_lm_dome, cap_up_idxs, cap_dn_idxs).to(device)

    # ── Baseline chamfer (for converged flag) ─────────────────────────────────
    with torch.no_grad():
        _, V_render0 = _render_ghd(V0, U_t, phi, w_ghd, log_s_ghd, t_vec_ghd)
        src_mesh0    = Meshes(verts=[V_render0], faces=[F0])
        X0           = sample_points_from_meshes(src_mesh0, num_samples=cfg.num_samples)
        chamfer_init = chamfer_distance(X0, Y_fixed)[0].item()
    print(f"Baseline Chamfer: {chamfer_init:.6f}")

    # ── History & best state ──────────────────────────────────────────────────
    history: dict[str, list] = {k: [] for k in [
        "chamfer", "chamfer_n1", "occupancy", "laplacian",
        "consistency", "edge", "rigid", "landmark", "total",
        "lambda_rigid_w", "lr", "rot_norm", "scale", "trans_norm",
    ]}

    best_chamfer = float("inf")
    best_iter    = 0
    best_state   = {
        "phi":   phi.detach().clone(),
        "w":     w_ghd.detach().clone(),
        "log_s": log_s_ghd.detach().clone(),
        "t":     t_vec_ghd.detach().clone(),
    }

    decay_iters = max(1, int(cfg.n_iter * cfg.rigid_decay_frac))
    if cfg.save_every > 0:
        os.makedirs(cfg.out_dir, exist_ok=True)

    # ── Fitting loop ──────────────────────────────────────────────────────────
    print("Starting GHD fitting loop...")
    _loop_t0 = time.time()
    for iteration in range(cfg.n_iter):
        optimizer.zero_grad()

        _, V_rendered = _render_ghd(V0, U_t, phi, w_ghd, log_s_ghd, t_vec_ghd)
        src_mesh      = Meshes(verts=[V_rendered], faces=[F0])

        # 1 & 2. Chamfer P0 + N1
        X, X_normals = sample_points_from_meshes(
            src_mesh, num_samples=cfg.num_samples, return_normals=True,
        )
        loss_chamfer, loss_chamfer_n1 = chamfer_distance(
            X, Y_fixed, x_normals=X_normals, y_normals=Y_normals_fixed,
        )

        # 3. DVS occupancy + Dice
        loss_occupancy = dvs_losser(
            src_mesh, target_positives, target_negatives,
            dist_weights_p2n, dist_weights_n2p,
        )

        # 4. Laplacian smoothness (cot)
        loss_laplacian = laplacian_loss(src_mesh, method="cot")

        # 5. Normal consistency
        loss_consistency = normal_consistency_loss(src_mesh)

        # 6. Edge length regularisation
        loss_edge = edge_losser(src_mesh)

        # 7. ARAP rigid
        loss_rigid = rigid_losser(V_rendered)

        # 8. Rigid weight decay schedule
        if iteration < decay_iters:
            prog = iteration / max(1, decay_iters - 1)
            lambda_rigid_curr = (
                cfg.lambda_rigid_start
                + prog * (cfg.lambda_rigid_end - cfg.lambda_rigid_start)
            )
        else:
            lambda_rigid_curr = cfg.lambda_rigid_end

        # 9. Landmark loss (tapered to zero over first lm_taper_frac of iters)
        loss_landmark = torch.tensor(0.0, device=device)
        if use_lm and lm_losser is not None:
            lm_taper      = max(0.0, 1.0 - iteration / (cfg.n_iter * cfg.lm_taper_frac))
            loss_landmark = lm_losser(
                V_rendered, lm_tgt_dome_t, lm_tgt_cap_up_t, lm_tgt_cap_dn_t,
            ) * lm_taper

        loss = (
              cfg.lambda_chamfer     * loss_chamfer
            + cfg.lambda_chamfer_n1  * loss_chamfer_n1
            + cfg.lambda_occupancy   * loss_occupancy
            + cfg.lambda_laplacian   * loss_laplacian
            + cfg.lambda_consistency * loss_consistency
            + cfg.lambda_edge        * loss_edge
            + lambda_rigid_curr      * loss_rigid
            + cfg.lambda_landmark    * loss_landmark
        )

        loss.backward()
        optimizer.step()
        scheduler.step()

        # Track history
        lc = loss_chamfer.item()
        history["chamfer"].append(lc)
        history["chamfer_n1"].append(loss_chamfer_n1.item())
        history["occupancy"].append(loss_occupancy.item())
        history["laplacian"].append(loss_laplacian.item())
        history["consistency"].append(loss_consistency.item())
        history["edge"].append(loss_edge.item())
        history["rigid"].append(loss_rigid.item())
        history["landmark"].append(loss_landmark.item()
                                    if isinstance(loss_landmark, torch.Tensor)
                                    else float(loss_landmark))
        history["total"].append(loss.item())
        history["lambda_rigid_w"].append(lambda_rigid_curr)
        history["lr"].append(scheduler.get_last_lr()[0])
        history["rot_norm"].append(torch.norm(w_ghd).item())
        history["scale"].append(torch.exp(log_s_ghd).item())
        history["trans_norm"].append(torch.norm(t_vec_ghd).item())

        if lc < best_chamfer:
            best_chamfer = lc
            best_iter    = iteration
            best_state   = {
                "phi":   phi.detach().clone(),
                "w":     w_ghd.detach().clone(),
                "log_s": log_s_ghd.detach().clone(),
                "t":     t_vec_ghd.detach().clone(),
            }

        if iteration % cfg.log_every == 0:
            _elapsed = time.time() - _loop_t0
            _iter_per_sec = (iteration + 1) / max(_elapsed, 1e-6)
            print(
                f"Iter {iteration:5d} | "
                f"Chamfer: {lc:.6f} | "
                f"N1: {loss_chamfer_n1.item():.4f} | "
                f"Occ: {loss_occupancy.item():.4f} | "
                f"Lap: {loss_laplacian.item():.2e} | "
                f"Cons: {loss_consistency.item():.2e} | "
                f"Edge: {loss_edge.item():.4f} | "
                f"Rigid: {loss_rigid.item():.4f} (w={lambda_rigid_curr:.3f}) | "
                f"LM: {loss_landmark.item() if isinstance(loss_landmark, torch.Tensor) else 0.:.4f} | "
                f"Total: {loss.item():.6f} | "
                f"lr: {scheduler.get_last_lr()[0]:.2e} | "
                f"{_iter_per_sec:.1f} it/s"
            )

        if cfg.save_every > 0 and iteration % cfg.save_every == 0:
            with torch.no_grad():
                V_out = V_rendered.detach().cpu().numpy()
            trimesh.Trimesh(vertices=V_out, faces=F, process=False).export(
                os.path.join(cfg.out_dir, f"ghd_fit_iter_{iteration:05d}.obj")
            )

    # ── Restore best checkpoint ───────────────────────────────────────────────
    phi.data.copy_(best_state["phi"])
    w_ghd.data.copy_(best_state["w"])
    log_s_ghd.data.copy_(best_state["log_s"])
    t_vec_ghd.data.copy_(best_state["t"])

    with torch.no_grad():
        _, V_final_t = _render_ghd(V0, U_t, phi, w_ghd, log_s_ghd, t_vec_ghd)
        V_final = V_final_t.detach().cpu().numpy()

    # ── Evaluation metrics ────────────────────────────────────────────────────
    final_mesh = Meshes(verts=[V_final_t], faces=[F0])

    dice         = _compute_dice(final_mesh, target_positives, target_negatives)
    gar          = _good_angle_ratio(V_final, F)
    chamfer_final = history["chamfer"][-1]
    converged    = chamfer_final < chamfer_init * cfg.converge_threshold

    print(f"\nFitting complete.")
    print(f"  Best   Chamfer : {best_chamfer:.6f} @ iter {best_iter}")
    print(f"  Final  Chamfer : {chamfer_final:.6f}")
    print(f"  Dice           : {dice:.4f}")
    print(f"  GAR            : {gar:.4f}")
    print(f"  Converged      : {converged}")

    return FitResult(
        verts         = V_final,
        loss_history  = history,
        chamfer_final = chamfer_final,
        chamfer_best  = best_chamfer,
        best_iter     = best_iter,
        dice          = dice,
        gar           = gar,
        converged     = converged,
    )
