"""clone_reframe.py -- put a foreign pipeline's fitted mesh into THIS repo's
alignment frame, using surface chamfer.

WHY THIS EXISTS
---------------
The old GHD project's ghd_fitted.obj is expressed in ITS OWN alignment frame:
prep_case_for_ghd.py built a RIGID frame (neck point at the origin, rotation
R_frame from ring/tangent geometry) and normalised by
    s_can = ||V_can||max * 1.10 * 2.5
This repo's alignment (ghd/fitting/alignment.py) instead fits a SIMILARITY
transform s*(x@R.T)+t by gradient descent and bakes its scale straight into
final_aligned.obj. The two frames are unrelated -- on a sample case they sit
53 degrees and 0.88x apart -- so an old fitted mesh dropped into this repo as
a warm-start target lands in the wrong place entirely.

Chaining through world coordinates (old R_frame/neck_pt -> world -> this
repo's transform) IS exact when the metadata is complete, but it is brittle:
save_stage1_checkpoints deliberately drops `s` from landmarks.npz to imitate
the legacy format, so this repo's own frames are not invertible to world, and
any convention drift in the old dataset produces a confidently wrong answer
with no signal. So this module ignores metadata entirely and recovers the
transform from GEOMETRY.

METHOD
------
Fit a similarity transform clone -> target by minimising bidirectional
surface chamfer (Adam, GPU, ~200 epochs, well under a second per case).

Chamfer -- not landmark correspondence -- is the arbiter, because the clone
mesh's SURFACE is near-identical to the target (the old fits reached
chamfer_best ~1e-5 in their own units) while its opening-ring centroids need
NOT be: the two pipelines clip the branches at different lengths, so ring
centroids sit at different points along the same vessel. Measured on a test
case: ring-centroid Umeyama alone left 0.41 mm of landmark residual and
0.117 mm of surface error, while chamfer refinement reached 0.063 mm.

Landmarks are still the best INITIALISER, though -- chamfer's basin of
attraction does not span an arbitrary foreign rotation. From identity, 200
epochs converge to chamfer 1.43 (wrong minimum); from random rotations only
4 of 16 seeds find the right one. From a ring-centroid Umeyama init, a single
200-epoch run lands on the same optimum the 16-seed search finds, ~16x
cheaper. Random multi-start remains as the fallback when the rings are
unusable or the landmark init lands badly.

A useful side effect: the similarity's own scale absorbs s_can, so the old
pipeline's hand-embedded *1.10*2.5 never has to be reproduced or even known.

Validated on 20 ImperialNHS cases with both an old fit and a current run:
mean surface error 0.87 mm before -> 0.051 mm after, zero failures.
"""

import collections
import itertools
from pathlib import Path

import numpy as np
import torch
import trimesh
from pytorch3d.loss import chamfer_distance
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle, random_rotations

from ghd.fitting.alignment import umeyama_similarity

__all__ = ["reframe_clone_to_target", "boundary_ring_centroids"]


def boundary_ring_centroids(mesh):
    """Centroid of each open boundary loop -- i.e. each opening ring.

    Pure mesh topology (edges used by exactly one face), so it needs no
    landmark file. The old pipeline's ghd_fitted.obj is capped/watertight and
    therefore has NO boundary; pass its ghd_fitted_uncapped.obj sibling.
    """
    uniq, counts = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
    adj = collections.defaultdict(list)
    for a, b in uniq[counts == 1]:
        adj[a].append(b)
        adj[b].append(a)
    seen, loops = set(), []
    for start in adj:
        if start in seen:
            continue
        comp, stack = [], [start]
        while stack:
            v = stack.pop()
            if v in seen:
                continue
            seen.add(v)
            comp.append(v)
            stack.extend(adj[v])
        loops.append(comp)
    return np.array([np.asarray(mesh.vertices)[loop].mean(axis=0) for loop in loops])


def _refine_chamfer(src_pts, tgt_pts, R0, s0, t0, iters, lr, device):
    """Similarity fit src->tgt minimising bidirectional chamfer."""
    aa = torch.nn.Parameter(matrix_to_axis_angle(
        torch.as_tensor(R0, dtype=torch.float32, device=device).unsqueeze(0)).squeeze(0).clone())
    log_s = torch.nn.Parameter(torch.tensor(float(np.log(max(s0, 1e-9))), device=device))
    t = torch.nn.Parameter(torch.as_tensor(t0, dtype=torch.float32, device=device).clone())
    opt = torch.optim.Adam([aa, log_s, t], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, eta_min=lr * 0.05)
    loss = torch.zeros((), device=device)
    for _ in range(iters):
        R = axis_angle_to_matrix(aa.unsqueeze(0)).squeeze(0)
        moved = (torch.exp(log_s) * (src_pts @ R.T) + t).unsqueeze(0)
        loss = chamfer_distance(moved, tgt_pts)[0]
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    R = axis_angle_to_matrix(aa.detach().unsqueeze(0)).squeeze(0).cpu().numpy()
    return loss.item(), R, float(torch.exp(log_s).detach()), t.detach().cpu().numpy()


def reframe_clone_to_target(clone_mesh, target_mesh, clone_uncapped=None,
                            target_branch_endpoints=None, n_points=4000, iters=200,
                            lr=5e-2, fallback_seeds=16, fallback_loss=0.25,
                            device="cuda", verbose=True):
    """Similarity transform mapping `clone_mesh` onto `target_mesh`.

    Parameters
    ----------
    clone_mesh : trimesh.Trimesh
        The foreign fitted mesh, in whatever frame/scale it arrived in. Its
        raw vertices are fine -- no s_can rescale needed, the fitted scale
        absorbs it.
    target_mesh : trimesh.Trimesh
        This repo's final_aligned.obj, i.e. the real target already in
        canonical space. This is what the clone gets moved onto; the target
        itself never moves, because Stage B must fit in canonical space.
    clone_uncapped : trimesh.Trimesh, optional
        Uncapped sibling of `clone_mesh`, used only to derive the landmark
        INIT from its open boundary rings. Falls back to random multi-start
        when absent.
    target_branch_endpoints : (K, 3), optional
        Target-side landmarks (landmarks.npz's "branch_endpoints"), matched
        against the clone's ring centroids over all K! permutations, so a
        branch-ordering difference between the two repos cannot mis-pair them.

    Returns
    -------
    (R, s, t, info) with the transform applied as  s * (v @ R.T) + t
    """
    device = torch.device(device)
    src_np = np.asarray(clone_mesh.vertices, dtype=np.float64)

    S = torch.as_tensor(np.asarray(trimesh.sample.sample_surface(clone_mesh, n_points)[0],
                                   dtype=np.float32), device=device)
    T = torch.as_tensor(np.asarray(trimesh.sample.sample_surface(target_mesh, n_points)[0],
                                   dtype=np.float32), device=device).unsqueeze(0)
    src_c, tgt_c = S.mean(0), T[0].mean(0)
    # scale init from the RMS-radius ratio -- this is what absorbs s_can
    s_rms = float((T[0] - tgt_c).pow(2).sum(1).mean().sqrt()
                  / (S - src_c).pow(2).sum(1).mean().sqrt())

    best, init_used = None, None

    # ---- landmark init: ring-centroid Umeyama, best of all permutations ----
    if clone_uncapped is not None and target_branch_endpoints is not None:
        try:
            src_ep = boundary_ring_centroids(clone_uncapped)
            dst_ep = np.asarray(target_branch_endpoints, dtype=np.float64)
            if len(src_ep) == len(dst_ep) >= 3:
                best_lm = None
                for perm in itertools.permutations(range(len(dst_ep))):
                    R_, s_, t_ = umeyama_similarity(src_ep[list(perm)], dst_ep)
                    resid = np.linalg.norm(
                        (s_ * (src_ep[list(perm)] @ R_.T) + t_) - dst_ep, axis=1).mean()
                    if best_lm is None or resid < best_lm[0]:
                        best_lm = (resid, R_, s_, t_, perm)
                resid, R_, s_, t_, perm = best_lm
                best = _refine_chamfer(S, T, R_, s_, t_, iters, lr, device)
                init_used = "landmark"
                if verbose:
                    print(f"  reframe: landmark init (perm={perm}, residual={resid:.4f} mm) "
                          f"-> chamfer={best[0]:.6f}", flush=True)
        except Exception as exc:  # noqa: BLE001 -- any landmark problem just falls back
            if verbose:
                print(f"  reframe: landmark init unavailable ({exc}); using multi-start.", flush=True)

    # ---- fallback: random multi-start over SO(3) ----
    if best is None or best[0] > fallback_loss:
        torch.manual_seed(0)
        seeds = torch.cat([torch.eye(3, device=device).unsqueeze(0),
                           random_rotations(max(fallback_seeds - 1, 0), device=device)], 0)
        cands = []
        for k in range(seeds.shape[0]):
            Rk = seeds[k]
            t0 = (tgt_c - s_rms * (src_c @ Rk.T)).cpu().numpy()
            cands.append(_refine_chamfer(S, T, Rk.cpu().numpy(), s_rms, t0, iters, lr, device))
        cands.sort(key=lambda c: c[0])
        if best is None or cands[0][0] < best[0]:
            best, init_used = cands[0], "multistart"
        if verbose:
            print(f"  reframe: multi-start over {seeds.shape[0]} seeds -> chamfer={best[0]:.6f} "
                  f"(2nd={cands[1][0]:.6f})" if len(cands) > 1 else "", flush=True)

    loss, R, s, t = best
    verts = s * (src_np @ R.T) + t
    reframed = trimesh.Trimesh(vertices=verts, faces=np.asarray(clone_mesh.faces), process=False)

    # residual gate -- surface error relative to the target's own size
    dist = np.abs(trimesh.proximity.ProximityQuery(target_mesh).signed_distance(verts))
    tv = np.asarray(target_mesh.vertices)
    obj_rms = float(np.sqrt(((tv - tv.mean(0)) ** 2).sum(1).mean()))
    info = {"chamfer": float(loss), "scale": float(s), "init": init_used,
            "mean_surface_dist": float(dist.mean()), "p95_surface_dist": float(np.percentile(dist, 95)),
            "max_surface_dist": float(dist.max()), "target_rms": obj_rms,
            "rel_error": float(dist.mean() / max(obj_rms, 1e-9)),
            "ok": bool(dist.mean() < 0.03 * obj_rms)}
    if verbose:
        flag = "" if info["ok"] else "   ** CHECK: exceeds 3% of target RMS **"
        print(f"  reframe: init={init_used} scale={s:.4f} mean_surf_dist={dist.mean():.4f} mm "
              f"({info['rel_error']*100:.2f}% of target RMS){flag}", flush=True)
    return R, s, t, reframed, info
