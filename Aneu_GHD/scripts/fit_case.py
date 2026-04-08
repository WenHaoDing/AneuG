#!/usr/bin/env python3
"""
fit_case.py — GHD fitting for a single target aneurysm mesh.

Run from the repo root:

  # Sidewall
  python scripts/fit_case.py \\
      --type sidewall \\
      --canonical  GHD2/Checkpoints/Alignment/p044_.../final_aligned.obj \\
      --eigenvecs  GHD2/Checkpoints/Alignment/p044_.../eigenvectors.npy \\
      --canonical-lm GHD2/Checkpoints/Alignment/p044_... \\
      --target     Aneu_GHD/Checkpoints/Alignment/CaseID/final_aligned.obj \\
      --target-lm  Aneu_GHD/Checkpoints/Alignment/CaseID \\
      --out        fitting_results/CaseID/

  # Bifurcation
  python scripts/fit_case.py \\
      --type bifurcation \\
      --canonical  Aneu_GHD/canonical/Bifricated/canonical_typeB/part_aligned.obj \\
      --eigenvecs  Aneu_GHD/canonical/Bifricated/canonical_typeB/eigenvectors.npy \\
      --canonical-rings Aneu_GHD/canonical/Bifricated/canonical_typeB/opa_checkpoint.pkl \\
      --target     Aneu_GHD/Checkpoints/Alignment/CaseID/final_aligned.obj \\
      --target-lm  Aneu_GHD/Checkpoints/Alignment/CaseID \\
      --out        fitting_results/CaseID/

    cd "/media/yaplab2/HDD Storage/almaha" && \
/home/yaplab2/miniconda3/envs/new/bin/python3 Aneu_GHD/scripts/fit_case.py \
    --target    "Aneu_GHD/Checkpoints/Alignment/ANSYS_UNIGE_09_cut1/final_aligned.obj" \
    --target-lm "Aneu_GHD/Checkpoints/Alignment/ANSYS_UNIGE_09_cut1" \
    --out       "Aneu_GHD/fitting_results/AnueX/ANSYS_UNIGE_09_cut1"


Key differences between types
------------------------------
Sidewall   : 2 rings from landmarks.npz (ring_up_idxs / ring_dn_idxs)
Bifurcation: 3 rings from landmarks.npz using Procrustes ring mapping
             (can_ring_for_up / can_ring_for_dn / can_ring_for_dn2)
Both types use Procrustes pre-alignment from prep_case_for_ghd.py;
ring-chamfer and centroid losses in GHD handle residual alignment.

Outputs (saved to --out):
    ghd_fitted.obj          — fitted mesh in normalised space
    ghd_fitted_world.obj    — fitted mesh in original patient coordinates
    ghd_fitted_uncapped_world.obj — same, with vmtk caps removed (if --uncap)
    metrics.json        — chamfer, dice, GAR, converged flag
    loss_history.npz    — per-iteration loss values
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Aneu_GHD import (
    load_obj, save_obj,
    load_landmarks,
    FitConfig, ghd_fit,
    compute_eigenvectors,
)


def main():
    parser = argparse.ArgumentParser(
        description="GHD fitting for a single target aneurysm mesh.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target",              required=True,  help="Path to target .obj")
    parser.add_argument("--out",                 required=True,  help="Output directory")
    parser.add_argument("--type",                default=None,   choices=["sidewall", "bifurcation"],
                                                                 help="Aneurysm type (auto-detected from landmarks.npz if not set)")
    # Ring sources
    parser.add_argument("--canonical-lm",        default=None,   help="[sidewall] Dir containing canonical landmarks.npz (auto-set if not given)")
    parser.add_argument("--canonical-rings",     default=None,   help="[bifurcation] Path to canonical opa_checkpoint.pkl (auto-set if not given)")
    parser.add_argument("--target-lm",           default=None,   help="[both] Dir containing target landmarks.npz")
    # Override canonical — only needed if using a non-default canonical
    parser.add_argument("--canonical",           default=None,   help="Path to canonical .obj (auto-set by type if not given)")
    # Fitting hyperparameters
    parser.add_argument("--eigenvecs",           default=None,   help="Path to precomputed eigenvectors.npy (auto-set by type if not given)")
    parser.add_argument("--device",              default="cuda:0")
    parser.add_argument("--n-iter",              type=int,   default=15_000)
    parser.add_argument("--n-basis",             type=int,   default=144)
    parser.add_argument("--lambda-ring-chamfer", type=float, default=0.5,   help="Ring chamfer loss weight (0=disabled)")
    parser.add_argument("--lambda-centroid",     type=float, default=0.1,   help="Centroid L2 loss weight (0=disabled)")
    parser.add_argument("--lambda-laplacian",    type=float, default=1e-2,  help="Laplacian smoothness weight")
    parser.add_argument("--lambda-rigid-end",    type=float, default=0.05,  help="Rigid loss weight at end of decay schedule")
    parser.add_argument("--lambda-rigid-start",  type=float, default=3.0,   help="Rigid loss weight at start of decay schedule")
    parser.add_argument("--rigid-decay-frac",    type=float, default=0.80,  help="Fraction of iterations over which rigid weight decays")
    parser.add_argument("--lambda-thickness",    type=float, default=0.5,   help="Minimum wall-thickness hinge loss weight (0=disabled)")
    parser.add_argument("--thickness-r",         type=float, default=0.2,   help="MeshThickness search radius in normalised space")
    parser.add_argument("--lambda-consistency",  type=float, default=0.5,   help="Normal consistency loss weight")
    parser.add_argument("--lambda-planarity",    type=float, default=0.1,   help="Ring planarity loss weight (0=disabled)")
    parser.add_argument("--lambda-edge",         type=float, default=0.1,   help="Edge length regularisation weight")
    parser.add_argument("--lambda-occupancy",    type=float, default=1.0,   help="DVS occupancy loss weight (increase to prevent branch-dome merging)")
    parser.add_argument("--dvs-surf-d-min",      type=float, default=0.01,  help="Min surface-offset distance for DVS sampling (normalised space)")
    parser.add_argument("--dvs-surf-d-max",      type=float, default=0.05,  help="Max surface-offset distance for DVS sampling (normalised space)")
    parser.add_argument("--save-eigenvecs",      action="store_true", help="Save computed eigenvectors next to canonical")
    parser.add_argument("--save-every",          type=int, default=2000, help="Save intermediate GHD mesh every N epochs (0=disabled)")
    parser.add_argument("--uncap",               action="store_true",   help="Remove vmtk caps from the fitted mesh after fitting")
    args = parser.parse_args()

    # Auto-detect type from landmarks.npz if not explicitly set
    if args.type is None:
        if args.target_lm is None:
            raise ValueError("--type could not be auto-detected: --target-lm not provided.")
        _lm = np.load(os.path.join(args.target_lm, "landmarks.npz"), allow_pickle=True)
        if "aneu_type" in _lm:
            args.type = str(_lm["aneu_type"])
        elif "ring_dn2_idxs" in _lm:
            args.type = "bifurcation"   # 3 rings → bifurcation
        else:
            args.type = "sidewall"      # 2 rings → sidewall
        print(f"Auto-detected type: {args.type}")

    # Set canonical paths based on type if not explicitly provided
    _REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _CAN_SIDEWALL = os.path.join(_REPO, "canonical/Sidewall")
    _CAN_BIF      = os.path.join(_REPO, "canonical/Bifricated/canonical_typeB")

    if args.canonical is None:
        args.canonical = os.path.join(
            _CAN_SIDEWALL if args.type == "sidewall" else _CAN_BIF,
            "final_aligned.obj" if args.type == "sidewall" else "part_aligned.obj",
        )
    if args.eigenvecs is None:
        args.eigenvecs = os.path.join(
            _CAN_SIDEWALL if args.type == "sidewall" else _CAN_BIF,
            "eigenvectors.npy",
        )
    if args.canonical_lm is None and args.type == "sidewall":
        args.canonical_lm = _CAN_SIDEWALL
    if args.canonical_rings is None and args.type == "bifurcation":
        args.canonical_rings = os.path.join(_CAN_BIF, "opa_checkpoint.pkl")

    print(f"Canonical : {args.canonical}")
    print(f"Eigenvecs : {args.eigenvecs}")

    os.makedirs(args.out, exist_ok=True)

    # ── 1. Load meshes ────────────────────────────────────────────────────────
    print("Loading meshes...")
    V_can, F_can = load_obj(args.canonical)
    V_tgt, F_tgt = load_obj(args.target)
    V_can = V_can.astype(np.float64)
    V_tgt = V_tgt.astype(np.float64)
    print(f"  Canonical : {V_can.shape[0]} verts")
    print(f"  Target    : {V_tgt.shape[0]} verts")

    # ── 2. Normalisation ──────────────────────────────────────────────────────
    # Neck is already at origin from the cutting pipeline.
    # Divide by canonical max-radius to preserve size ratio.
    s_can   = float(np.linalg.norm(V_can, axis=1).max())
    V_n     = (V_can / s_can).astype(np.float32)
    V_tgt_n = (V_tgt / s_can).astype(np.float32)
    print(f"  s_can={s_can:.3f} mm  target size ratio={float(np.linalg.norm(V_tgt, axis=1).max()) / s_can:.3f}")

    # ── 3. Eigenvectors ───────────────────────────────────────────────────────
    if args.eigenvecs and os.path.exists(args.eigenvecs):
        print(f"Loading eigenvectors: {args.eigenvecs}")
        U = np.load(args.eigenvecs).astype(np.float32)
        if U.shape[1] > args.n_basis:
            U = U[:, :args.n_basis]
            print(f"  Truncated to {args.n_basis} basis vectors")
    else:
        print(f"Computing eigenvectors (n_basis={args.n_basis})...")
        U = compute_eigenvectors(V_n.astype(np.float64), F_can, n_basis=args.n_basis)
        if args.save_eigenvecs:
            cache = os.path.join(os.path.dirname(args.canonical), "eigenvectors.npy")
            np.save(cache, U)
            print(f"  Saved → {cache}")
    print(f"  U shape: {U.shape}")

    # ── 4. Ring pairs + centroid pairs ────────────────────────────────────────
    ring_pairs               = None
    centroid_pairs           = None
    cap_interior_vertex_sets = None

    if args.type == "sidewall":
        if args.canonical_lm and args.target_lm:
            print("Loading landmarks (sidewall)...")
            lm_can = load_landmarks(args.canonical_lm, "")
            lm_tgt = load_landmarks(args.target_lm,    "")
            can_up = lm_can["ring_up_idxs"]
            can_dn = lm_can["ring_dn_idxs"]
            tgt_up = V_tgt_n[lm_tgt["ring_up_idxs"]].astype(np.float32)
            tgt_dn = V_tgt_n[lm_tgt["ring_dn_idxs"]].astype(np.float32)
            ring_pairs = [(can_up, tgt_up), (can_dn, tgt_dn)]
            centroid_pairs = [
                (can_up, tgt_up.mean(axis=0)),
                (can_dn, tgt_dn.mean(axis=0)),
            ]
            print(f"  ring up : can={len(can_up)}  tgt={tgt_up.shape[0]} verts")
            print(f"  ring dn : can={len(can_dn)}  tgt={tgt_dn.shape[0]} verts")
        else:
            print("No sidewall landmark dirs provided — ring losses disabled.")

    else:  # bifurcation
        if args.canonical_rings and args.target_lm:
            import pickle
            print("Loading ring openings (bifurcation)...")
            with open(args.canonical_rings, "rb") as f:
                can_opa = pickle.load(f)
            # Target ring indices + Procrustes ring mapping both live in landmarks.npz
            tgt_lm = load_landmarks(args.target_lm, "")

            # Procrustes-computed canonical ring assignment (saved by prep_case_for_ghd.py)
            _can_up  = int(tgt_lm["can_ring_for_up"])
            _can_dn  = int(tgt_lm["can_ring_for_dn"])
            _can_dn2 = int(tgt_lm["can_ring_for_dn2"])

            _can_up_idxs  = np.array(can_opa["op_v_indices"][_can_up])
            _can_dn_idxs  = np.array(can_opa["op_v_indices"][_can_dn])
            _can_dn2_idxs = np.array(can_opa["op_v_indices"][_can_dn2])

            _tgt_up_full  = V_tgt_n[tgt_lm["ring_up_idxs"]].astype(np.float32)
            _tgt_dn_full  = V_tgt_n[tgt_lm["ring_dn_idxs"]].astype(np.float32)
            _tgt_dn2_full = V_tgt_n[tgt_lm["ring_dn2_idxs"]].astype(np.float32)

            cap_interior_vertex_sets = [
                np.array(can_opa["op_rec_v_indices_map"][i]) for i in range(3)
            ]
            ring_pairs = [
                (_can_up_idxs,  _tgt_up_full),
                (_can_dn_idxs,  _tgt_dn_full),
                (_can_dn2_idxs, _tgt_dn2_full),
            ]
            centroid_pairs = [
                (_can_up_idxs,  _tgt_up_full.mean(axis=0)),
                (_can_dn_idxs,  _tgt_dn_full.mean(axis=0)),
                (_can_dn2_idxs, _tgt_dn2_full.mean(axis=0)),
            ]
            print(f"  canonical ring mapping: up→{_can_up}  dn→{_can_dn}  dn2→{_can_dn2}")
            for i, (ci, tp) in enumerate(ring_pairs):
                print(f"  ring {i}: can={len(ci)}  tgt={tp.shape[0]} verts")
        else:
            print("No bifurcation ring info provided — ring losses disabled.")

    # ── 5. GHD fitting ────────────────────────────────────────────────────────
    cfg = FitConfig(
        n_iter              = args.n_iter,
        device              = args.device,
        lambda_ring_chamfer = args.lambda_ring_chamfer if ring_pairs is not None else 0.0,
        lambda_centroid     = args.lambda_centroid     if centroid_pairs is not None else 0.0,
        lambda_laplacian    = args.lambda_laplacian,
        lambda_rigid_start  = args.lambda_rigid_start,
        lambda_rigid_end    = args.lambda_rigid_end,
        rigid_decay_frac    = args.rigid_decay_frac,
        lambda_thickness    = args.lambda_thickness,
        thickness_r         = args.thickness_r,
        lambda_consistency  = args.lambda_consistency,
        lambda_planarity    = args.lambda_planarity,
        lambda_edge         = args.lambda_edge,
        lambda_occupancy    = args.lambda_occupancy,
        dvs_surf_d_min      = args.dvs_surf_d_min,
        dvs_surf_d_max      = args.dvs_surf_d_max,
        save_every          = args.save_every,
        out_dir             = os.path.join(args.out, "ghd_checkpoints"),
    )

    result = ghd_fit(
        V_init                   = V_n,
        F                        = F_can,
        V_tgt                    = V_tgt_n,
        F_tgt                    = F_tgt,
        U                        = U,
        cfg                      = cfg,
        ring_pairs               = ring_pairs,
        centroid_pairs           = centroid_pairs,
        uncap                    = args.uncap,
        cap_interior_vertex_sets = cap_interior_vertex_sets,
    )

    # ── 6. Export ─────────────────────────────────────────────────────────────
    # Always save normalised-space mesh (the primary GHD output).
    fitted_path = os.path.join(args.out, "ghd_fitted.obj")
    save_obj(fitted_path, result.verts.astype(np.float32), F_can)
    print(f"Saved fitted mesh (normalised) → {fitted_path}")

    # World-space export: invert the alignment applied in prep_case_for_ghd.py.
    #   forward:  verts_world → (v - neck_pt) @ R.T → / s_can → verts_norm
    #   inverse:  verts_norm  → * s_can → @ R + neck_pt → verts_world
    if args.target_lm:
        lm_tgt  = np.load(os.path.join(args.target_lm, "landmarks.npz"), allow_pickle=True)
        neck_pt = np.array(lm_tgt["neck_pt_world"], dtype=np.float64)
        R_frame = np.array(lm_tgt["R_frame"],       dtype=np.float64)

        def _to_world(verts_norm: np.ndarray) -> np.ndarray:
            return ((verts_norm.astype(np.float64) * s_can) @ R_frame + neck_pt).astype(np.float32)

        world_path = os.path.join(args.out, "ghd_fitted_world.obj")
        save_obj(world_path, _to_world(result.verts), F_can)
        print(f"Saved fitted mesh (world)     → {world_path}")

        if args.uncap and result.verts_uncapped is not None:
            uncapped_world_path = os.path.join(args.out, "ghd_fitted_uncapped_world.obj")
            save_obj(uncapped_world_path, _to_world(result.verts_uncapped), result.faces_uncapped)
            print(f"Saved uncapped mesh (world)   → {uncapped_world_path}")
    else:
        print("  WARNING: --target-lm not provided; skipping world-space export.")

    if args.uncap and result.verts_uncapped is not None:
        uncapped_path = os.path.join(args.out, "ghd_fitted_uncapped.obj")
        save_obj(uncapped_path, result.verts_uncapped.astype(np.float32), result.faces_uncapped)
        print(f"Saved uncapped mesh (normalised) → {uncapped_path}")

    metrics = {
        "chamfer_best":  result.chamfer_best,
        "chamfer_final": result.chamfer_final,
        "best_iter":     result.best_iter,
        "dice":          result.dice,
        "gar":           result.gar,
        "converged":     result.converged,
        "s_can":         float(s_can),
    }
    metrics_path = os.path.join(args.out, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics      → {metrics_path}")
    print(json.dumps(metrics, indent=2))

    history_path = os.path.join(args.out, "loss_history.npz")
    np.savez(history_path, **{k: np.array(v) for k, v in result.loss_history.items()})
    print(f"Saved loss history → {history_path}")

    coeffs_path = os.path.join(args.out, "ghd_coefficients.npz")
    np.savez(
        coeffs_path,
        phi       = result.phi,
        w_rot     = result.w_rot,
        log_scale = np.array(result.log_scale),
        t_vec     = result.t_vec,
    )
    print(f"Saved GHD coefficients → {coeffs_path}")


if __name__ == "__main__":
    main()
