#!/usr/bin/env python3
"""
fit_case.py — GHD fitting for a single target aneurysm mesh.

Run from the repo root:
    python scripts/fit_case.py \\
        --canonical  aneu_ghd/canonical/canonical.obj \\
        --target     path/to/target/final_aligned.obj \\
        --out        results/C0002/ \\
        [--eigenvecs aneu_ghd/canonical/eigenvectors.npy] \\
        [--landmarks-dir /path/to/aneu_ghd] \\
        [--canonical-id  canonical] \\
        [--target-id     GHD2/Checkpoints/Alignment/C0002] \\
        [--device cuda:2] [--n-iter 12000] [--n-basis 121]

Outputs (saved to --out):
    fitted_mesh.obj     — fitted canonical mesh in normalised space
    metrics.json        — chamfer, dice, GAR, converged flag
    loss_history.npz    — per-iteration loss values for plotting
"""

import argparse
import json
import os
import sys

import numpy as np

# ── Allow running from the repo root without pip install ──────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aneu_ghd import (
    load_obj, save_obj,
    load_landmarks, normalize_lm,
    cpd_align,
    FitConfig, ghd_fit,
    compute_eigenvectors,
)


# ── Normalisation (same as ghd.ipynb STEP 1b) ─────────────────────────────────

def normalise_mesh(V):
    """Centroid subtraction + max-norm scaling → vertices inside unit sphere."""
    c = V.mean(axis=0)
    Vn = V - c
    s = float(np.linalg.norm(Vn, axis=1).max())
    return Vn / s, c, s


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GHD fitting for a single target aneurysm mesh.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--canonical",      required=True,  help="Path to canonical final_aligned.obj")
    parser.add_argument("--target",         required=True,  help="Path to target   final_aligned.obj")
    parser.add_argument("--out",            required=True,  help="Output directory")
    parser.add_argument("--eigenvecs",      default=None,   help="Path to precomputed eigenvectors.npy (skips Laplacian build)")
    parser.add_argument("--landmarks-dir",  default=None,   help="Root dir for landmark .npz files")
    parser.add_argument("--canonical-id",   default=None,   help="Canonical case ID relative to --landmarks-dir")
    parser.add_argument("--target-id",      default=None,   help="Target    case ID relative to --landmarks-dir")
    parser.add_argument("--device",         default="cuda:0")
    parser.add_argument("--n-iter",         type=int, default=12_000)
    parser.add_argument("--n-basis",        type=int, default=121,   help="Number of Laplacian eigenvectors")
    parser.add_argument("--save-eigenvecs", action="store_true",     help="Save computed eigenvectors next to the canonical mesh")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # ── 1. Load & normalise meshes ────────────────────────────────────────────
    print("Loading meshes...")
    V_can, F_can = load_obj(args.canonical)
    V_tgt, F_tgt = load_obj(args.target)

    V_can = V_can.astype(np.float64)
    V_tgt = V_tgt.astype(np.float64)

    V_n,   c_can, s_can = normalise_mesh(V_can)
    V_tgt_n, c_tgt, s_tgt = normalise_mesh(V_tgt)
    V_n     = V_n.astype(np.float32)
    V_tgt_n = V_tgt_n.astype(np.float32)

    print(f"  Canonical : {V_n.shape[0]} verts  scale={s_can:.3f} mm")
    print(f"  Target    : {V_tgt_n.shape[0]} verts  scale={s_tgt:.3f} mm")

    # ── 2. Eigenvectors ───────────────────────────────────────────────────────
    if args.eigenvecs and os.path.exists(args.eigenvecs):
        print(f"Loading eigenvectors from {args.eigenvecs}")
        U = np.load(args.eigenvecs).astype(np.float32)
    else:
        print("Computing eigenvectors (this takes ~1 min; save with --save-eigenvecs)...")
        U = compute_eigenvectors(V_n.astype(np.float64), F_can, n_basis=args.n_basis)
        if args.save_eigenvecs:
            cache = os.path.join(os.path.dirname(args.canonical), "eigenvectors.npy")
            np.save(cache, U)
            print(f"  Saved eigenvectors to {cache}")

    print(f"  U shape: {U.shape}")

    # ── 3. Landmarks (optional) ───────────────────────────────────────────────
    lm_can_s = lm_tgt_s = None
    idx_lm_dome = cap_up_idxs = cap_dn_idxs = None

    if args.landmarks_dir and args.canonical_id and args.target_id:
        print("Loading landmarks...")
        lm_can = load_landmarks(args.landmarks_dir, args.canonical_id)
        lm_tgt = load_landmarks(args.landmarks_dir, args.target_id)
        lm_can_s = normalize_lm(lm_can, c_can, s_can)
        lm_tgt_s = normalize_lm(lm_tgt, c_tgt, s_tgt)

        idx_lm_dome = int(np.argmin(np.linalg.norm(V_n - lm_can_s["dome"], axis=1)))
        k_cap       = 20
        cap_up_idxs = np.argsort(np.linalg.norm(V_n - lm_can_s["cap_up"],   axis=1))[:k_cap]
        cap_dn_idxs = np.argsort(np.linalg.norm(V_n - lm_can_s["cap_down"], axis=1))[:k_cap]
        print(f"  Landmark indices: dome={idx_lm_dome}, cap k={k_cap}")
    else:
        print("No landmarks provided — landmark loss disabled.")

    # ── 4. CPD alignment ──────────────────────────────────────────────────────
    print("Running CPD alignment...")
    V_aligned_np, R_cpd, t_cpd = cpd_align(V_n, V_tgt_n)
    V_init_np = V_aligned_np.astype(np.float32)
    print(f"  CPD: det(R)={np.linalg.det(R_cpd):.4f}  |t|={np.linalg.norm(t_cpd):.4f}")

    # ── 5. GHD fitting ────────────────────────────────────────────────────────
    cfg = FitConfig(
        n_iter          = args.n_iter,
        device          = args.device,
        lambda_landmark = 0.1 if lm_can_s is not None else 0.0,
    )

    result = ghd_fit(
        V_init      = V_init_np,
        F           = F_can,
        V_tgt       = V_tgt_n.astype(np.float32),
        F_tgt       = F_tgt,
        U           = U,
        cfg         = cfg,
        lm_can_s    = lm_can_s,
        lm_tgt_s    = lm_tgt_s,
        idx_lm_dome = idx_lm_dome,
        cap_up_idxs = cap_up_idxs,
        cap_dn_idxs = cap_dn_idxs,
    )

    # ── 6. Save outputs ───────────────────────────────────────────────────────
    fitted_path = os.path.join(args.out, "fitted_mesh.obj")
    save_obj(fitted_path, result.verts, F_can)
    print(f"Saved fitted mesh → {fitted_path}")

    metrics = {
        "chamfer_best":  result.chamfer_best,
        "chamfer_final": result.chamfer_final,
        "best_iter":     result.best_iter,
        "dice":          result.dice,
        "gar":           result.gar,
        "converged":     result.converged,
    }
    metrics_path = os.path.join(args.out, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics     → {metrics_path}")
    print(json.dumps(metrics, indent=2))

    history_path = os.path.join(args.out, "loss_history.npz")
    np.savez(history_path, **{k: np.array(v) for k, v in result.loss_history.items()})
    print(f"Saved loss history → {history_path}")


if __name__ == "__main__":
    main()
