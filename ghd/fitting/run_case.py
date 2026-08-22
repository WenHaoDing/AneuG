"""
run_case.py -- single-command driver: Stage 1 (rigid-similarity alignment,
ghd/fitting/alignment.py) followed by Stage 2 (GHD-coefficient fitting +
residual pose refinement, ghd/fitting/ghd_fit.py), for one case.

Stage 1 saves final_aligned.obj / landmarks.npz / centerline.vtp /
alignment_result.npy / alignment_sanity.png; Stage 2 reads final_aligned.obj
+ landmarks.npz from that same directory and adds ghd_fitted.obj /
ghd_coefficients.npz / metrics.json / loss_history.npz, plus a
rigid_checkpoints/ subdir with one checkpoint (.obj/.npz) per rigid-loss-
WEIGHT threshold crossed during fitting (see ghd_fit.py's
n_rigid_checkpoints), and a sanity/ subdir with the matching sanity images
(one per threshold + a final one) -- kept separate from the checkpoint
files for easier monitoring. Both stages write to the same --save-dir.

conda activate new
python ghd/fitting/run_case.py --case-dir "/path/to/case"
python ghd/fitting/run_case.py --cases-root "/path/to/all_cases" --all
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ghd.fitting.alignment import (
    load_case_data, load_canonical_data, fit_alignment,
    render_sanity_images, save_stage1_checkpoints, DOME_LOADERS,
)
from ghd.fitting.ghd_fit import ghd_fit
from ghd.fitting.skeleton_alignment import fit_skeleton_alignment
import numpy as np


def case_is_done(save_dir):
    save_dir = Path(save_dir)
    return (save_dir / "ghd_coefficients.npz").exists() and (save_dir / "ghd_fitted.obj").exists()


def run_one_case(case_dir, save_dir, args):
    case_dir, save_dir = Path(case_dir), Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Stage 1 (alignment): {case_dir.name} ===", flush=True)
    case = load_case_data(case_dir, n_surface_samples=args.n_surface_samples,
                          extrude_length=args.extrude_length,
                          dome_points_loader=DOME_LOADERS[args.dome_source])
    canonical_dir = (args.bifurcated_canonical_dir if case["aneurysm_type"] == 0
                     else args.sidewall_canonical_dir)
    canonical = load_canonical_data(canonical_dir, n_surface_samples=args.n_surface_samples)
    print(f"  aneurysm_type={case['aneurysm_type']}  case volume={case['volume']:.2f} mm^3  "
          f"canonical volume={canonical['volume']:.2f} mm^3", flush=True)

    if args.alignment_mode == "skeleton":
        transform, log = fit_skeleton_alignment(case, canonical, device=args.device)
        result = {
            "R": transform.rotation_matrix().detach().cpu().numpy(),
            "s": transform.scale().detach().cpu().item(),
            "t": transform.t.detach().cpu().numpy(),
            "history": [log], "final_loss": log,
            "case_dir": case["case_dir"], "use_fallback": case["use_fallback"],
        }
    else:
        transform, result = fit_alignment(
            case, canonical, epochs=args.align_epochs, lr=args.align_lr,
            w_centerline=args.w_centerline, w_surface=args.w_surface, w_endpoint=args.w_endpoint,
            w_volume=args.w_volume, w_dome=args.w_dome, device=args.device,
            branch_focus_id=args.branch_focus_id,
            branch_focus_centerline_weight=args.branch_focus_centerline_weight,
            branch_focus_endpoint_weight=args.branch_focus_endpoint_weight,
            centerline_matching=args.centerline_matching,
            centerline_n_resample=args.centerline_n_resample,
        )
    np.save(save_dir / "alignment_result.npy", result, allow_pickle=True)
    save_stage1_checkpoints(save_dir, transform, case, canonical)
    render_sanity_images(save_dir, transform, case, canonical)

    print(f"=== Stage 2 (GHD fitting): {case_dir.name} ===", flush=True)
    extra_rigid_checkpoint_values = tuple(float(v) for v in args.extra_rigid_checkpoint_values.split(",") if v.strip())
    metrics = ghd_fit(
        stage1_dir=save_dir, canonical_root=args.canonical_root, out_dir=save_dir,
        aneurysm_type=case["aneurysm_type"], n_iter=args.n_iter, lr=args.fit_lr, eta_min=args.eta_min,
        device=args.device, lambda_chamfer_n1=args.lambda_chamfer_n1, lambda_laplacian=args.lambda_laplacian,
        lambda_rigid_start=args.lambda_rigid_start, lambda_rigid_end=args.lambda_rigid_end,
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
    return metrics


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case-dir", default=None, help="Single case directory.")
    parser.add_argument("--cases-root", default=None, help="Directory containing many case subdirectories (batch mode).")
    parser.add_argument("--all", action="store_true", help="With --cases-root: run every case subdirectory, skipping ones already done (ghd_coefficients.npz + ghd_fitted.obj present).")
    parser.add_argument("--save-root", default=None,
                        help="Root under which per-case output dirs are created (save_root/<case-name>/). "
                             "Default: runtime/ghd_fit/<case-name>/, matching alignment.py's runtime/ convention.")

    parser.add_argument("--sidewall-canonical-dir", default=str(ROOT / "dataset" / "canonical" / "Sidewall"))
    parser.add_argument("--bifurcated-canonical-dir", default=str(ROOT / "dataset" / "canonical" / "Bifurcated"))
    parser.add_argument("--canonical-root", default=str(ROOT / "dataset" / "canonical"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-surface-samples", type=int, default=4000)
    parser.add_argument("--extrude-length", type=float, default=0.25)

    # Stage 1 (alignment)
    parser.add_argument("--alignment-mode", choices=["chamfer", "skeleton"], default="chamfer",
                        help="'chamfer' (default): fit_alignment's closed-form init + gradient-"
                             "descent refinement against surface/centerline/volume/dome losses. "
                             "'skeleton': skeleton_alignment.py's closed-form-only fit (branch "
                             "endpoints + dome/neck centroid), no refinement -- see that module.")
    parser.add_argument("--align-epochs", type=int, default=800)
    parser.add_argument("--align-lr", type=float, default=1e-2)
    parser.add_argument("--w-centerline", type=float, default=1.0)
    parser.add_argument("--w-surface", type=float, default=1.0)
    parser.add_argument("--w-endpoint", type=float, default=1.0)
    parser.add_argument("--w-volume", type=float, default=1.0)
    parser.add_argument("--w-dome", type=float, default=1.0,
                        help="Weight for the dome-region chamfer term, auto-skipped for cases "
                             "without dome-region data (see alignment.py's load_dome_points_from_nrrd).")
    parser.add_argument("--dome-source", choices=list(DOME_LOADERS), default="nrrd",
                        help="Which per-case dome-region loader to use: nrrd (label.nrrd "
                             "voxel label==2, ImperialNHS-style), mesh (dome_sac.ply, "
                             "AneuX_stable-style), or none (disable the dome term).")
    parser.add_argument("--branch-focus-id", type=int, default=None,
                        help="branch_ranking index (0-based) to upweight in the centerline+"
                             "endpoint terms. None (default) = plain unweighted mean.")
    parser.add_argument("--branch-focus-centerline-weight", type=float, default=1.0,
                        help="Relative weight for --branch-focus-id's CENTERLINE term vs. "
                             "every other branch. 1.0 = no effect.")
    parser.add_argument("--branch-focus-endpoint-weight", type=float, default=1.0,
                        help="Relative weight for --branch-focus-id's ENDPOINT term vs. "
                             "every other branch. 1.0 = no effect.")
    parser.add_argument("--centerline-matching", choices=["chamfer", "point2point"], default="chamfer",
                        help="chamfer (default): order-agnostic nearest-neighbor per branch. "
                             "point2point: resample both sides to --centerline-n-resample points "
                             "at uniform arc-length and use per-point MSE instead.")
    parser.add_argument("--centerline-n-resample", type=int, default=64,
                        help="Points per branch when --centerline-matching point2point.")

    # Stage 2 (GHD fitting)
    parser.add_argument("--n-iter", type=int, default=15000)
    parser.add_argument("--fit-lr", type=float, default=1e-3)
    parser.add_argument("--eta-min", type=float, default=1e-4)
    parser.add_argument("--lambda-chamfer-n1", type=float, default=0.8)
    parser.add_argument("--lambda-laplacian", type=float, default=1e-2)
    parser.add_argument("--lambda-rigid-start", type=float, default=2.0)
    parser.add_argument("--lambda-rigid-end", type=float, default=0.001)
    parser.add_argument("--rigid-decay-frac", type=float, default=0.80)
    parser.add_argument("--lambda-volume", type=float, default=1.0,
                         help="FINAL weight, reached via cosine ramp-up over --volume-ramp-frac of iterations.")
    parser.add_argument("--volume-ramp-frac", type=float, default=0.5,
                         help="Fraction of iterations over which the volume loss weight ramps 0 -> "
                              "--lambda-volume (cosine), then holds at full weight.")
    parser.add_argument("--volume-ceiling-ratio", type=float, default=1.2,
                         help="Hard ceiling on live/target volume ratio, constant throughout training.")
    parser.add_argument("--volume-target-frac-start", type=float, default=1.0,
                         help="Live volume target as a fraction of true target volume (default 1.0 -- "
                              "no artificial undershoot).")
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
    parser.add_argument("--lambda-occupancy", type=float, default=2.0,
                        help="On by default (matches the old reference pipeline's own default weight).")
    parser.add_argument("--dvs-surf-d-min", type=float, default=0.0001)
    parser.add_argument("--dvs-surf-d-max", type=float, default=0.05)
    parser.add_argument("--lambda-opening-chamfer", type=float, default=0.0,
                         help="EXPERIMENTAL, opening-index-dependent (0=disabled default).")
    parser.add_argument("--n-opening-chamfer-pts", type=int, default=3000)
    parser.add_argument("--lambda-roundness", type=float, default=0.0,
                         help="EXPERIMENTAL, opening-index-dependent (0=disabled default).")
    parser.add_argument("--lambda-normal-alignment", type=float, default=0.0,
                         help="EXPERIMENTAL, opening-index-dependent (0=disabled default), aka 'ring tangent'.")
    parser.add_argument("--lambda-geodesic", type=float, default=0.0,
                         help="EXPERIMENTAL, opening-index-dependent (0=disabled default).")
    parser.add_argument("--geodesic-weight", type=float, default=1.0)
    parser.add_argument("--geodesic-mask-eps", type=float, default=0.05)
    parser.add_argument("--n-rigid-checkpoints", type=int, default=10,
                         help="Snapshot checkpoints/sanity images at this many evenly-spaced points along "
                              "the rigid-loss WEIGHT's own decay, instead of a fixed iteration cadence.")
    parser.add_argument("--extra-rigid-checkpoint-values", type=str, default="0.1,0.05,0.01",
                         help="Comma-separated additional guaranteed rigid-weight checkpoint values, merged "
                              "with the evenly-spaced --n-rigid-checkpoints ones.")
    parser.add_argument("--rigid-pose-freeze-frac", type=float, default=0.0,
                         help="Freeze Stage 2's residual rotation (w) for this leading fraction "
                              "of n_iter, so it can't rotate away a correct Stage-1 branch "
                              "correspondence before phi gets to use it. 0.0 (default) = no freeze.")
    parser.add_argument("--log-every", type=int, default=200)
    return parser


def main():
    args = build_parser().parse_args()
    if not args.case_dir and not args.cases_root:
        raise ValueError("Provide --case-dir (single case) or --cases-root (+ --all for batch mode).")

    save_root = Path(args.save_root) if args.save_root else ROOT / "runtime" / "ghd_fit"

    if args.case_dir:
        case_dir = Path(args.case_dir)
        save_dir = save_root / case_dir.name
        run_one_case(case_dir, save_dir, args)
        return

    cases_root = Path(args.cases_root)
    case_dirs = sorted(p for p in cases_root.iterdir() if p.is_dir())
    print(f"Found {len(case_dirs)} case(s) under {cases_root}")
    for case_dir in case_dirs:
        save_dir = save_root / case_dir.name
        if case_is_done(save_dir):
            print(f"Skipping {case_dir.name} (already done)")
            continue
        try:
            run_one_case(case_dir, save_dir, args)
        except Exception as e:
            print(f"FAILED {case_dir.name}: {e}", flush=True)


if __name__ == "__main__":
    main()
