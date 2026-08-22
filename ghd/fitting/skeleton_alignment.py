"""skeleton_alignment.py -- ALTERNATIVE alignment strategy: closed-form
similarity fit (Umeyama) using ONLY the branch-endpoint / dome-centroid
"skeleton" correspondence -- no mesh-surface chamfer/centerline/volume/
dome LOSS REFINEMENT at all (dome points are still loaded/available for
diagnostics, just not optimized against).

Ported from AneuSeg/prep_case_for_ghd.py's design philosophy: both its
bifurcation branch (Procrustes on 3 branch-direction vectors against a
canonical reference) and its sidewall branch (hand-built Z/X/Y frame from
vessel-axis tangent + dome direction) share the same core idea despite
looking different in code -- align using only a handful of skeleton
landmarks (branch endpoints + dome/neck position), completely ignoring
the actual mesh surface. This is a much cheaper, fully deterministic fit
with no local-optimum risk from chamfer's nearest-neighbor correspondence
search.

ghd/fitting/alignment.py's fit_alignment() already computes this exact
closed-form fit (umeyama_similarity on branch_endpoints + aneurysm/neck
centroid) as its INITIALIZATION, then refines it further with 800 epochs
of gradient descent against surface/centerline/volume/dome losses. This
script stops right there and uses the closed-form fit as the FINAL
answer -- an option for cases where that refinement occasionally makes
things worse (chamfer's nearest-neighbor matching can latch onto a wrong
correspondence and pull the transform somewhere worse than its own
starting point; empirically the gradient-refined result's quality can be
inconsistent case to case).

Produces the SAME THREE output files as the standard pipeline
(final_aligned.obj, landmarks.npz, centerline.vtp) by reusing
save_stage1_checkpoints/render_sanity_images UNCHANGED, so downstream
tooling (ghd_fit.py, run_case.py, fit_with_arap_init.py, manage.py)
can't tell the difference -- it's a drop-in alternative source of
`transform`.

Does NOT modify ghd/fitting/alignment.py -- imports its loaders/savers/
umeyama_similarity/SimilarityTransform/alignment_loss as-is.

conda activate new
python ghd/fitting/skeleton_alignment.py --case-dir /path/to/case
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ghd.fitting.alignment import (
    load_case_data, load_canonical_data, umeyama_similarity, SimilarityTransform,
    save_stage1_checkpoints, render_sanity_images, alignment_loss, DOME_LOADERS,
)


def fit_skeleton_alignment(case, canonical, device="cpu"):
    """Closed-form-only similarity transform (R, isotropic scale, t) from
    branch_endpoints + aneurysm/neck centroid correspondences -- the exact
    same point set fit_alignment() uses for its own closed-form init, just
    used here as the FINAL transform instead of a gradient-descent starting
    point. No mesh surface, no centerline chamfer, no volume, no dome loss
    ever gets optimized against.

    Returns (transform, log) -- log is alignment_loss's diagnostic dict,
    computed but NEVER optimized against (reporting only, so callers can
    compare against the gradient-refined pipeline's own logged losses)."""
    if case["aneurysm_type"] != canonical["aneurysm_type"] and not (
        case["aneurysm_type"] in (1, 2) and canonical["aneurysm_type"] in (1, 2)
    ):
        raise ValueError(f"Case aneurysm_type={case['aneurysm_type']} doesn't match "
                         f"canonical aneurysm_type={canonical['aneurysm_type']}.")
    if len(case["branch_segments"]) != len(canonical["branch_segments"]):
        raise ValueError(f"Case has {len(case['branch_segments'])} branch(es), canonical has "
                         f"{len(canonical['branch_segments'])} -- can't establish correspondence.")

    src_pts = np.vstack([case["branch_endpoints"], case["aneurysm_centroid"][None]])
    dst_pts = np.vstack([canonical["branch_endpoints"], canonical["neck_centroid"][None]])
    R0, s0, t0 = umeyama_similarity(src_pts, dst_pts)
    transform = SimilarityTransform(R_init=R0, s_init=s0, t_init=t0, device=device)

    _, log = alignment_loss(transform, case, canonical)
    return transform, log


def run_skeleton_alignment(case_dir, canonical_root=None, out_dir=None, device="cpu",
                           dome_source="nrrd"):
    case_dir = Path(case_dir)
    canonical_root = Path(canonical_root) if canonical_root else ROOT / "dataset" / "canonical"
    out_dir = Path(out_dir) if out_dir else ROOT / "runtime" / "skeleton_align_test" / case_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    case = load_case_data(case_dir, dome_points_loader=DOME_LOADERS[dome_source])
    atype = case["aneurysm_type"]
    canonical_dir = canonical_root / ("Bifurcated" if atype == 0 else "Sidewall")
    canonical = load_canonical_data(canonical_dir)

    transform, log = fit_skeleton_alignment(case, canonical, device=device)
    print(f"  [skeleton, closed-form only] {log}", flush=True)

    save_stage1_checkpoints(out_dir, transform, case, canonical)
    render_sanity_images(out_dir, transform, case, canonical)
    print(f"  Done -> {out_dir}", flush=True)
    return transform, case, canonical


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--canonical-root", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dome-source", choices=list(DOME_LOADERS), default="nrrd")
    args = parser.parse_args()
    run_skeleton_alignment(args.case_dir, args.canonical_root, args.out_dir, args.device,
                           dome_source=args.dome_source)


if __name__ == "__main__":
    main()
