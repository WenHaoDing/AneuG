"""
Vessel skeleton point-sequence dataset — unclipped-centerline variant.

Parallel to dataset/skeleton_dataset.py (which reads clipped branch stubs,
start = pts[0]) — feeds the point-sequence transformer model family
(models/branch_transformer_claydoll.py) rather than the Fourier-coefficient
MLP-VAE family (dataset/skeleton_dataset_fourier_claydoll.py). Reads
preprocess_*_claydoll.py's "unclipped_centerline" field instead of
"clipped_centerline".

_load() is overridden to read "unclipped_centerline" and to cap every branch
to at most MAX_LENGTH_MM of arc length from its own start (see
_clip_to_max_length below) — regardless of source, since preprocessing's own
cap differs (preprocess_AneuX_claydoll.py clips to 15mm at preprocessing time,
preprocess_ImperialNHS_claydoll.py has no cap at all, so real unclipped
branches otherwise range up to >100mm). Without this cap, VesselSkeletonDataset's
_resample_by_interval (unchanged, reused as-is) widens point_interval per branch
so it still fits within max_points_per_branch — but that widening applies
independently per branch, so a 15mm and a 100mm branch both end up resampled
to the same 127 points, just at wildly different (and model-invisible) physical
spacing, and >75% of real branches were saturating the point cap with no
headroom left for the length-prediction head to learn anything from. Capping
every branch to the same MAX_LENGTH_MM first bounds that worst-case spacing
(127 points always span at most MAX_LENGTH_MM, at point_interval as fine as
0.1mm for shorter branches) instead of leaving it up to the longest outlier.
VesselSkeletonDataset's __getitem__/_fit_norm already treat an empty (len 0)
branch_points entry as "no branch" (arc_length=0 -> short/pad branch_state,
no crash) — so a failed-match branch (None in the raw checkpoint) only needs
converting to an empty array here, not any deeper special-casing.

The base class's own start_points/branch_direction fields (computed from real
data, unmodified) are used ONLY as the claydoll model's start_points_loss
regression target at train time — NOT fed to the model as input conditioning.
Conditioning instead uses models/branch_transformer_claydoll.py's own
predicted per-branch start point (see that module's docstring for why: an
unclipped branch's real start has no fixed generation-time landmark the way a
mesh-opening clip point does).

conda activate new
python dataset/skeleton_dataset_claydoll.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.skeleton_dataset import VesselSkeletonDataset

MAX_LENGTH_MM = 20.0


def _clip_to_max_length(pts, max_length_mm):
    """Truncate pts (oriented so pts[0] is nearest the aneurysm) to at most
    max_length_mm of arc length from pts[0]. No-op if already within budget.

    Duplicated from dataset/preprocess_AneuX_claydoll.py's helper of the same
    name (kept local rather than imported — that module is a preprocessing
    script, not a library import target) — same truncate-from-the-far-end
    logic, applied here at dataset-load time instead of preprocessing time so
    it covers every source uniformly regardless of what each preprocessing
    script's own cap (if any) already did.
    """
    if len(pts) < 2:
        return pts
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] <= max_length_mm:
        return pts
    cut_idx = max(int(np.searchsorted(cum, max_length_mm)) + 1, 2)
    return pts[:cut_idx]


class VesselSkeletonDatasetClaydoll(VesselSkeletonDataset):
    """VesselSkeletonDataset over unclipped (dome-to-tip) branches, each
    capped to at most max_length_mm of arc length from its own start."""

    def __init__(self, root, cases=None, max_branches=3, max_points_per_branch=128,
                 point_interval=None, min_arc_length=1.0, normalize=True, curvature_weight=1.0,
                 max_length_mm=MAX_LENGTH_MM):
        self.max_length_mm = max_length_mm   # read by _load(), so set before super().__init__() calls it
        if point_interval is None:
            # Size the resample interval so a branch at exactly max_length_mm
            # uses the full max_points_per_branch-1 points. Without this, the
            # base class's own 0.1mm default already saturates the point cap
            # (widens past it) for any branch beyond 0.1 * (max_points_per_branch-1)
            # = 12.7mm — well under the ~15mm median real unclipped branch —
            # so >75% of real branches all get the SAME branch_length=127
            # target regardless of whether they're 13mm or max_length_mm long,
            # leaving the length-prediction head almost nothing to learn from.
            # Tying the default interval to max_length_mm instead means only
            # branches actually at/near the cap saturate; shorter ones get a
            # branch_length that varies meaningfully with their real extent.
            point_interval = max_length_mm / (max_points_per_branch - 1)
        super().__init__(root, cases=cases, max_branches=max_branches,
                         max_points_per_branch=max_points_per_branch, point_interval=point_interval,
                         min_arc_length=min_arc_length, normalize=normalize, curvature_weight=curvature_weight)

    def _load(self):
        samples = []
        for path in self.paths:
            if not path.exists():
                continue
            chk = np.load(path, allow_pickle=True).item()
            centerline = chk.get("unclipped_centerline")
            if centerline is None:
                continue
            if centerline.get("coordinate_system") != "ghd":
                raise ValueError(f"Expected GHD-space centerline in {path}")
            samples.append({
                "case": chk["case"],
                "aneurysm_type": int(chk["aneurysm_type"]),
                "canonical_type": chk["canonical_type"],
                "branch_points": [
                    np.zeros((0, 3), dtype=np.float32) if points is None
                    else _clip_to_max_length(np.asarray(points, dtype=np.float32), self.max_length_mm)
                    for points in centerline["branch_points"]
                ],
                "phi": np.asarray(chk["ghd"]["phi"], dtype=np.float32),
                "scale": float(np.exp(np.asarray(chk["ghd"]["log_scale"], dtype=np.float64))),
            })
        if not samples:
            raise RuntimeError(f"No processed unclipped-centerline checkpoints found in {self.root}")
        return samples


if __name__ == "__main__":
    from dataset.skeleton_dataset import visualize_sample

    PROCESSED_ROOT = ROOT / "dataset" / "processed_claydoll"
    dataset = VesselSkeletonDatasetClaydoll(PROCESSED_ROOT)
    print(f"Loaded {len(dataset)} samples.")
    print(f"point_mean={dataset.point_mean.tolist()}  point_std={dataset.point_std.tolist()}")

    arc_all = np.array([
        float(item["arc_length"][b])
        for idx in range(len(dataset))
        for item in [dataset[idx]]
        for b in range(dataset.max_branches)
        if item["branch_mask"][b]
    ])
    print(f"Unclipped branch arc length (mm) over {len(arc_all)} branches: "
          f"mean={arc_all.mean():.2f}  median={np.median(arc_all):.2f}  max={arc_all.max():.2f}")

    out_dir = PROCESSED_ROOT / "sanity_branches_claydoll"
    for idx in range(min(5, len(dataset))):
        case = dataset.samples[idx]["case"]
        save_path = out_dir / f"{case}_sanity_dataset.png"
        visualize_sample(dataset, idx, save_path=save_path)
        print(f"Saved sanity figure for {case} -> {save_path}")
