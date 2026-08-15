"""
Fourier-series vessel skeleton dataset — unclipped-centerline variant.

Parallel to dataset/skeleton_dataset_fourier.py (which fits clipped branch
stubs, start = pts[0]). This one reads dataset/preprocess_ImperialNHS_claydoll.py /
preprocess_AneuX_claydoll.py's "unclipped_centerline" field instead of
"clipped_centerline", and fits each branch's own real start
(unclipped_centerline["branch_start_points"][i], per branch — NOT a shared
per-case anchor, see preprocess_*_claydoll.py's docstring for why) rather than
re-deriving start from the branch's own first point. In today's data these
are the same value (branch_start_points[i] == branch_points[i][0]), but
fitting through an explicit anchor argument (fit_branch_fourier_anchored,
below) means the same fit machinery also works at generation time, when the
anchor is a *predicted* point from a model head rather than read off real data.

Per-branch encoding is otherwise identical to skeleton_dataset_fourier.py's
fit_branch_fourier: resample to fit_points, subtract the straight chord
start->end, fit a k-harmonic sine-only (DST-I) series per coordinate. See
that module's docstring for the basis rationale.

conda activate new
python dataset/skeleton_dataset_fourier_claydoll.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.skeleton_dataset import VesselSkeletonDataset
from dataset.skeleton_dataset_fourier import S_MAX, reconstruct_branch, resample_uniform


def fit_branch_fourier_anchored(points, anchor, k=8, fit_points=128):
    """Like skeleton_dataset_fourier.fit_branch_fourier, but start is forced to
    `anchor` (that branch's own real/predicted start) instead of re-deriving it
    from points[0]. In practice anchor IS points[0] at dataset-build time (see
    module docstring) — this fit should reproduce fit_branch_fourier's own
    start/end/coeffs almost exactly for real data; the separate anchor argument
    exists so the same function also works at generation time with a
    *predicted* anchor, without needing points[0] to exist at all.

    Returns
    -------
    start : (3,) float32   == anchor
    end   : (3,) float32
    coeffs: (k, 3) float32   sine coefficients c[n, d], n = 1..k
    rms   : float            RMS fit residual (physical units), for diagnostics
    """
    pts = resample_uniform(points, fit_points)             # [N, 3], uniform arc length
    N = len(pts)
    start = np.asarray(anchor, dtype=np.float32)
    end = pts[-1].copy()

    frac = np.linspace(0.0, 1.0, N)                         # arc-length fraction
    line = start[None] + (end - start)[None] * frac[:, None]
    rel = pts - line                                       # [N, 3]

    s = S_MAX * frac                                        # [N], s ∈ [0, π]
    A = np.sin(np.outer(s, np.arange(1, k + 1)))           # [N, k]
    coeffs, *_ = np.linalg.lstsq(A, rel, rcond=None)       # [k, 3]

    rms = float(np.sqrt(np.mean((A @ coeffs - rel) ** 2)))
    return start.astype(np.float32), end.astype(np.float32), coeffs.astype(np.float32), rms


# ── dataset ─────────────────────────────────────────────────────────────────

class VesselSkeletonDatasetFourierClaydoll(VesselSkeletonDataset):
    """
    Fourier-coefficient vessel skeleton dataset over UNCLIPPED branches, each
    anchored at its own real start point (per-branch, not a shared per-case
    anchor — see module docstring).

    Each item provides, per branch (padded to `max_branches`):
      start_points   : float32 [max_branches, 3]   real per-branch start —
                       this is the regression target for the future model's
                       per-branch start-point prediction head, NOT a fixed
                       conditioning input (there is no generation-time source
                       for it other than that head's own prediction)
      end_points     : float32 [max_branches, 3]
      branch_vector  : float32 [max_branches, 3]      end - start (the chord)
      fourier_coeffs : float32 [max_branches, k, 3]   sine coeffs (normalized if normalize)
      arc_length     : float32 [max_branches]         total (post-anchor) branch arc length
      branch_mask    : bool    [max_branches]         True = valid branch
      branch_state   : int64   [max_branches]         0=pad, 1=valid, 2=short/no-match
    plus, per sample:
      case, aneurysm_type (int64), canonical_type (str),
      phi (float32 [144, 3]), scale (float32 scalar).

    Branch fits are computed once at construction and cached in self.fits.
    """

    def __init__(self, root, cases=None, max_branches=3, k=8, fit_points=128,
                 min_arc_length=1.0, normalize=True):
        # Load samples via the parent; pass normalize=False so the parent skips
        # its point-sequence normalization (we don't use point_mean/point_std).
        super().__init__(root, cases=cases, max_branches=max_branches,
                         min_arc_length=min_arc_length, normalize=False)
        self.k = k
        self.fit_points = fit_points
        self.normalize = normalize
        self.fits = self._build_cache()
        self.coeff_mean, self.coeff_std = self._fit_coeff_norm()

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
                    None if points is None else np.asarray(points, dtype=np.float32)
                    for points in centerline["branch_points"]
                ],
                "branch_start_points": [
                    None if pt is None else np.asarray(pt, dtype=np.float32)
                    for pt in centerline["branch_start_points"]
                ],
                "phi": np.asarray(chk["ghd"]["phi"], dtype=np.float32),
                "scale": float(np.exp(np.asarray(chk["ghd"]["log_scale"], dtype=np.float64))),
            })
        if not samples:
            raise RuntimeError(f"No processed unclipped-centerline checkpoints found in {self.root}")
        return samples

    def _branch_is_valid(self, points, start):
        return (
            points is not None and start is not None
            and len(points) >= 2 and self._arc_length(points) >= self.min_arc_length
        )

    def _build_cache(self):
        """Fit every (valid) branch once; store start/end/coeffs/rms or None."""
        fits = []
        for sample in self.samples:
            branch_fits = []
            points_list = sample["branch_points"][: self.max_branches]
            starts_list = sample["branch_start_points"][: self.max_branches]
            for points, start in zip(points_list, starts_list):
                if self._branch_is_valid(points, start):
                    fitted = fit_branch_fourier_anchored(points, start, self.k, self.fit_points)
                    branch_fits.append({
                        "start": fitted[0], "end": fitted[1], "coeffs": fitted[2], "rms": fitted[3],
                    })
                else:
                    branch_fits.append(None)
            fits.append(branch_fits)
        return fits

    def _fit_coeff_norm(self):
        if not self.normalize:
            return None, None
        coeffs = [bf["coeffs"] for sample_fits in self.fits
                  for bf in sample_fits if bf is not None]
        if not coeffs:
            return torch.zeros(self.k, 3), torch.ones(self.k, 3)
        coeffs = np.stack(coeffs, axis=0)                   # [B, k, 3]
        mean = torch.as_tensor(coeffs.mean(0), dtype=torch.float32)
        std = torch.as_tensor(coeffs.std(0) + 1e-4, dtype=torch.float32)
        return mean, std

    def __getitem__(self, idx):
        sample = self.samples[idx]
        fits = self.fits[idx]

        start_points     = torch.zeros(self.max_branches, 3)
        end_points       = torch.zeros(self.max_branches, 3)
        branch_vector    = torch.zeros(self.max_branches, 3)
        fourier_coeffs   = torch.zeros(self.max_branches, self.k, 3)
        arc_lengths      = torch.zeros(self.max_branches)
        branch_mask      = torch.zeros(self.max_branches, dtype=torch.bool)
        branch_state     = torch.zeros(self.max_branches, dtype=torch.long)

        points_list = sample["branch_points"][: self.max_branches]
        for b, points in enumerate(points_list):
            if points is not None and len(points) > 0:
                arc_lengths[b] = self._arc_length(points)

            bf = fits[b]
            if bf is None:
                branch_state[b] = 2 if points is not None and len(points) > 0 else 0
                continue

            c = torch.as_tensor(bf["coeffs"], dtype=torch.float32)
            if self.normalize:
                c = (c - self.coeff_mean) / self.coeff_std

            start_points[b]   = torch.as_tensor(bf["start"], dtype=torch.float32)
            end_points[b]     = torch.as_tensor(bf["end"], dtype=torch.float32)
            branch_vector[b]  = end_points[b] - start_points[b]
            fourier_coeffs[b] = c
            branch_mask[b]    = True
            branch_state[b]   = 1

        return {
            "case": sample["case"],
            "start_points": start_points,
            "end_points": end_points,
            "branch_vector": branch_vector,
            "fourier_coeffs": fourier_coeffs,
            "arc_length": arc_lengths,
            "branch_mask": branch_mask,
            "branch_state": branch_state,
            "aneurysm_type": torch.as_tensor(sample["aneurysm_type"], dtype=torch.long),
            "canonical_type": sample["canonical_type"],
            "phi": torch.as_tensor(sample["phi"], dtype=torch.float32),
            "scale": torch.as_tensor(sample["scale"], dtype=torch.float32),
        }

    def denormalize_coeffs(self, coeffs):
        """coeffs: [..., k, 3] normalized → physical."""
        if not self.normalize:
            return coeffs
        return coeffs * self.coeff_std.to(coeffs.device) + self.coeff_mean.to(coeffs.device)

    def get_coeff_dim(self):
        return self.k * 3


def visualize_sample(dataset, idx, save_path=None, n_points=128):
    """Overlay original (resampled) branches and their Fourier reconstructions
    on the GHD mesh, and report per-branch RMS fitting error."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from dataset.preprocess_ImperialNHS import _reconstruct_ghd_numpy, _set_axes_equal

    sample = dataset.samples[idx]
    item = dataset[idx]
    verts, faces = _reconstruct_ghd_numpy(
        {"aneurysm_type": int(item["aneurysm_type"]), "ghd": {"phi": item["phi"].numpy()}},
        denormalize_shape=True,
    )

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces,
                    color="lightgray", edgecolor="none", alpha=0.12)

    colors = ["black", "blue", "red"]
    recon_color = "yellow"
    msg, all_pts = [], [verts]
    for b, points in enumerate(sample["branch_points"][: dataset.max_branches]):
        if not bool(item["branch_mask"][b]) or points is None:
            continue
        orig = resample_uniform(points, n_points)
        coeffs = dataset.denormalize_coeffs(item["fourier_coeffs"][b]).numpy()
        recon = reconstruct_branch(item["start_points"][b].numpy(),
                                   item["end_points"][b].numpy(), coeffs, n_points)
        rms = float(np.sqrt(np.mean(np.sum((orig - recon) ** 2, axis=1))))
        msg.append(f"b{b} rms={rms:.3f}")
        c = colors[b % len(colors)]
        # ground truth: thick, solid, per-branch color; reconstruction: yellow, dashed on top
        ax.plot(orig[:, 0], orig[:, 1], orig[:, 2], color=c, lw=3.0,
                solid_capstyle="round", label=f"b{b} orig")
        ax.plot(recon[:, 0], recon[:, 1], recon[:, 2], color=recon_color, lw=2.0, ls=(0, (4, 2)),
                label=f"b{b} fourier")
        ax.scatter(*orig[0], s=55, color=c, edgecolor="black", zorder=6)   # start
        all_pts += [orig, recon]

    ax.set_title(f"{item['case']} (k={dataset.k})  " + "  ".join(msg), fontsize=8)
    ax.set_axis_off()
    _set_axes_equal(ax, *all_pts)
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig


if __name__ == "__main__":
    import os

    PROCESSED_ROOT = ROOT / "runtime" / "dataset" / "processed_claydoll"
    K = 8

    dataset = VesselSkeletonDatasetFourierClaydoll(PROCESSED_ROOT, k=K)
    print(f"Loaded {len(dataset)} samples.  coeff_dim = {dataset.get_coeff_dim()}")

    rms_all = np.array([bf["rms"] for sf in dataset.fits for bf in sf if bf is not None])
    print(f"Fourier fit RMS over {len(rms_all)} branches (k={K}): "
          f"mean={rms_all.mean():.4f}  median={np.median(rms_all):.4f}  max={rms_all.max():.4f}")

    arc_all = np.array([
        float(item["arc_length"][b])
        for idx in range(len(dataset))
        for item in [dataset[idx]]
        for b in range(dataset.max_branches)
        if item["branch_mask"][b]
    ])
    print(f"Unclipped branch arc length (mm) over {len(arc_all)} branches: "
          f"mean={arc_all.mean():.2f}  median={np.median(arc_all):.2f}  max={arc_all.max():.2f}")

    out_dir = PROCESSED_ROOT / "sanity_fourier_claydoll"
    failed = []
    for idx in range(len(dataset)):
        case = dataset.samples[idx]["case"]
        try:
            visualize_sample(dataset, idx, save_path=os.path.join(out_dir, f"{case}_fourier.png"))
        except Exception as exc:
            failed.append((case, str(exc)))
            continue
        if (idx + 1) % 25 == 0 or idx + 1 == len(dataset):
            print(f"  [{idx + 1}/{len(dataset)}] sanity images written -> {out_dir}")
    print(f"Done: {len(dataset) - len(failed)} saved, {len(failed)} failed.")
    for case, err in failed:
        print(f"  [FAIL] {case}: {err}")

"""
conda activate new
python dataset/skeleton_dataset_fourier_claydoll.py
"""
