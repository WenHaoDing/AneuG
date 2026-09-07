"""
Fourier-series vessel skeleton dataset.

Subclasses VesselSkeletonDataset (reusing its checkpoint loading) and replaces
the point-sequence branch representation with a compact set of sine-series
coefficients. Each branch's fit is computed once and cached at construction.

Per-branch encoding
--------------------
1. Resample the branch centerline to `fit_points` points, uniform in arc length.
2. start = p[0], end = p[-1]. Subtract the straight chord:
       line(s)     = start + (end - start) * (s / π)
       relative(s) = p(s) - line(s)                       # 0 at both endpoints
   where s is the normalized arc-length parameter, s ∈ [0, π].
3. Fit each coordinate d ∈ {x, y, z} with a sine-only (DST-I) series of k harmonics:
       relative_d(s) = Σ_{n=1..k} c[d, n] · sin(n · s)
   On [0, π] this is the standard Fourier sine basis: every sin(n·s) is zero at
   s = 0 and s = π, and sin(s) is a single hump — so a one-sided bulge (the
   usual branch deviation, incl. the proximal hook) is captured with few terms.
   (Using [0, 2π] instead makes every sin(n·s) antisymmetric about the midpoint,
   i.e. orthogonal to a symmetric bulge — it cannot represent it. Hence [0, π].)
4. The fit is an ordinary linear least-squares solve:
       A[i, n] = sin(n · s_i)                 # [fit_points, k]
       C = np.linalg.lstsq(A, relative)       # [k, 3]

Reconstruct:  p(s) = start + (end - start) * (s / π) + Σ_n C[n] · sin(n·s).
A branch is fully described by (start[3], end[3], coeffs[k, 3]); arc length is
kept as an extra scalar feature.
"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.skeleton_dataset import VesselSkeletonDataset

S_MAX = np.pi   # arc-length parameter range [0, S_MAX]; DST-I sine basis


# ── branch Fourier fitting helpers ──────────────────────────────────────────

def resample_uniform(points, n):
    """Resample a polyline (M, 3) to n points, uniform in arc length."""
    points = np.asarray(points, dtype=float)
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total <= 0:
        return np.repeat(points[:1], n, axis=0)
    target = np.linspace(0.0, total, n)
    return np.stack([np.interp(target, cum, points[:, d]) for d in range(3)], axis=1)


def fit_branch_fourier(points, k=8, fit_points=128):
    """Fit a branch to sine-series coefficients.

    Returns
    -------
    start : (3,) float32
    end   : (3,) float32
    coeffs: (k, 3) float32   sine coefficients c[n, d], n = 1..k
    rms   : float            RMS fit residual (physical units), for diagnostics
    """
    pts = resample_uniform(points, fit_points)            # [N, 3], uniform arc length
    N = len(pts)
    start, end = pts[0].copy(), pts[-1].copy()

    frac = np.linspace(0.0, 1.0, N)                        # arc-length fraction
    line = start[None] + (end - start)[None] * frac[:, None]
    rel = pts - line                                      # [N, 3], zero at both ends

    s = S_MAX * frac                                       # [N], s ∈ [0, π]
    A = np.sin(np.outer(s, np.arange(1, k + 1)))          # [N, k]
    coeffs, *_ = np.linalg.lstsq(A, rel, rcond=None)      # [k, 3]

    rms = float(np.sqrt(np.mean((A @ coeffs - rel) ** 2)))
    return start.astype(np.float32), end.astype(np.float32), coeffs.astype(np.float32), rms


def reconstruct_branch(start, end, coeffs, n_points=128):
    """Inverse of fit_branch_fourier: coefficients → 3D polyline."""
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    coeffs = np.asarray(coeffs, dtype=float)              # [k, 3]
    k = coeffs.shape[0]
    s = np.linspace(0.0, S_MAX, n_points)
    frac = s / S_MAX
    line = start[None] + (end - start)[None] * frac[:, None]
    A = np.sin(np.outer(s, np.arange(1, k + 1)))          # [n_points, k]
    return line + A @ coeffs


# ── dataset ─────────────────────────────────────────────────────────────────

class VesselSkeletonDatasetFourier(VesselSkeletonDataset):
    """
    Fourier-coefficient vessel skeleton dataset.

    Each item provides, per branch (padded to `max_branches`):
      start_points   : float32 [max_branches, 3]
      end_points     : float32 [max_branches, 3]
      branch_vector  : float32 [max_branches, 3]      end - start (the chord)
      fourier_coeffs : float32 [max_branches, k, 3]   sine coeffs (normalized if normalize)
      arc_length     : float32 [max_branches]         total branch arc length
      branch_mask    : bool    [max_branches]         True = valid branch
      branch_state   : int64   [max_branches]         0=pad, 1=valid, 2=short
    plus, per sample:
      case, aneurysm_type (int64), canonical_type (str),
      phi (float32 [144, 3]), scale (float32 scalar).

    Branch fits are computed once at construction and cached in self.fits.
    """

    def __init__(self, root, cases=None, max_branches=3, k=8, fit_points=128,
                 min_arc_length=1.0, max_arc_length=15.0, normalize=True):
        # Load samples via the parent; pass normalize=False so the parent skips
        # its point-sequence normalization (we don't use point_mean/point_std).
        # max_arc_length is enforced by the parent at load time, so the branches
        # this class fits are already truncated -- which is what the fixed
        # 128-point resample and k-harmonic budget need to mean the same thing
        # from one branch to the next.
        super().__init__(root, cases=cases, max_branches=max_branches,
                         min_arc_length=min_arc_length,
                         max_arc_length=max_arc_length, normalize=False)
        self.k = k
        self.fit_points = fit_points
        self.normalize = normalize
        self.fits = self._build_cache()
        self.coeff_mean, self.coeff_std = self._fit_coeff_norm()

    def _branch_is_valid(self, points):
        return len(points) >= 2 and self._arc_length(points) >= self.min_arc_length

    def _build_cache(self):
        """Fit every (valid) branch once; store start/end/coeffs/rms or None."""
        fits = []
        for sample in self.samples:
            branch_fits = []
            for points in sample["branch_points"][: self.max_branches]:
                if self._branch_is_valid(points):
                    start, end, coeffs, rms = fit_branch_fourier(points, self.k, self.fit_points)
                    branch_fits.append({"start": start, "end": end, "coeffs": coeffs, "rms": rms})
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
        coeffs = np.stack(coeffs, axis=0)                  # [B, k, 3]
        mean = torch.as_tensor(coeffs.mean(0), dtype=torch.float32)
        std = torch.as_tensor(coeffs.std(0) + 1e-4, dtype=torch.float32)
        return mean, std

    def __getitem__(self, idx):
        sample = self.samples[idx]
        fits = self.fits[idx]

        start_points     = torch.zeros(self.max_branches, 3)
        end_points       = torch.zeros(self.max_branches, 3)
        branch_vector    = torch.zeros(self.max_branches, 3)
        branch_direction = torch.zeros(self.max_branches, 3)
        fourier_coeffs   = torch.zeros(self.max_branches, self.k, 3)
        arc_lengths      = torch.zeros(self.max_branches)
        branch_mask      = torch.zeros(self.max_branches, dtype=torch.bool)
        branch_state     = torch.zeros(self.max_branches, dtype=torch.long)

        for b, points in enumerate(sample["branch_points"][: self.max_branches]):
            if len(points) > 0:
                start_points[b] = torch.as_tensor(points[0], dtype=torch.float32)
            branch_direction[b] = torch.as_tensor(self._branch_direction(points), dtype=torch.float32)
            arc_lengths[b] = self._arc_length(points)

            bf = fits[b]
            if bf is None:
                branch_state[b] = 2 if len(points) > 0 else 0
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
            "branch_direction": branch_direction,
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

    # High-contrast scheme: ground-truth branches in black, Fourier
    # reconstructions in bright saturated colours so any deviation stands out.
    bright = ["#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4"]
    msg, all_pts = [], [verts]
    for b, points in enumerate(sample["branch_points"][: dataset.max_branches]):
        if not bool(item["branch_mask"][b]):
            continue
        orig = resample_uniform(points, n_points)
        coeffs = dataset.denormalize_coeffs(item["fourier_coeffs"][b]).numpy()
        recon = reconstruct_branch(item["start_points"][b].numpy(),
                                   item["end_points"][b].numpy(), coeffs, n_points)
        rms = float(np.sqrt(np.mean(np.sum((orig - recon) ** 2, axis=1))))
        msg.append(f"b{b} rms={rms:.3f}")
        c = bright[b % len(bright)]
        # ground truth: thick black; reconstruction: bright dashed on top
        ax.plot(orig[:, 0], orig[:, 1], orig[:, 2], color="black", lw=3.0,
                solid_capstyle="round", label=f"b{b} orig")
        ax.plot(recon[:, 0], recon[:, 1], recon[:, 2], color=c, lw=2.0, ls=(0, (4, 2)),
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

    PROCESSED_ROOT = ROOT / "runtime" / "dataset" / "processed"
    K = 8

    dataset = VesselSkeletonDatasetFourier(PROCESSED_ROOT, k=K)
    print(f"Loaded {len(dataset)} samples.  coeff_dim = {dataset.get_coeff_dim()}")

    rms_all = np.array([bf["rms"] for sf in dataset.fits for bf in sf if bf is not None])
    print(f"Fourier fit RMS over {len(rms_all)} branches (k={K}): "
          f"mean={rms_all.mean():.4f}  median={np.median(rms_all):.4f}  max={rms_all.max():.4f}")

    out_dir = PROCESSED_ROOT / "sanity_fourier"
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
python dataset/skeleton_dataset_fourier.py
"""
