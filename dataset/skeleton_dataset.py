from pathlib import Path

import numpy as np
import torch


class VesselSkeletonDataset(torch.utils.data.Dataset):
    """
    Transformer-ready vessel skeleton dataset.

    Branches are resampled at a fixed arc-length interval (`point_interval`),
    so curve points are evenly spaced in physical units rather than fixed in
    count. Branches longer than `max_points_per_branch` (in resampled points)
    have their effective interval enlarged so they still fit within the cap.

    Returns:
      local_points:    float32 [max_branches, max_points_per_branch - 1, 3]
                       resampled points 1..N relative to the branch start,
                       zero-padded beyond branch_length.
      start_points:    float32 [max_branches, 3]
      branch_direction: float32 [max_branches, 3]
                       unit vector from raw point 0 to raw point 1; zero if
                       fewer than 2 raw points exist for that branch.
      branch_length:   int64   [max_branches]
                       number of valid entries in local_points (0 for
                       short/pad branches).
      point_mask:      bool    [max_branches, max_points_per_branch - 1]
      token_mask:      bool    [max_branches * (max_points_per_branch - 1)]
      point_weights:   float32 [max_branches, max_points_per_branch - 1]
                       per-point loss weight, larger where the curve bends more.
                       Within each branch the valid weights sum to the branch's
                       point count (mean weight 1), so a weighted loss keeps the
                       same scale as an unweighted one. 0 on pad/short entries.
      token_weights:   float32 [max_branches * (max_points_per_branch - 1)]
                       point_weights flattened (parallel to token_mask).
      branch_mask:     bool    [max_branches]
      branch_state:    int64   [max_branches], 0=pad, 1=valid, 2=short
      arc_length:      float32 [max_branches]
      aneurysm_type:   int64   scalar
      canonical_type:  str
      phi:             float32 [144, 3]
      scale:           float32 scalar, exp(log_scale)
    """

    def __init__(
        self,
        root,
        cases=None,
        max_branches=3,
        max_points_per_branch=128,
        point_interval=0.1,
        min_arc_length=1.0,
        normalize=True,
        curvature_weight=1.0,
    ):
        self.root = Path(root)
        self.max_branches = max_branches
        self.max_points_per_branch = max_points_per_branch
        self.max_local_points = max_points_per_branch - 1
        self.point_interval = point_interval
        self.min_arc_length = min_arc_length
        self.normalize = normalize
        # strength of the curvature boost in point_weights (0 = uniform weights)
        self.curvature_weight = curvature_weight
        self.paths = self._collect_paths(cases)
        self.samples = self._load()
        self.point_mean, self.point_std = self._fit_norm()

    def _collect_paths(self, cases):
        if cases is None:
            return sorted(
                path for path in self.root.glob("*.npy")
                if not path.name.endswith("_sanity.npy")
            )
        return [self.root / f"{case}.npy" for case in cases]

    def _load(self):
        samples = []
        for path in self.paths:
            if not path.exists():
                continue
            chk = np.load(path, allow_pickle=True).item()
            centerline = chk["clipped_centerline"]
            if centerline.get("coordinate_system") != "ghd":
                raise ValueError(f"Expected GHD-space centerline in {path}")
            samples.append({
                "case": chk["case"],
                "aneurysm_type": int(chk["aneurysm_type"]),
                "canonical_type": chk["canonical_type"],
                "branch_points": [
                    np.asarray(points, dtype=np.float32)
                    for points in centerline["branch_points"]
                ],
                "phi": np.asarray(chk["ghd"]["phi"], dtype=np.float32),
                "scale": float(np.exp(np.asarray(chk["ghd"]["log_scale"], dtype=np.float64))),
            })
        if not samples:
            raise RuntimeError(f"No processed skeleton checkpoints found in {self.root}")
        return samples

    def _fit_norm(self):
        if not self.normalize:
            return None, None
        points = []
        for sample in self.samples:
            for branch in sample["branch_points"][: self.max_branches]:
                if len(branch) >= 2 and self._arc_length(branch) >= self.min_arc_length:
                    resampled = self._resample_by_interval(
                        branch, self.point_interval, self.max_points_per_branch
                    )
                    points.append(resampled[1:] - resampled[0])
        if not points:
            return torch.zeros(1, 3), torch.ones(1, 3)
        points = np.concatenate(points, axis=0)
        return (
            torch.as_tensor(points.mean(0, keepdims=True), dtype=torch.float32),
            torch.as_tensor(points.std(0, keepdims=True) + 0.01, dtype=torch.float32),
        )

    @staticmethod
    def _arc_length(points):
        if len(points) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())

    @staticmethod
    def _resample_by_interval(points, point_interval, max_points):
        """Resample `points` at a fixed arc-length interval.

        Returns at least 2 points (the endpoints). If the natural sample
        count (total_arc_length / point_interval + 1) exceeds `max_points`,
        the effective interval is enlarged so the branch fits within
        `max_points` resampled points.
        """
        points = np.asarray(points, dtype=np.float32)
        seg_len = np.linalg.norm(np.diff(points, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(seg_len)])
        total = arc[-1]

        n_points = int(round(total / point_interval)) + 1
        n_points = max(2, min(n_points, max_points))

        target = np.linspace(0.0, total, n_points)
        return np.stack([
            np.interp(target, arc, points[:, dim])
            for dim in range(3)
        ], axis=1).astype(np.float32)

    @staticmethod
    def _branch_direction(points):
        if len(points) < 2:
            return np.zeros(3, dtype=np.float32)
        diff = points[1] - points[0]
        norm = np.linalg.norm(diff)
        if norm <= 1e-8:
            return np.zeros(3, dtype=np.float32)
        return (diff / norm).astype(np.float32)

    @staticmethod
    def _curvature(points):
        """Per-point spatial curvature proxy of a polyline (N, 3), shape (N,) >= 0.

        Uses the second-difference magnitude |p[i-1] - 2 p[i] + p[i+1]|; since the
        points are resampled at a (within-branch) uniform arc-length interval this
        is proportional to the geometric curvature. Endpoints replicate their
        nearest interior value.
        """
        points = np.asarray(points, dtype=np.float64)
        n = len(points)
        curv = np.zeros(n)
        if n >= 3:
            second = points[2:] - 2.0 * points[1:-1] + points[:-2]   # [n-2, 3]
            curv[1:-1] = np.linalg.norm(second, axis=1)
            curv[0], curv[-1] = curv[1], curv[-2]
        return curv

    def _curvature_weights(self, resampled, n_local):
        """Loss weights for resampled[1:] (length n_local), larger where curvature
        is high, normalized so the weights sum to n_local (mean weight 1)."""
        curv = self._curvature(resampled)[1:]                       # weights align with local_points
        mean_c = curv.mean()
        if mean_c > 1e-12:
            raw = 1.0 + self.curvature_weight * (curv / mean_c)
        else:
            raw = np.ones_like(curv)                                # straight branch → uniform
        raw *= n_local / raw.sum()                                  # sum(weights) == n_local
        return raw.astype(np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        local_points = torch.zeros(self.max_branches, self.max_local_points, 3)
        start_points = torch.zeros(self.max_branches, 3)
        branch_direction = torch.zeros(self.max_branches, 3)
        branch_length = torch.zeros(self.max_branches, dtype=torch.long)
        point_mask = torch.zeros(self.max_branches, self.max_local_points, dtype=torch.bool)
        point_weights = torch.zeros(self.max_branches, self.max_local_points)
        branch_mask = torch.zeros(self.max_branches, dtype=torch.bool)
        branch_state = torch.zeros(self.max_branches, dtype=torch.long)
        arc_length = torch.zeros(self.max_branches)

        for branch_idx, points in enumerate(sample["branch_points"][: self.max_branches]):
            arc_length[branch_idx] = self._arc_length(points)
            if len(points) > 0:
                start_points[branch_idx] = torch.as_tensor(points[0], dtype=torch.float32)
            branch_direction[branch_idx] = torch.as_tensor(
                self._branch_direction(points), dtype=torch.float32
            )

            if arc_length[branch_idx] < self.min_arc_length:
                branch_state[branch_idx] = 2
                continue

            resampled = self._resample_by_interval(
                points, self.point_interval, self.max_points_per_branch
            )
            start = torch.as_tensor(resampled[0], dtype=torch.float32)
            local = torch.as_tensor(resampled[1:] - resampled[0], dtype=torch.float32)
            n_local = local.shape[0]
            if self.normalize:
                local = (local - self.point_mean) / self.point_std

            weights = torch.as_tensor(
                self._curvature_weights(resampled, n_local), dtype=torch.float32
            )
            local_points[branch_idx, :n_local] = local
            start_points[branch_idx] = start
            point_mask[branch_idx, :n_local] = True
            point_weights[branch_idx, :n_local] = weights
            branch_mask[branch_idx] = True
            branch_state[branch_idx] = 1
            branch_length[branch_idx] = n_local

        return {
            "case": sample["case"],
            "local_points": local_points,
            "start_points": start_points,
            "branch_direction": branch_direction,
            "branch_length": branch_length,
            "point_mask": point_mask,
            "token_mask": point_mask.flatten(),
            "point_weights": point_weights,
            "token_weights": point_weights.flatten(),
            "branch_mask": branch_mask,
            "branch_state": branch_state,
            "arc_length": arc_length,
            "aneurysm_type": torch.as_tensor(sample["aneurysm_type"], dtype=torch.long),
            "canonical_type": sample["canonical_type"],
            "phi": torch.as_tensor(sample["phi"], dtype=torch.float32),
            "scale": torch.as_tensor(sample["scale"], dtype=torch.float32),
        }

    def denormalize_local_points(self, points):
        if not self.normalize:
            return points
        return points * self.point_std.to(points.device) + self.point_mean.to(points.device)


def visualize_sample(dataset, idx, save_path=None):
    """Sanity-check one dataset item.

    Reconstructs absolute branch points from `start_points` +
    `denormalize_local_points(local_points)` and overlays them on the
    GHD mesh reconstructed from `phi`, mirroring
    `dataset.preprocess.sanity_check_processed_checkpoint`.
    """
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from dataset.preprocess_ImperialNHS import _reconstruct_ghd_numpy, _set_axes_equal

    item = dataset[idx]
    checkpoint = {
        "aneurysm_type": int(item["aneurysm_type"]),
        "ghd": {"phi": item["phi"].numpy()},
    }
    verts, faces = _reconstruct_ghd_numpy(checkpoint, denormalize_shape=True)

    local_points = dataset.denormalize_local_points(item["local_points"])
    branch_points = []
    for branch_idx in range(dataset.max_branches):
        if not item["branch_mask"][branch_idx]:
            continue
        n = int(item["branch_length"][branch_idx])
        start = item["start_points"][branch_idx].numpy()
        offsets = local_points[branch_idx, :n].numpy()
        pts = np.concatenate([start[None, :], start[None, :] + offsets], axis=0)
        branch_points.append(pts)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces,
                    color="lightgray", edgecolor="none", alpha=0.25)

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(branch_points), 1)))
    for i, pts in enumerate(branch_points):
        color = colors[i % len(colors)]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], linewidth=2.0, color=color, label=f"branch {i}")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=8, color=color)
        ax.scatter(*pts[0], s=40, marker="x", color=color)

    ax.set_title(f"{item['case']} (type {int(item['aneurysm_type'])}: {item['canonical_type']})")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    _set_axes_equal(ax, verts, *branch_points)
    ax.legend(loc="upper right")
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
    plt.close(fig)
    return fig


if __name__ == "__main__":
    import sys
    import os

    ROOT = '/media/yaplab2/HDD Storage/wenhao/AneuG/dataset'
    
    dataset = VesselSkeletonDataset(os.path.join(ROOT, 'processed'))
    print(f"Loaded {len(dataset)} samples.")

    out_dir = os.path.join(ROOT, 'processed', 'sanity_branches')
    for idx in range(min(5, len(dataset))):
        case = dataset.samples[idx]["case"]
        save_path = os.path.join(out_dir, f"{case}_sanity_dataset.png")
        visualize_sample(dataset, idx, save_path=save_path)
        print(f"Saved sanity figure for {case} -> {save_path}")



"""
conda activate new
python dataset/skeleton_dataset.py

"""
