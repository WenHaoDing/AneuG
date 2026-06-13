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
    ):
        self.root = Path(root)
        self.max_branches = max_branches
        self.max_points_per_branch = max_points_per_branch
        self.max_local_points = max_points_per_branch - 1
        self.point_interval = point_interval
        self.min_arc_length = min_arc_length
        self.normalize = normalize
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        local_points = torch.zeros(self.max_branches, self.max_local_points, 3)
        start_points = torch.zeros(self.max_branches, 3)
        branch_direction = torch.zeros(self.max_branches, 3)
        branch_length = torch.zeros(self.max_branches, dtype=torch.long)
        point_mask = torch.zeros(self.max_branches, self.max_local_points, dtype=torch.bool)
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

            local_points[branch_idx, :n_local] = local
            start_points[branch_idx] = start
            point_mask[branch_idx, :n_local] = True
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
