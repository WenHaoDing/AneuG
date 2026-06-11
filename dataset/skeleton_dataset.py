from pathlib import Path

import numpy as np
import torch


class VesselSkeletonDataset(torch.utils.data.Dataset):
    """
    Transformer-ready vessel skeleton dataset.

    Returns:
      local_points:  float32 [max_branches, num_points, 3]
      start_points:  float32 [max_branches, 3]
      point_mask:    bool    [max_branches, num_points]
      token_mask:    bool    [max_branches * num_points]
      branch_mask:   bool    [max_branches]
      branch_state:  int64   [max_branches], 0=pad, 1=valid, 2=short
      branch_ids:    int64   [max_branches]
      arc_length:    float32 [max_branches]
      aneurysm_type: int64   scalar
      canonical_type: str
    """

    def __init__(
        self,
        root,
        cases=None,
        max_branches=3,
        num_points=128,
        min_arc_length=0.0,
        normalize=True,
    ):
        self.root = Path(root)
        self.max_branches = max_branches
        self.num_points = num_points
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
                "branch_ids": np.asarray(centerline["branch_ids"], dtype=np.int64),
                "branch_points": [
                    np.asarray(points, dtype=np.float32)
                    for points in centerline["branch_points"]
                ],
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
                if len(branch) > 0 and self._arc_length(branch) >= self.min_arc_length:
                    points.append(self._resample(branch, self.num_points + 1)[1:] - branch[0])
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
    def _resample(points, num_points):
        points = np.asarray(points, dtype=np.float32)
        if len(points) == num_points:
            return points
        if len(points) == 0:
            return np.zeros((num_points, 3), dtype=np.float32)
        if len(points) == 1:
            return np.repeat(points, num_points, axis=0)

        seg_len = np.linalg.norm(np.diff(points, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(seg_len)])
        if arc[-1] <= 1e-8:
            return np.repeat(points[:1], num_points, axis=0)
        target = np.linspace(0.0, arc[-1], num_points)
        return np.stack([
            np.interp(target, arc, points[:, dim])
            for dim in range(3)
        ], axis=1).astype(np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        local_points = torch.zeros(self.max_branches, self.num_points, 3)
        start_points = torch.zeros(self.max_branches, 3)
        point_mask = torch.zeros(self.max_branches, self.num_points, dtype=torch.bool)
        branch_mask = torch.zeros(self.max_branches, dtype=torch.bool)
        branch_state = torch.zeros(self.max_branches, dtype=torch.long)
        branch_ids = torch.full((self.max_branches,), -1, dtype=torch.long)
        arc_length = torch.zeros(self.max_branches)

        for branch_idx, points in enumerate(sample["branch_points"][: self.max_branches]):
            branch_ids[branch_idx] = int(sample["branch_ids"][branch_idx])
            arc_length[branch_idx] = self._arc_length(points)
            if len(points) > 0:
                start_points[branch_idx] = torch.as_tensor(points[0], dtype=torch.float32)
            if arc_length[branch_idx] < self.min_arc_length:
                branch_state[branch_idx] = 2
                continue

            resampled = self._resample(points, self.num_points + 1)
            start = torch.as_tensor(resampled[0], dtype=torch.float32)
            local = torch.as_tensor(resampled[1:] - resampled[0], dtype=torch.float32)
            if self.normalize:
                local = (local - self.point_mean) / self.point_std

            local_points[branch_idx] = local
            start_points[branch_idx] = start
            point_mask[branch_idx] = True
            branch_mask[branch_idx] = True
            branch_state[branch_idx] = 1

        return {
            "case": sample["case"],
            "local_points": local_points,
            "start_points": start_points,
            "point_mask": point_mask,
            "token_mask": point_mask.flatten(),
            "branch_mask": branch_mask,
            "branch_state": branch_state,
            "branch_ids": branch_ids,
            "arc_length": arc_length,
            "aneurysm_type": torch.as_tensor(sample["aneurysm_type"], dtype=torch.long),
            "canonical_type": sample["canonical_type"],
        }

    def denormalize_local_points(self, points):
        if not self.normalize:
            return points
        return points * self.point_std.to(points.device) + self.point_mean.to(points.device)
