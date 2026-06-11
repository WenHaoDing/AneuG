from pathlib import Path

import numpy as np
import torch


def _collect_processed_paths(root, cases):
    root = Path(root)
    if cases is None:
        return sorted(
            path for path in root.glob("*.npy")
            if not path.name.endswith("_sanity.npy")
        )
    return [root / f"{case}.npy" for case in cases]


class ProcessedGHDDataset(torch.utils.data.Dataset):
    def __init__(self, root, cases=None, withscale=False, normalize=True):
        self.root = Path(root)
        self.withscale = withscale
        self.normalize = normalize
        self.paths = _collect_processed_paths(self.root, cases)
        self.cases, self.ghd, self.scale, self.aneurysm_type = self._load()
        self.mean, self.std = self._fit_norm()

    def _load(self):
        cases, ghd, scale, aneurysm_type = [], [], [], []
        for path in self.paths:
            if not path.exists():
                continue
            chk = np.load(path, allow_pickle=True).item()
            cases.append(chk["case"])
            ghd.append(torch.as_tensor(chk["ghd"]["phi"], dtype=torch.float32).reshape(-1))
            scale.append(torch.as_tensor([float(chk["affine"]["scale"])], dtype=torch.float32))
            aneurysm_type.append(torch.as_tensor(int(chk["aneurysm_type"]), dtype=torch.long))
        if not ghd:
            raise RuntimeError(f"No processed GHD checkpoints found in {self.root}")
        return cases, ghd, scale, aneurysm_type

    def _fit_norm(self):
        data = torch.stack([
            torch.cat([ghd, scale]) if self.withscale else ghd
            for ghd, scale in zip(self.ghd, self.scale)
        ])
        return data.mean(0, keepdim=True), data.std(0, keepdim=True) + 0.01

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        x = torch.cat([self.ghd[idx], self.scale[idx]]) if self.withscale else self.ghd[idx]
        if self.normalize:
            x = (x - self.mean.squeeze(0)) / self.std.squeeze(0)
        return x, self.aneurysm_type[idx]

    def get_dim(self):
        return self.ghd[0].numel()

    def get_mean_std(self):
        if self.withscale:
            return self.mean[:, :-1], self.std[:, :-1]
        return self.mean, self.std

    def get_scale_mean_std(self):
        if not self.withscale:
            return None, None
        return self.mean[:, -1:], self.std[:, -1:]

    def denormalize(self, x):
        if not self.normalize:
            return x
        return x * self.std.to(x.device) + self.mean.to(x.device)


GHDDataset = ProcessedGHDDataset
