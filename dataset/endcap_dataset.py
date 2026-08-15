"""
Endcap dataset — thin wrapper over dataset/processed_endcaps (see
dataset/preprocess_endcaps.py), for training the endcap predictor
(models/endcap_predictor.py).

Deliberately minimal: unlike the branch-generation datasets, mesh
reconstruction isn't done here at all — the model reconstructs it on the fly
from phi via MultiCanonicalGHDReconstruct.to_pyg_batch, the same convention
every other GCN-conditioned model in this codebase already uses. in_patch is
precomputed per-vertex, indexed by the SAME canonical vertex ordering
to_pyg_batch produces for that aneurysm_type (both ultimately load the same
dataset/canonical/<type>/mesh.obj), so it can be used directly without this
dataset ever building a mesh itself.

Batching in_patch needs a custom collate: different aneurysm types have
different (fixed, per-type) vertex counts, so it can't be torch.stack'd like
the fixed-size fields (endpoints/tangents/branch_mask). It's concatenated
along the node dimension instead, mirroring how PyG's own
Batch.from_data_list concatenates node features — paired with a node_batch
index vector so the model can align it with its own PyG batch (built
separately, from phi, via to_pyg_batch) after the fact.

conda activate new
python dataset/endcap_dataset.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class EndcapDataset(torch.utils.data.Dataset):
    def __init__(self, root, cases=None, max_branches=3):
        self.root = Path(root)
        self.max_branches = max_branches
        self.paths = ([self.root / f"{c}.npy" for c in cases] if cases is not None
                      else sorted(self.root.glob("*.npy")))
        self.samples = self._load()

    def _load(self):
        samples = []
        for path in self.paths:
            if not path.exists():
                continue
            samples.append(np.load(path, allow_pickle=True).item())
        if not samples:
            raise RuntimeError(f"No processed endcap checkpoints found in {self.root}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rec = self.samples[idx]
        # Types 1 and 2 both resolve to the same canonical reconstructor
        # (MultiCanonicalGHDReconstruct.specs: 1 and 2 -> Sidewall) — merged
        # here for consistency with scripts/train/train_ghd_vae.py's
        # MERGE_TYPE2_INTO_TYPE1, so type 2 never surfaces anywhere in this
        # pipeline's data, training, or sanity panels.
        aneurysm_type = 1 if int(rec["aneurysm_type"]) == 2 else int(rec["aneurysm_type"])
        return {
            "case": rec["case"],
            "aneurysm_type": torch.as_tensor(aneurysm_type, dtype=torch.long),
            "phi": torch.as_tensor(rec["phi"], dtype=torch.float32),
            "endpoints": torch.as_tensor(rec["endpoints"][:self.max_branches], dtype=torch.float32),
            "tangents": torch.as_tensor(rec["tangents"][:self.max_branches], dtype=torch.float32),
            "branch_mask": torch.as_tensor(rec["branch_mask"][:self.max_branches], dtype=torch.bool),
            "in_patch": torch.as_tensor(rec["in_patch"][:self.max_branches], dtype=torch.bool),  # [mb, N_type]
        }


def collate_endcaps(batch):
    """Fixed-size fields stack normally; in_patch (variable per-sample vertex
    count, depends on aneurysm_type) is concatenated along the node dimension
    with a node_batch index, matching PyG's own batching convention so it
    aligns with a separately-built to_pyg_batch(phi, aneurysm_type, ...)."""
    out = {
        "case": [item["case"] for item in batch],
        "aneurysm_type": torch.stack([item["aneurysm_type"] for item in batch]),
        "phi": torch.stack([item["phi"] for item in batch]),
        "endpoints": torch.stack([item["endpoints"] for item in batch]),
        "tangents": torch.stack([item["tangents"] for item in batch]),
        "branch_mask": torch.stack([item["branch_mask"] for item in batch]),
    }
    in_patch_list = [item["in_patch"] for item in batch]              # each [mb, N_i]
    node_counts = torch.tensor([ip.shape[1] for ip in in_patch_list])
    out["in_patch"] = torch.cat(in_patch_list, dim=1)                 # [mb, sum(N_i)]
    out["node_batch"] = torch.repeat_interleave(
        torch.arange(len(batch)), node_counts)                        # [sum(N_i)]
    return out


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    ds = EndcapDataset(ROOT / "dataset" / "processed_endcaps")
    print(f"Loaded {len(ds)} samples.")
    loader = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate_endcaps)
    batch = next(iter(loader))
    for k, v in batch.items():
        print(k, v.shape if torch.is_tensor(v) else v)
