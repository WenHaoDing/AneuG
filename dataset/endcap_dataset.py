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

Manual labels (see dataset/label_endcaps.py) live in the same checkpoint as
manual_in_patch, alongside the automatic endpoints / tangents / branch_mask /
in_patch. __getitem__ prefers manual_in_patch over automatic in_patch when
present, but ALWAYS uses the automatic endpoints/tangents/branch_mask —
brushing only refines the cap region; endpoint/tangent stay the
ray-crossing-derived real values (see dataset/label_endcaps.py's module
docstring for why). A real case with no automatic record at all (shouldn't
normally happen) falls back to manual_endpoints/manual_tangents/manual_branch_mask
as its only source. A synthetic case (dataset/label_endcaps.py's synthetic
mode) has only the plain fields to begin with — no manual_* — so it's read
exactly as written, endpoint/tangent included.

`root` accepts either one directory or a list of them, so real
(processed_endcaps) and manually-labeled synthetic (processed_endcaps_synthetic
— a separate folder, since synthetic cases have no automatic counterpart to
sit alongside) can be combined into one dataset:
EndcapDataset([ROOT/"processed_endcaps", ROOT/"processed_endcaps_synthetic"]).

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
        self.roots = [Path(root)] if isinstance(root, (str, Path)) else [Path(r) for r in root]
        self.max_branches = max_branches
        self.paths = ([r / f"{c}.npy" for r in self.roots for c in cases] if cases is not None
                      else [p for r in self.roots for p in sorted(r.glob("*.npy"))])
        self.samples = self._load()

    def _load(self):
        samples = []
        for path in self.paths:
            if not path.exists():
                continue
            samples.append(np.load(path, allow_pickle=True).item())
        if not samples:
            raise RuntimeError(f"No processed endcap checkpoints found in {self.roots}")
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
        # Ground-truth (automatic) endpoint/tangent/branch_mask win when present — only
        # in_patch prefers the manual brush (see module docstring for why the split isn't
        # symmetric). Falls back to manual_* per-field for the rare case with no automatic
        # record at all (real, no ground truth) or none at all (won't happen — synthetic
        # records only ever have the plain fields).
        endpoints_key = "endpoints" if "endpoints" in rec else "manual_endpoints"
        tangents_key = "tangents" if "tangents" in rec else "manual_tangents"
        branch_mask_key = "branch_mask" if "branch_mask" in rec else "manual_branch_mask"
        in_patch_key = "manual_in_patch" if "manual_in_patch" in rec else "in_patch"
        return {
            "case": rec["case"],
            "aneurysm_type": torch.as_tensor(aneurysm_type, dtype=torch.long),
            "phi": torch.as_tensor(rec["phi"], dtype=torch.float32),
            "endpoints": torch.as_tensor(rec[endpoints_key][:self.max_branches], dtype=torch.float32),
            "tangents": torch.as_tensor(rec[tangents_key][:self.max_branches], dtype=torch.float32),
            "branch_mask": torch.as_tensor(rec[branch_mask_key][:self.max_branches], dtype=torch.bool),
            "in_patch": torch.as_tensor(rec[in_patch_key][:self.max_branches], dtype=torch.bool),  # [mb, N_type]
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

    ds = EndcapDataset(ROOT / "runtime" / "dataset" / "processed_endcaps")
    print(f"Loaded {len(ds)} samples.")
    loader = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate_endcaps)
    batch = next(iter(loader))
    for k, v in batch.items():
        print(k, v.shape if torch.is_tensor(v) else v)
