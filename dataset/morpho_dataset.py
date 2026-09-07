"""
Endcap dataset — thin wrapper over runtime_dataset/AneuG_morpho (see
dataset/preprocess_endcaps.py), for training the endcap predictor
(models/morphoformer.py).

Deliberately minimal: unlike the branch-generation datasets, mesh
reconstruction isn't done here at all — the model reconstructs it on the fly
from phi via MultiCanonicalGHDReconstruct.to_pyg_batch, the same convention
every other GCN-conditioned model in this codebase already uses. in_patch is
precomputed per-vertex, indexed by the SAME canonical vertex ordering
to_pyg_batch produces for that aneurysm_type (both ultimately load the same
dataset/canonical/<type>/mesh.obj), so it can be used directly without this
dataset ever building a mesh itself.

Manual labels (see dataset/label_morpho.py) live in the same checkpoint as
manual_in_patch, alongside the automatic endpoints / tangents / branch_mask /
in_patch. __getitem__ prefers manual_in_patch over automatic in_patch when
present, but ALWAYS uses the automatic endpoints/tangents/branch_mask —
brushing only refines the cap region; endpoint/tangent stay the
ray-crossing-derived real values (see dataset/label_morpho.py's module
docstring for why). A real case with no automatic record at all (shouldn't
normally happen) falls back to manual_endpoints/manual_tangents/manual_branch_mask
as its only source. A synthetic case (dataset/label_morpho.py's synthetic
mode) has only the plain fields to begin with — no manual_* — so it's read
exactly as written, endpoint/tangent included.

`root` accepts either one directory or a list of them, so real
(AneuG_morpho) and manually-labeled synthetic (AneuG_morpho_synthetic
— a separate folder, since synthetic cases have no automatic counterpart to
sit alongside) can be combined into one dataset:
MorphoDataset([ROOT/"AneuG_morpho", ROOT/"AneuG_morpho_synthetic"]).

Batching in_patch needs a custom collate: different aneurysm types have
different (fixed, per-type) vertex counts, so it can't be torch.stack'd like
the fixed-size fields (endpoints/tangents/branch_mask). It's concatenated
along the node dimension instead, mirroring how PyG's own
Batch.from_data_list concatenates node features — paired with a node_batch
index vector so the model can align it with its own PyG batch (built
separately, from phi, via to_pyg_batch) after the fact.

conda activate new
python dataset/morpho_dataset.py
"""

import sys
from pathlib import Path

import numpy as np


def item_n_verts(rec):
    """Vertex count implied by a record's own in_patch/dome arrays."""
    for key in ("manual_in_patch", "in_patch"):
        if key in rec and rec[key] is not None:
            return int(np.asarray(rec[key]).shape[-1])
    raise KeyError("cannot infer vertex count from record")
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class MorphoDataset(torch.utils.data.Dataset):
    def __init__(self, root, cases=None, max_branches=3, with_dome=False,
                 assembled_root=None, require_dome=True):
        """with_dome: also return a per-vertex aneurysm-dome mask.

        TWO MODELS ARE TRAINED FROM THIS CLASS and they want different targets:
          * uncapping model      caps only  -- what actually opens a mesh
          * descriptor model     caps+dome  -- the feature extractor behind the
                                 FPD/KPD generation metrics, where dome
                                 supervision is what forces the features to
                                 encode sac shape rather than just openings
        so the dome is an OPTION here rather than a second dataset class.

        The mask is taken from the record when a human has labelled it
        (manual_dome), else the automatic proposal -- either already in the
        record (dome_mask) or, failing that, read from the assembled corpus's
        per-case dome_mask.npy, which manage.py's assemble precomputes so the
        labelling machine needs no access to the raw geometry.
        """
        self.roots = [Path(root)] if isinstance(root, (str, Path)) else [Path(r) for r in root]
        self.max_branches = max_branches
        self.with_dome = with_dome
        self.assembled_root = Path(assembled_root) if assembled_root else None
        self.require_dome = require_dome
        self.paths = ([r / f"{c}.npy" for r in self.roots for c in cases] if cases is not None
                      else [p for r in self.roots for p in sorted(r.glob("*.npy"))])
        self.samples = self._load()

    def _dome_for(self, rec):
        """Per-vertex dome mask for one record, or None. Manual beats automatic."""
        for key in ("manual_dome", "dome_mask", "dome"):
            if key in rec and rec[key] is not None:
                return np.asarray(rec[key], dtype=bool)
        if self.assembled_root is not None:
            for ds in (rec.get("dataset"), "AneuX", "ImperialNHS"):
                if not ds:
                    continue
                p = self.assembled_root / ds / rec["case"] / "dome_mask.npy"
                if p.exists():
                    d = np.load(p, allow_pickle=True).item()
                    return np.asarray(d["mask"] if isinstance(d, dict) else d, dtype=bool)
        return None

    def _load(self):
        samples = []
        dropped = []
        for path in self.paths:
            if not path.exists():
                continue
            rec = np.load(path, allow_pickle=True).item()
            if self.with_dome:
                dome = self._dome_for(rec)
                if dome is None:
                    # Silently training the descriptor model on cases with no
                    # dome target would teach it that those meshes have no
                    # dome at all, so drop them and say so.
                    if self.require_dome:
                        dropped.append(rec.get("case", str(path)))
                        continue
                else:
                    rec["_dome"] = dome
            samples.append(rec)
        if dropped:
            print(f"[MorphoDataset] dropped {len(dropped)} case(s) with no dome label"
                  f" (e.g. {', '.join(dropped[:3])})")
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
        out = {
            "case": rec["case"],
            "aneurysm_type": torch.as_tensor(aneurysm_type, dtype=torch.long),
            "phi": torch.as_tensor(rec["phi"], dtype=torch.float32),
            "endpoints": torch.as_tensor(rec[endpoints_key][:self.max_branches], dtype=torch.float32),
            "tangents": torch.as_tensor(rec[tangents_key][:self.max_branches], dtype=torch.float32),
            "branch_mask": torch.as_tensor(rec[branch_mask_key][:self.max_branches], dtype=torch.bool),
            "in_patch": torch.as_tensor(rec[in_patch_key][:self.max_branches], dtype=torch.bool),  # [mb, N_type]
        }
        if self.with_dome:
            d = rec.get("_dome")
            out["dome"] = (torch.as_tensor(d, dtype=torch.bool) if d is not None
                           else torch.zeros(item_n_verts(rec), dtype=torch.bool))
        return out


def collate_morpho(batch):
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
    if "dome" in batch[0]:
        # per-vertex like in_patch -> concatenate along the node dimension so it
        # lines up with the same node_batch index
        out["dome"] = torch.cat([item["dome"] for item in batch], dim=0)   # [sum(N_i)]
    in_patch_list = [item["in_patch"] for item in batch]              # each [mb, N_i]
    node_counts = torch.tensor([ip.shape[1] for ip in in_patch_list])
    out["in_patch"] = torch.cat(in_patch_list, dim=1)                 # [mb, sum(N_i)]
    out["node_batch"] = torch.repeat_interleave(
        torch.arange(len(batch)), node_counts)                        # [sum(N_i)]
    return out


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    ds = MorphoDataset(ROOT / "runtime_dataset" / "AneuG_morpho")
    print(f"Loaded {len(ds)} samples.")
    loader = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate_morpho)
    batch = next(iter(loader))
    for k, v in batch.items():
        print(k, v.shape if torch.is_tensor(v) else v)
