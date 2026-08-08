"""
Preprocess v1 GHD checkpoints into condensed data checkpoints.

This mirrors `dataset/preprocess.py` (the v2 pipeline) and emits the *same*
processed-checkpoint schema, so the existing dataset classes
(`ProcessedGHDDataset`, `VesselSkeletonDataset`) consume v1 cases unchanged.
Once validated, `V1GHDPreprocessor` is intended to be transplanted into
`dataset/preprocess.py`.

Key differences from v2
-----------------------
* v1 has a single canonical (``canonical_typeB``), which is identical to the
  v2 Bifurcated canonical (``dataset/canonical/Bifurcated``). Every v1 case is
  therefore treated as ``aneurysm_type = 0`` / ``canonical_type = "bifurcated"``.
* v1 GHD fitting stores ``s`` as the scale directly (not log-scale); we store
  ``log_scale = log(s)`` to match the v2 schema.
* v1 centerline points are already expressed in the GHD canonical coordinate
  system, so NO world->GHD alignment (R/s/T) is applied. They are, however,
  stored by v1 in the *normalized* canonical scale (verts / norm_canonical).
  To keep the same convention as v2 -- where ``branch_points`` live in the
  *denormalized* canonical frame that overlays the ``denormalize_shape=True``
  mesh -- we multiply the v1 points by ``norm_canonical`` so they share v2's
  frame, and the sanity render reconstructs with ``denormalize_shape=True``.

v1 checkpoint format
--------------------
ghd_root (checkpoints_v1/ghd_fitting)
├── case name
    └── vanilla
        └── ghb_fitting_checkpoint_5.pkl
            └── pickled dict:
                ├── R: float32 [1, 3]  (axis-angle rotation, w_rot)
                ├── s: float32 [1, 1]  (scale)
                ├── T: float32 [1, 3]  (translation, t_vec)
                └── GHD_coefficient: float32 [144, 3]  (phi)

centreline_root (checkpoints_v1/centreline_fitting/stable)
├── case_name.pth
    └── torch dict (relevant fields):
        ├── label: str
        ├── split_centerline: list[N_branch] of float32 [N_pts, 3]
        │   └── the measured/clipped centerline in GHD coords (used here)
        ├── pred_centerline_glo: list[N_branch] of float32 [N_pts, 3]
        │   └── Fourier-reconstructed centerline (not used)
        ├── branch_length: list[N_branch] of float32 scalar
        ├── ghd: float32 [432]  (== GHD_coefficient flattened)
        └── scale: float32 [1]  (== s)

Processed v1 output format (parallel to v2)
-------------------------------------------
output_dir
├── case_name.npy
    └── object scalar dict:
        ├── case: str
        ├── version: "v1"
        ├── aneurysm_type: 0
        ├── canonical_type: "bifurcated"
        ├── ghd: dict
        │   ├── phi: float32 [144, 3]
        │   ├── w_rot: float32 [3]
        │   ├── log_scale: float64 scalar  (= log(s))
        │   └── t_vec: float32 [3]
        ├── affine: dict
        │   ├── R: float64 [3, 3]  (so3 exp of w_rot)
        │   └── scale: float64 scalar  (= s = exp(log_scale))
        ├── clipped_centerline: dict
        │   ├── coordinate_system: "ghd"  (denormalized, == v2 convention)
        │   ├── source_coordinate_system: "ghd_normalized"
        │   ├── norm_canonical: float64 scalar  (applied denorm factor)
        │   ├── branch_ids: int64 [N_branch]  (native order 0..N-1)
        │   ├── branch_points: list[N_branch] of float32 [N_pts, 3]
        │   └── branches: list[N_branch] of dict {branch_id, group, pts}
        └── sources: dict {ghd_coefficients, centerline}
├── sanity_/case_name_sanity.png
    └── reconstructed GHD mesh (normalized) overlaid with GHD-space centerlines
"""

import argparse
import math
import pickle
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Reuse the v2 helpers; the v1 canonical == v2 Bifurcated canonical.
from dataset.preprocess_ImperialNHS import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as V2_OUTPUT_DIR,
    _canonical_paths_for_aneurysm_type,
    _load_obj_mesh,
    _reconstruct_ghd_numpy,
    _set_axes_equal,
    _so3_exp_map_numpy,
)

DEFAULT_GHD_ROOT = str(_REPO_ROOT / "checkpoints_v1" / "ghd_fitting")
DEFAULT_CENTRELINE_ROOT = str(_REPO_ROOT / "checkpoints_v1" / "centreline_fitting" / "stable")
# Write into the same folder as the v2 pipeline so a single processed root holds
# both versions for combined downstream training.
DEFAULT_OUTPUT_DIR = V2_OUTPUT_DIR

GHD_PKL_RELPATH = Path("vanilla") / "ghb_fitting_checkpoint_5.pkl"
V1_ANEURYSM_TYPE = 0
V1_CANONICAL_TYPE = "bifurcated"


_NORM_CANONICAL_CACHE = {}


def _norm_canonical_for_type(aneurysm_type):
    """Intrinsic normalization scale of a canonical mesh.

    Matches the factor used in `dataset.preprocess._reconstruct_ghd_numpy`
    (and v1 `GHD_Reconstruct.normalize`): max vertex norm * 1.10 * 2.50.
    """
    key = int(aneurysm_type)
    if key not in _NORM_CANONICAL_CACHE:
        mesh_path, _ = _canonical_paths_for_aneurysm_type(key)
        verts, _ = _load_obj_mesh(mesh_path)
        _NORM_CANONICAL_CACHE[key] = float(
            np.linalg.norm(verts, axis=-1).max() * 1.10 * 2.50
        )
    return _NORM_CANONICAL_CACHE[key]


def _to_numpy(value):
    if value.__class__.__module__.startswith("torch"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _load_ghd_pkl(path):
    with open(path, "rb") as f:
        chk = pickle.load(f)
    return {
        "R": _to_numpy(chk["R"]).reshape(-1).astype(np.float64),
        "s": float(_to_numpy(chk["s"]).reshape(-1)[0]),
        "T": _to_numpy(chk["T"]).reshape(-1).astype(np.float64),
        "phi": _to_numpy(chk["GHD_coefficient"]).astype(np.float32),
    }


def _load_centerline_pth(path):
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


class V1GHDPreprocessor:
    """
    Condense v1 GHD fitting outputs and centerlines into one .npy file per case.

    Only cases that have BOTH a GHD pkl and a centerline .pth are processed, so
    every output is valid for both `ProcessedGHDDataset` and
    `VesselSkeletonDataset`.
    """

    def __init__(self, ghd_root, centreline_root, output_dir,
                 overwrite=False, save_sanity=True):
        self.ghd_root = Path(ghd_root)
        self.centreline_root = Path(centreline_root)
        self.output_dir = Path(output_dir)
        self.overwrite = overwrite
        self.save_sanity = save_sanity

    def _ghd_pkl_path(self, case):
        return self.ghd_root / case / GHD_PKL_RELPATH

    def _centreline_path(self, case):
        return self.centreline_root / f"{case}.pth"

    def iter_cases(self):
        for case_dir in sorted(self.ghd_root.iterdir()):
            if not case_dir.is_dir():
                continue
            case = case_dir.name
            if self._ghd_pkl_path(case).exists() and self._centreline_path(case).exists():
                yield case

    def run(self, cases=None):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        summary = {"processed": [], "failed": [], "sanity_failed": []}

        for case in (list(cases) if cases is not None else self.iter_cases()):
            try:
                out_path = self.process_case(case)
                summary["processed"].append(out_path)
                print(f"[OK] {case} -> {out_path}")
            except Exception as exc:
                summary["failed"].append((case, str(exc)))
                print(f"[FAIL] {case}: {exc}")
                continue

            if self.save_sanity:
                try:
                    sanity_path = self.save_sanity_picture(out_path)
                    print(f"[OK] {case} sanity -> {sanity_path}")
                except Exception as exc:
                    summary["sanity_failed"].append((case, str(exc)))
                    print(f"[FAIL] {case} sanity: {exc}")

        return summary

    def process_case(self, case):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / f"{case}.npy"
        if out_path.exists() and not self.overwrite:
            return out_path

        ghd_pkl_path = self._ghd_pkl_path(case)
        centreline_path = self._centreline_path(case)
        ghd_raw = _load_ghd_pkl(ghd_pkl_path)
        centreline = _load_centerline_pth(centreline_path)

        ghd = self._build_ghd(ghd_raw)
        affine = self._build_affine(ghd_raw)
        norm_canonical = _norm_canonical_for_type(V1_ANEURYSM_TYPE)

        checkpoint = {
            "case": case,
            "version": "v1",
            "aneurysm_type": V1_ANEURYSM_TYPE,
            "canonical_type": V1_CANONICAL_TYPE,
            "ghd": ghd,
            "affine": affine,
            "clipped_centerline": self._build_centerline(centreline, norm_canonical),
            "sources": {
                "ghd_coefficients": str(ghd_pkl_path),
                "centerline": str(centreline_path),
            },
        }
        np.save(out_path, checkpoint, allow_pickle=True)
        return out_path

    @staticmethod
    def _build_ghd(ghd_raw):
        scale = ghd_raw["s"]
        if scale <= 0:
            raise ValueError(f"Non-positive scale s={scale}")
        return {
            "phi": ghd_raw["phi"].astype(np.float32),
            "w_rot": ghd_raw["R"].astype(np.float32),
            "log_scale": np.array(math.log(scale), dtype=np.float64),
            "t_vec": ghd_raw["T"].astype(np.float32),
        }

    @staticmethod
    def _build_affine(ghd_raw):
        return {
            "R": _so3_exp_map_numpy(ghd_raw["R"]),
            "scale": np.array(ghd_raw["s"], dtype=np.float64),
        }

    @staticmethod
    def _build_centerline(centreline, norm_canonical):
        # v1 stores centerlines in the normalized canonical frame; scale by
        # norm_canonical so they match v2's denormalized ("ghd") convention.
        branches_raw = centreline["split_centerline"]
        branch_points = []
        branches = []
        for branch_id, pts in enumerate(branches_raw):
            pts = _to_numpy(pts).astype(np.float32)
            if pts.ndim != 2 or pts.shape[1] != 3:
                raise ValueError(f"Branch {branch_id} has unexpected shape {pts.shape}")
            pts = (pts * norm_canonical).astype(np.float32)
            branch_points.append(pts)
            branches.append({
                "branch_id": int(branch_id),
                "group": int(branch_id),
                "pts": pts,
            })

        return {
            "coordinate_system": "ghd",
            "source_coordinate_system": "ghd_normalized",
            "norm_canonical": np.array(norm_canonical, dtype=np.float64),
            "branch_ids": np.arange(len(branches), dtype=np.int64),
            "branch_points": branch_points,
            "branches": branches,
        }

    def save_sanity_picture(self, processed_path):
        processed_path = Path(processed_path)
        sanity_path = self.output_dir / "sanity_" / f"{processed_path.stem}_sanity.png"
        fig = sanity_check_processed_checkpoint_v1(processed_path, save_path=sanity_path)
        import matplotlib.pyplot as plt
        plt.close(fig)
        return sanity_path


def sanity_check_processed_checkpoint_v1(processed_path, save_path=None,
                                         denormalize_shape=True):
    """
    Reconstruct the GHD mesh (denormalized canonical space) and overlay the
    processed v1 centerlines, which are stored in that same denormalized GHD
    frame (matching the v2 convention).
    """
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    checkpoint = np.load(processed_path, allow_pickle=True).item()
    verts, faces = _reconstruct_ghd_numpy(checkpoint, denormalize_shape=denormalize_shape)

    centerline = checkpoint["clipped_centerline"]
    if centerline.get("coordinate_system") != "ghd":
        raise ValueError(f"Expected GHD-space centerlines, got {centerline.get('coordinate_system')}")
    branch_points = [np.asarray(pts, dtype=np.float32) for pts in centerline["branch_points"]]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces,
                    color="lightgray", edgecolor="none", alpha=0.25)

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(branch_points), 1)))
    branch_ids = centerline["branch_ids"]
    for idx, pts in enumerate(branch_points):
        color = colors[idx % len(colors)]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], linewidth=2.0, color=color,
                label=f"branch {int(branch_ids[idx])}")
        ax.scatter(*pts[0], s=18, color=color)

    case = checkpoint.get("case", Path(processed_path).stem)
    ax.set_title(f"{case} (v1, type {checkpoint['aneurysm_type']}: {checkpoint['canonical_type']})")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    _set_axes_equal(ax, verts, *branch_points)
    ax.legend(loc="upper right")
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
    return fig


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess v1 GHD checkpoints into one .npy file per case."
    )
    parser.add_argument("--ghd-root", default=DEFAULT_GHD_ROOT)
    parser.add_argument("--centreline-root", default=DEFAULT_CENTRELINE_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--overwrite", action="store_true", default=True)
    parser.add_argument(
        "--save-sanity",
        action="store_true",
        help="Also render sanity PNGs (imports the reconstruction stack).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    preprocessor = V1GHDPreprocessor(
        ghd_root=args.ghd_root,
        centreline_root=args.centreline_root,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        save_sanity=args.save_sanity,
    )
    summary = preprocessor.run(cases=args.cases)
    print(
        f"Processed {len(summary['processed'])} case(s), "
        f"failed {len(summary['failed'])} case(s), "
        f"sanity image failed {len(summary['sanity_failed'])} case(s)."
    )

"""
conda activate new
python dataset/preprocess_v1.py
python dataset/preprocess_v1.py --save-sanity

"""
