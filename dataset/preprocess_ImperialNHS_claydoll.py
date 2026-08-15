"""
Variant of preprocess_ImperialNHS.py that additionally stores the UNCLIPPED
centerline alongside the clipped one, per case.

merged_centerline.npy (or merged_centerline_fallback.npy, matching whichever
pair clipped_centerline came from) holds the whole, unclipped VMTK-extracted
centerline: every branch's full point sequence, not just the post-cut tail
that clipped_centerline.npy keeps. Its branch group ids/list indices are not
guaranteed to line up with clipped_centerline.npy's (both files can come from
different centerline-extraction runs), so branches are matched geometrically:
for each connected branch's clipped points, find the merged branch whose point
cloud they lie closest to, then take that merged branch's FULL extent, oriented
(via the same "pts[0] nearest the aneurysm" convention used everywhere else in
this pipeline) so its point sequence is a superset of the clipped one — i.e.
unclipped_centerline["branch_points"][i] extends clipped_centerline["branch_points"][i]
for the same branch_ids[i], not a re-ordering of it.

Output format (see preprocess_ImperialNHS.py's own docstring for the ghd_root /
post_root input layout — unchanged here):

output_dir
├── case_name.npy
    └── object scalar dict:
        ├── case: str
        ├── version: "v2"
        ├── aneurysm_type: int
        │   └── 0 uses bifurcated canonical; 1 and 2 use sidewall canonical
        ├── canonical_type: "bifurcated" or "sidewall"
        ├── ghd: dict
        │   ├── phi: float32 [144, 3]
        │   ├── w_rot: float32 [3]
        │   ├── log_scale: float64 scalar
        │   └── t_vec: float32 [3]
        ├── affine: dict
        │   ├── R, R_frame, R_equivalent: float64 [3, 3]
        │   ├── scale, s_can, s_equivalent: float64 scalar
        │   └── neck_pt_world, t_equivalent: float64 [3]
        ├── clipped_centerline: dict — identical to preprocess_ImperialNHS.py's output
        │   ├── coordinate_system: "ghd"
        │   ├── source_coordinate_system: "world"
        │   ├── upstream_id: int
        │   ├── connected_branch_ids, base_branch_ids, branch_ranking,
        │   │   branch_ids: int64 [N_connected_branch]
        │   ├── branch_points: list length N_connected_branch,
        │   │   each float32 [N_pts, 3] (post-cut tail only)
        │   └── branches: list of {branch_id: int, group: int, pts: float32 [N_pts, 3]}
        ├── unclipped_centerline: dict — NEW in this script
        │   ├── coordinate_system, source_coordinate_system, upstream_id,
        │   │   connected_branch_ids, base_branch_ids, branch_ranking,
        │   │   branch_ids: same values/order as clipped_centerline above,
        │   │   so branch_points[i] here and in clipped_centerline both
        │   │   describe branch_ids[i]
        │   ├── branch_points: list length N_connected_branch,
        │   │   each float32 [N_pts', 3] or None (match failed — see below);
        │   │   N_pts' >= clipped_centerline's N_pts for the same branch, and
        │   │   clipped_centerline's points are a byte-identical suffix of it
        │   │   (unclipped adds the near-aneurysm points the clip trimmed off,
        │   │   prepended; it does not extend past the clip outward — the
        │   │   clipped branch already reaches the true branch tip)
        │   ├── branches: list of {branch_id: int, group: int,
        │   │   pts: float32 [N_pts', 3] or None}
        │   ├── match_mean_dist_mm: list length N_connected_branch, float
        │   │   mean nearest-neighbor distance (mm) from the clipped branch's
        │   │   points to the merged branch selected as its unclipped match;
        │   │   a branch_points/branches pts of None means every candidate
        │   │   merged branch scored above --match-dist-threshold (default
        │   │   2.0mm), i.e. no confident match was found
        │   └── branch_start_points: list length N_connected_branch,
        │       each float32 [3] or None — branch_points[i][0] for the same
        │       branch_ids[i] (each branch's own true start, one per branch,
        │       not a shared/averaged point — sidewall branches' starts tend
        │       to sit close together, but bifurcation branches' starts can
        │       be meaningfully offset from each other); None where that
        │       branch's match failed
        └── sources: dict of str — ghd_coefficients, landmarks, metrics,
            clipped_centerline, branch_ranking, clipped_reconstruction,
            endpoints_manual, merged_centerline (this last one is new)
├── case_name_sanity.png
    └── non-interactive sanity render: GHD reconstruction mesh (gray),
        clipped reconstruction mesh (orange), and ONLY the unclipped
        centerline branches, one solid color per branch — a circle marks each
        branch's start (nearest the aneurysm) and a triangle marks its end
        (branch tip). clipped_centerline is not drawn.

Only tested against the ImperialNHS dataset for now.

conda activate new
python dataset/preprocess_ImperialNHS_claydoll.py --save-sanity
"""

import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_DATASET_DIR = Path(__file__).resolve().parent
if str(_DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(_DATASET_DIR))

from preprocess_ImperialNHS import (
    DATASET_CONFIGS,
    V2GHDPreprocessor,
    _align_branch_points,
    _build_affine,
    _build_centerline,
    _canonical_info,
    _load_clipped_reconstruction_ghd,
    _load_npz,
    _load_pickle_npy,
    _reconstruct_ghd_numpy,
    _set_axes_equal,
)

DEFAULT_OUTPUT_DIR = str(_REPO_ROOT / "runtime" / "dataset" / "processed_claydoll")

# Mean nearest-neighbor distance (mm) above which a clipped->merged branch match
# is rejected as unreliable rather than silently accepted.
UNCLIPPED_MATCH_DIST_THRESHOLD_MM = 2.0


def _ensure_outward(pts, aneurysm_centroid):
    """Flip branch so pts[0] is the end closest to the aneurysm centroid.

    Local copy of AneuSeg/IAgents/tools/vessel_clipping.py's helper — not
    imported from there since that package pulls in vmtk_autogen-only deps
    (pyvista, SimpleITK) this script has no other reason to need.
    """
    pts = np.asarray(pts, dtype=np.float32)
    if np.linalg.norm(pts[-1] - aneurysm_centroid) < np.linalg.norm(pts[0] - aneurysm_centroid):
        return pts[::-1]
    return pts


def _match_merged_branch(clipped_pts, merged_branches, aneurysm_centroid, dist_threshold):
    """Find the merged branch whose points best overlap clipped_pts.

    Returns (full_pts_world, mean_dist) oriented via _ensure_outward, or
    (None, mean_dist) if no branch matches within dist_threshold (mm).
    """
    best_idx, best_score = None, np.inf
    for i, branch in enumerate(merged_branches):
        pts = branch.get("pts")
        if pts is None or len(pts) == 0:
            continue
        pts = np.asarray(pts, dtype=np.float32)
        d = np.linalg.norm(pts[None, :, :] - clipped_pts[:, None, :], axis=-1).min(axis=1)
        score = float(d.mean())
        if score < best_score:
            best_score, best_idx = score, i

    if best_idx is None or best_score > dist_threshold:
        return None, best_score

    full_pts = _ensure_outward(merged_branches[best_idx]["pts"], aneurysm_centroid)
    return full_pts, best_score


def _build_unclipped_centerline(clipped_built, clipped_raw, merged_raw, affine,
                                aneurysm_centroid, dist_threshold=UNCLIPPED_MATCH_DIST_THRESHOLD_MM):
    """Same branch_ids/branch_ranking/order as clipped_built, but with each
    branch's full (unclipped) point sequence instead of the post-cut tail."""
    merged_branches = merged_raw["branches"]
    branches = []
    match_mean_dist_mm = []

    for branch_id in clipped_built["branch_ids"]:
        branch_id = int(branch_id)
        clipped_pts_world = np.asarray(clipped_raw["branches"][branch_id]["pts"], dtype=np.float32)
        group = int(clipped_raw["branches"][branch_id].get("group", branch_id))

        full_pts_world, score = _match_merged_branch(
            clipped_pts_world, merged_branches, aneurysm_centroid, dist_threshold
        )
        match_mean_dist_mm.append(score)

        if full_pts_world is None:
            print(f"    [WARN] unclipped match failed for branch_id={branch_id} "
                  f"(best mean dist={score:.2f}mm > {dist_threshold}mm threshold)")
            branches.append({"branch_id": branch_id, "group": group, "pts": None})
            continue

        branches.append({
            "branch_id": branch_id,
            "group": group,
            "pts": _align_branch_points(full_pts_world, affine),
        })

    branch_start_points = [
        (branch["pts"][0] if branch["pts"] is not None else None) for branch in branches
    ]

    return {
        "coordinate_system": "ghd",
        "source_coordinate_system": "world",
        "upstream_id": clipped_built["upstream_id"],
        "connected_branch_ids": clipped_built["connected_branch_ids"],
        "base_branch_ids": clipped_built["base_branch_ids"],
        "branch_ranking": clipped_built["branch_ranking"],
        "branch_ids": clipped_built["branch_ids"],
        "branch_points": [branch["pts"] for branch in branches],
        "branches": branches,
        "match_mean_dist_mm": match_mean_dist_mm,
        "branch_start_points": branch_start_points,
    }


class ClaydollGHDPreprocessor(V2GHDPreprocessor):
    """V2GHDPreprocessor plus an "unclipped_centerline" dict per case."""

    def __init__(self, *args, match_dist_threshold=UNCLIPPED_MATCH_DIST_THRESHOLD_MM, **kwargs):
        super().__init__(*args, **kwargs)
        self.match_dist_threshold = match_dist_threshold

    def _case_paths(self, case):
        paths = super()._case_paths(case)
        clipped_name = paths["clipped_centerline"].name
        merged_name = clipped_name.replace("clipped_centerline", "merged_centerline")
        paths["merged_centerline"] = (self.post_root / case / merged_name)
        return paths

    def process_case(self, case):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / f"{case}.npy"
        if out_path.exists() and not self.overwrite:
            return out_path

        paths = self._case_paths(case)
        coeffs = _load_npz(paths["ghd_coefficients"])
        landmarks = _load_npz(paths["landmarks"])
        with open(paths["metrics"], "r") as f:
            metrics = json.load(f)

        clipped = _load_pickle_npy(paths["clipped_centerline"]).item()
        branch_ranking = np.asarray(np.load(paths["branch_ranking"]), dtype=np.int64)
        aneurysm_type = self._load_aneurysm_type(paths["endpoints_manual"])
        canonical_type, _ = _canonical_info(aneurysm_type)

        ghd = self._build_ghd(coeffs)
        affine = _build_affine(ghd, landmarks, metrics)
        clipped_built = _build_centerline(clipped, branch_ranking, affine)

        merged = _load_pickle_npy(paths["merged_centerline"]).item()
        endpoints_ = _load_pickle_npy(paths["endpoints_manual"]).item()
        aneurysm_centroid = np.asarray(endpoints_["aneurysm_centroid"], dtype=np.float32)
        unclipped_built = _build_unclipped_centerline(
            clipped_built, clipped, merged, affine, aneurysm_centroid,
            dist_threshold=self.match_dist_threshold,
        )

        checkpoint = {
            "case": case,
            "version": "v2",
            "aneurysm_type": aneurysm_type,
            "canonical_type": canonical_type,
            "ghd": ghd,
            "affine": {key: val for key, val in affine.items() if not key.startswith("_")},
            "clipped_centerline": clipped_built,
            "unclipped_centerline": unclipped_built,
            "sources": {key: str(path) for key, path in paths.items()},
        }
        np.save(out_path, checkpoint, allow_pickle=True)
        return out_path

    def save_sanity_picture(self, processed_path, ghd_reconstruct=None):
        processed_path = Path(processed_path)
        sanity_path = self.output_dir / "sanity_" / f"{processed_path.stem}_sanity.png"
        fig, _, _ = sanity_check_processed_checkpoint_with_unclipped(
            processed_path, ghd_reconstruct=ghd_reconstruct, save_path=sanity_path,
        )
        import matplotlib.pyplot as plt
        plt.close(fig)
        return sanity_path


def sanity_check_processed_checkpoint_with_unclipped(
    processed_path, ghd_reconstruct=None, save_path=None, device=None, denormalize_shape=True,
):
    """sanity_check_processed_checkpoint (preprocess_ImperialNHS.py), plus the
    unclipped centerline branches overlaid as dashed extensions of the clipped ones."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    checkpoint = np.load(processed_path, allow_pickle=True).item()
    aneurysm_type = int(checkpoint["aneurysm_type"])
    if ghd_reconstruct is None:
        mesh = None
        verts, faces = _reconstruct_ghd_numpy(checkpoint, denormalize_shape=denormalize_shape)
    else:
        from preprocess_ImperialNHS import _reconstruct_with_backend
        mesh = _reconstruct_with_backend(checkpoint, ghd_reconstruct, aneurysm_type, device, denormalize_shape)
        verts, faces = (item.detach().cpu().numpy() for item in (mesh.verts_list()[0], mesh.faces_list()[0]))

    clipped_centerline = checkpoint["clipped_centerline"]
    unclipped_centerline = checkpoint.get("unclipped_centerline")
    if clipped_centerline.get("coordinate_system") != "ghd":
        raise ValueError(f"Expected GHD-space centerlines, got {clipped_centerline.get('coordinate_system')}")
    clipped_branch_points = [np.asarray(pts, dtype=np.float32) for pts in clipped_centerline["branch_points"]]
    unclipped_branch_points = (
        [None if pts is None else np.asarray(pts, dtype=np.float32) for pts in unclipped_centerline["branch_points"]]
        if unclipped_centerline is not None else [None] * len(clipped_branch_points)
    )

    clipped_verts, clipped_faces = _load_clipped_reconstruction_ghd(checkpoint)

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces,
                    color="lightgray", edgecolor="none", alpha=0.45)
    if clipped_verts is not None:
        ax.plot_trisurf(
            clipped_verts[:, 0], clipped_verts[:, 1], clipped_verts[:, 2],
            triangles=clipped_faces, color="darkorange", edgecolor="none", alpha=0.16,
        )

    colors = ["black", "blue", "red", "yellow"]
    branch_ids = clipped_centerline["branch_ids"]
    for idx in range(len(clipped_branch_points)):
        color = colors[idx % len(colors)]
        unclipped_pts = unclipped_branch_points[idx]
        if unclipped_pts is None:
            continue
        ax.plot(unclipped_pts[:, 0], unclipped_pts[:, 1], unclipped_pts[:, 2],
                linewidth=2.0, color=color, label=f"branch {int(branch_ids[idx])}")
        ax.scatter(*unclipped_pts[0], s=36, marker="o", color=color)   # start
        ax.scatter(*unclipped_pts[-1], s=36, marker="^", color=color)  # end

    case = checkpoint.get("case", Path(processed_path).stem)
    ax.set_title(
        f"{case} (type {aneurysm_type}: {checkpoint.get('canonical_type', 'unknown')})\n"
        "GHD reconstruction: gray | clipped reconstruction: orange | "
        "unclipped centerline: o start, ▲ end"
    )
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    geometry_points = [verts, *[p for p in unclipped_branch_points if p is not None]]
    if clipped_verts is not None:
        geometry_points.append(clipped_verts)
    _set_axes_equal(ax, *geometry_points)
    ax.legend(loc="upper right")
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
    return fig, ax, mesh


def _parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="Preprocess ImperialNHS GHD checkpoints into condensed .npy files, "
                    "each also carrying an unclipped_centerline dict alongside clipped_centerline."
    )
    parser.add_argument("--dataset", choices=tuple(DATASET_CONFIGS), default="imperialnhs")
    parser.add_argument("--ghd-root", default=None)
    parser.add_argument("--geometry-root", "--post-root", dest="geometry_root", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--aneurysm-types", nargs="+", type=int, choices=(0, 1, 2), default=None)
    parser.add_argument("--overwrite", action="store_true", default=True)
    parser.add_argument("--save-sanity", action="store_true")
    parser.add_argument("--match-dist-threshold", type=float, default=UNCLIPPED_MATCH_DIST_THRESHOLD_MM,
                        help="Max mean nearest-neighbor distance (mm) for a clipped->merged "
                             "branch match to be accepted (default: %(default)s).")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    dataset_config = DATASET_CONFIGS[args.dataset]
    aneurysm_types = args.aneurysm_types if args.aneurysm_types is not None else dataset_config["default_aneurysm_types"]

    preprocessor = ClaydollGHDPreprocessor(
        ghd_root=args.ghd_root or dataset_config["ghd_root"],
        post_root=args.geometry_root or dataset_config["geometry_root"],
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        save_sanity=args.save_sanity,
        aneurysm_types=aneurysm_types,
        dataset_name=args.dataset,
        match_dist_threshold=args.match_dist_threshold,
    )
    summary = preprocessor.run(cases=args.cases)
    print(
        f"Processed {len(summary['processed'])} case(s), "
        f"failed {len(summary['failed'])} case(s), "
        f"skipped {len(summary['skipped_type'])} case(s) by type, "
        f"sanity image failed {len(summary['sanity_failed'])} case(s)."
    )


"""
conda activate new
python dataset/preprocess_ImperialNHS_claydoll.py --save-sanity
python dataset/preprocess_ImperialNHS_claydoll.py --case 0m5cVmE8R1_aneurysm1 --save-sanity
"""
