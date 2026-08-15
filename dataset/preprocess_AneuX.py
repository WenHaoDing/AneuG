"""
Preprocess v1 and v2 GHD checkpoints into condensed data checkpoints.

v1 checkpoint format:

ghd_root
├── case name
    ├── case_name
    ├── case_name.json

v2 checkpoint format:

ghd_root
├── case name
    ├── ghd_coefficients.npz
    │   ├── phi: float32 [144, 3]
    │   ├── w_rot: float32 [3]
    │   ├── log_scale: float64 scalar
    │   └── t_vec: float32 [3]
    ├── landmarks.npz
    │   ├── neck, dome, cap_up, cap_down: float64 [3]
    │   ├── R_frame: float64 [3, 3]
    │   ├── neck_pt_world: float64 [3]
    │   ├── r_up_mm, r_down_mm: float64 scalar
    │   ├── dome_valid: bool scalar
    │   ├── ring_up_idxs, ring_dn_idxs: int32 [N_ring]
    │   ├── z_source, dome_centroid_source, dome_centroid_method: str scalar
    │   ├── tangent_* metrics: float64 scalar
    │   ├── dome_centroid_world: float64 [3]
    │   └── dome_centroid_n_verts, dome_centroid_n_faces: int32 scalar
    ├── metrics.json
    │   └── chamfer_best, chamfer_final, best_iter, dice, gar,
    │       converged, s_can
    └── loss_history.npz (optional training curves)
        └── each recorded loss/parameter curve: float64 [N_iter]

post_root
├── case name
    ├── clipped_centerline.npy / clipped_centerline_fallback.npy
    │   └── object scalar dict:
    │       ├── branches: list length N_branch
    │       │   └── each branch dict:
    │       │       ├── group: int
    │       │       ├── pts: float32 [N_pts, 3]
    │       │       ├── radii: float64 [N_pts]
    │       │       └── arc_length: float scalar
    │       ├── bif_centroids: list length N_bif
    │       │   └── each dict: group int, centroid float32 [3]
    │       ├── nearest_bif: dict with group int, centroid float32 [3]
    │       ├── connected_branch_ids: list[int]
    │       └── upstream_id: int
    ├── branch_ranking.npy / branch_ranking_fallback.npy
    │   └── int64 [N_connected_branch]
    ├── merged_centerline.npy / merged_centerline_fallback.npy
    │   └── same object scalar dict structure as clipped_centerline,
    │       usually without upstream_id
    ├── forward_fusion_info.npz / forward_fusion_info_fallback.npz
    │   ├── cpcd_glo: object [N_opening], each item float64 [N_pts, 3]
    │   ├── cpcd_glo_tangent: object [N_opening], each item float64 [N_pts, 3]
    │   ├── opening_centroids: float32 [N_opening, 3]
    │   ├── opening_vertex_ids: object [N_opening], each item int64 [N_vertex]
    │   └── allow_pickle: bool scalar
    ├── endpoints_manual.npy
    │   └── object scalar dict:
    │       ├── endpoints: float64 [N_endpoint, 3]
    │       ├── aneurysm_centroid: float64 [3]
    │       └── aneurysm_type: int
    ├── endpoints.npy (optional)
    │   └── object scalar dict with endpoints [N_endpoint, 3],
    │       aneurysm_centroid [3]
    └── regional_growth_array.npy (optional)
        └── uint8 [D, H, W]

Primary vs. fallback centerline/mesh selection (see V2GHDPreprocessor._centerline_paths):
  - aneurysm_type == 2: the fallback triple (clipped_centerline_fallback.npy,
    branch_ranking_fallback.npy, clipped_reconstruction_fallback.ply) is
    REQUIRED — raises FileNotFoundError if any of the three is missing. Type-2
    cases are always GHD-fitted against the short type-I fallback clip, never
    the long primary/CFD clip.
  - aneurysm_type in (0, 1): opportunistic — use the fallback triple if and
    only if all three fallback files exist, regardless of type; otherwise fall
    back to the primary triple (clipped_centerline.npy, branch_ranking.npy,
    clipped_reconstruction.ply), raising if even the primary triple is
    incomplete. This lets a manually re-cut type-0/1 case (e.g. a "_cut2"
    variant) be picked up automatically once its fallback files are complete,
    with no aneurysm_type change required.
  In both branches, an incomplete fallback set for a type-0/1 case is not an
  error by itself — it silently uses the primary pair. If a case unexpectedly
  keeps using its primary/CFD-clip centerline after you added fallback files,
  check that all three fallback files exist (not just some), and that
  --geometry-root/--post-root actually points at the directory you wrote them to.

Processed v2 output format:

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
        │   ├── R: float64 [3, 3]
        │   ├── scale: float64 scalar, exp(log_scale)
        │   ├── s_can: float64 scalar
        │   ├── R_frame: float64 [3, 3]
        │   ├── neck_pt_world: float64 [3]
        │   ├── R_equivalent: float64 [3, 3]
        │   ├── s_equivalent: float64 scalar
        │   └── t_equivalent: float64 [3]
        ├── clipped_centerline: dict
        │   ├── coordinate_system: "ghd"
        │   ├── source_coordinate_system: "world"
        │   ├── upstream_id: int
        │   ├── connected_branch_ids: int64 [N_connected_branch]
        │   ├── base_branch_ids: int64 [N_connected_branch]
        │   │   └── [upstream_id] + connected_branch_ids excluding upstream
        │   ├── branch_ranking: int64 [N_connected_branch]
        │   ├── branch_ids: int64 [N_connected_branch]
        │   │   └── base_branch_ids reordered by branch_ranking
        │   ├── branch_points: list length N_connected_branch
        │   │   └── each item float32 [N_pts, 3], converted world -> GHD
        │   │       with ghd_world_alignment(..., style="w2c_no_scale")
        │   └── branches: list length N_connected_branch
        │       └── each dict:
        │           ├── branch_id: int
        │           ├── group: int
        │           └── pts: float32 [N_pts, 3], same arrays as branch_points
        └── sources: dict
            ├── ghd_coefficients: str
            ├── landmarks: str
            ├── metrics: str
            ├── clipped_centerline: str
            ├── branch_ranking: str
            └── endpoints_manual: str
├── case_name_sanity.png
    └── non-interactive sanity-check render of reconstructed GHD mesh and
        GHD-space clipped centerlines
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DATASET_CONFIGS = {
    "imperialnhs": {
        "ghd_root": "/media/yaplab2/HDD Storage/almaha/Aneu_GHD/Fitting_Results_Final/ImperialNHS",
        "geometry_root": "/media/yaplab2/wd8tb/wenhao/angioflow/data/geometry/ImperialNHS",
        "default_aneurysm_types": None,
    },
    "aneux": {
        "ghd_root": "/media/yaplab2/HDD Storage/almaha/Aneu_GHD/fitting_results_NewAneuX/Final",
        "geometry_root": "/media/yaplab2/wd8tb/wenhao/angioflow/data/geometry/AneuX",
        "default_aneurysm_types": (1, 2),
    },
}
DEFAULT_DATASET = "aneux"
DEFAULT_GHD_ROOT = DATASET_CONFIGS[DEFAULT_DATASET]["ghd_root"]
DEFAULT_POST_ROOT = DATASET_CONFIGS[DEFAULT_DATASET]["geometry_root"]
DEFAULT_OUTPUT_DIR = "/media/yaplab2/HDD Storage/wenhao/AneuG/runtime/dataset/temp"
CANONICAL_ROOT = _REPO_ROOT / "dataset" / "canonical"

CANONICAL_BY_TYPE = {
    0: ("bifurcated", CANONICAL_ROOT / "Bifurcated"),
    1: ("sidewall", CANONICAL_ROOT / "Sidewall"),
    2: ("sidewall", CANONICAL_ROOT / "Sidewall"),
}


def ghd_world_alignment(data, R, s, t_vec, s_can, R_frame, neck_pt, 
                        style='c2w'):
    """
    canonical -> world:
    equivalent formula:
    w = c @ R_equivalent.T * s_equivalent + t_equivalent
    c = (w - t_equivalent) / s_equivalent @ R_equivalent
    R_equivalent = (R_frame.T @ R)
    s_equivalent = s * s_can
    t_equivalent = (t_vec @ R_frame) * s_can + neck_pt
    reverse ->
    c = (w - t_equivalent) / s_equivalent @ R_equivalent
    c* = (w - t_equivalent) / s_equivalent @ R_equivalent * s_equivalent
    c* = (w - t_equivalent) @ R_equivalent
    """
    if data.__class__.__module__.startswith("torch"):
        data = data.detach().cpu().numpy()

    R_equivalent = R_frame.T @ R
    t_equivalent = s_can * t_vec @ R_frame + neck_pt
    s_equivalent = s * s_can

    if style == "c2w":
        return data @ R_equivalent.T * s_equivalent + t_equivalent
    if style == "w2c":
        return (data - t_equivalent) / s_equivalent @ R_equivalent
    if style == "w2c_no_scale":
        return (data - t_equivalent) @ R_equivalent
    raise ValueError(f"Unknown alignment style: {style}")


def _so3_exp_map_numpy(w):
    """Rodrigues exponential map: axis-angle vector -> SO(3) matrix."""
    w = np.asarray(w, dtype=np.float64)
    theta = np.linalg.norm(w)
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    k = w / theta
    K = np.array([
        [0.0, -k[2], k[1]],
        [k[2], 0.0, -k[0]],
        [-k[1], k[0], 0.0],
    ], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def _load_npz(path):
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _load_pickle_npy(path):
    try:
        return np.load(path, allow_pickle=True)
    except ModuleNotFoundError as exc:
        # Compatibility fallback for files pickled with NumPy 2.x module names
        # and loaded in a NumPy 1.x environment. Avoid installing these aliases
        # unless loading actually requires them: doing so eagerly can destabilize
        # compiled packages in otherwise compatible environments.
        if not str(exc).startswith("No module named 'numpy._core"):
            raise
        np_core = getattr(np, "core", None)
        if np_core is None:
            raise
        sys.modules.setdefault("numpy._core", np_core)
        sys.modules.setdefault("numpy._core.multiarray", np_core.multiarray)
        sys.modules.setdefault("numpy._core.numeric", np_core.numeric)
        return np.load(path, allow_pickle=True)


def _primary_or_fallback(case_dir, stem, suffix):
    fallback_path = case_dir / f"{stem}_fallback{suffix}"
    primary_path = case_dir / f"{stem}{suffix}"
    if fallback_path.exists():
        return fallback_path
    if primary_path.exists():
        return primary_path
    raise FileNotFoundError(f"Missing {primary_path.name} and {fallback_path.name} in {case_dir}")


def _canonical_info(aneurysm_type):
    try:
        return CANONICAL_BY_TYPE[int(aneurysm_type)]
    except KeyError as exc:
        raise ValueError(f"Unsupported aneurysm_type={aneurysm_type}") from exc


class V1GHDPreprocessor:
    pass


class V2GHDPreprocessor:
    """
    Condense v2 GHD fitting outputs and clipped centerlines into one .npy file per case.

    The saved checkpoint is a scalar dict with:
      - case: case id
      - ghd: phi, w_rot, log_scale, t_vec
      - affine: original affine params and derived canonical/world transform params
      - clipped_centerline: reordered branch ids and branch point arrays
      - sources: input file paths used
    """

    def __init__(
        self,
        ghd_root,
        post_root,
        output_dir,
        overwrite=False,
        save_sanity=True,
        aneurysm_types=None,
        dataset_name=None,
    ):
        self.ghd_root = Path(ghd_root)
        self.post_root = Path(post_root)
        self.output_dir = Path(output_dir)
        self.overwrite = overwrite
        self.save_sanity = save_sanity
        self.aneurysm_types = (
            None if aneurysm_types is None
            else tuple(sorted(set(int(value) for value in aneurysm_types)))
        )
        self.dataset_name = dataset_name

    def iter_cases(self):
        for case_dir in sorted(self.ghd_root.iterdir()):
            if case_dir.is_dir() and (case_dir / "ghd_coefficients.npz").exists():
                yield case_dir.name

    def run(self, cases=None):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        summary = {"processed": [], "failed": [], "skipped_type": [], "sanity_failed": []}

        for case in (list(cases) if cases is not None else self.iter_cases()):
            try:
                aneurysm_type = self._load_aneurysm_type(
                    self.post_root / case / "endpoints_manual.npy"
                )
                if self.aneurysm_types is not None and aneurysm_type not in self.aneurysm_types:
                    summary["skipped_type"].append((case, aneurysm_type))
                    print(f"[SKIP] {case}: aneurysm_type={aneurysm_type}")
                    continue
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

    def save_sanity_picture(self, processed_path, ghd_reconstruct=None):
        processed_path = Path(processed_path)
        sanity_path = self.output_dir / "sanity_" / f"{processed_path.stem}_sanity.png"
        fig, _, _ = sanity_check_processed_checkpoint(
            processed_path,
            ghd_reconstruct=ghd_reconstruct,
            save_path=sanity_path,
        )
        import matplotlib.pyplot as plt
        plt.close(fig)
        return sanity_path

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

        checkpoint = {
            "case": case,
            "version": "v2",
            "aneurysm_type": aneurysm_type,
            "canonical_type": canonical_type,
            "ghd": ghd,
            "affine": {key: val for key, val in affine.items() if not key.startswith("_")},
            "clipped_centerline": _build_centerline(clipped, branch_ranking, affine),
            "sources": {key: str(path) for key, path in paths.items()},
        }
        np.save(out_path, checkpoint, allow_pickle=True)
        return out_path

    def _case_paths(self, case):
        ghd_dir = self.ghd_root / case
        post_dir = self.post_root / case
        aneurysm_type = self._load_aneurysm_type(post_dir / "endpoints_manual.npy")
        clipped_centerline, branch_ranking, clipped_reconstruction = self._centerline_paths(
            post_dir, aneurysm_type
        )
        return {
            "ghd_coefficients": ghd_dir / "ghd_coefficients.npz",
            "landmarks": ghd_dir / "landmarks.npz",
            "metrics": ghd_dir / "metrics.json",
            "clipped_centerline": clipped_centerline,
            "branch_ranking": branch_ranking,
            "clipped_reconstruction": clipped_reconstruction,
            "endpoints_manual": post_dir / "endpoints_manual.npy",
        }

    @staticmethod
    def _centerline_paths(case_dir, aneurysm_type):
        primary_centerline = case_dir / "clipped_centerline.npy"
        primary_ranking = case_dir / "branch_ranking.npy"
        primary_reconstruction = case_dir / "clipped_reconstruction.ply"
        fallback_centerline = case_dir / "clipped_centerline_fallback.npy"
        fallback_ranking = case_dir / "branch_ranking_fallback.npy"
        fallback_reconstruction = case_dir / "clipped_reconstruction_fallback.ply"

        if int(aneurysm_type) == 2:
            missing = [
                path.name for path in (
                    fallback_centerline,
                    fallback_ranking,
                    fallback_reconstruction,
                )
                if not path.exists()
            ]
            if missing:
                raise FileNotFoundError(
                    f"Type-2 case requires the fallback centerline pair in {case_dir}; "
                    f"missing {', '.join(missing)}"
                )
            return fallback_centerline, fallback_ranking, fallback_reconstruction

        if (
            fallback_centerline.exists()
            and fallback_ranking.exists()
            and fallback_reconstruction.exists()
        ):
            return fallback_centerline, fallback_ranking, fallback_reconstruction

        missing = [
            path.name for path in (
                primary_centerline,
                primary_ranking,
                primary_reconstruction,
            )
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing primary centerline pair in {case_dir}: {', '.join(missing)}"
            )
        return primary_centerline, primary_ranking, primary_reconstruction

    def _load_aneurysm_type(self, endpoints_path):
        if not endpoints_path.exists():
            raise FileNotFoundError(f"Missing endpoints_manual.npy in {endpoints_path.parent}")
        endpoints = _load_pickle_npy(endpoints_path).item()
        return int(endpoints["aneurysm_type"])

    @staticmethod
    def _build_ghd(coeffs):
        return {
            "phi": np.asarray(coeffs["phi"], dtype=np.float32),
            "w_rot": np.asarray(coeffs["w_rot"], dtype=np.float32),
            "log_scale": np.asarray(coeffs["log_scale"], dtype=np.float64),
            "t_vec": np.asarray(coeffs["t_vec"], dtype=np.float32),
        }


def _build_affine(ghd, landmarks, metrics):
    w_rot = np.asarray(ghd["w_rot"], dtype=np.float64)
    t_vec = np.asarray(ghd["t_vec"], dtype=np.float64)
    scale = float(np.exp(float(np.asarray(ghd["log_scale"]))))
    s_can = float(metrics["s_can"])
    R = _so3_exp_map_numpy(w_rot)
    R_frame = np.asarray(landmarks["R_frame"], dtype=np.float64)
    neck_pt_world = np.asarray(landmarks["neck_pt_world"], dtype=np.float64)
    t_equivalent = s_can * t_vec @ R_frame + neck_pt_world

    return {
        "R": R,
        "scale": np.array(scale, dtype=np.float64),
        "s_can": np.array(s_can, dtype=np.float64),
        "R_frame": R_frame,
        "neck_pt_world": neck_pt_world,
        "R_equivalent": R_frame.T @ R,
        "s_equivalent": np.array(scale * s_can, dtype=np.float64),
        "t_equivalent": t_equivalent,
        "_t_vec": t_vec.astype(np.float32),
    }


def _ordered_branch_ids(clipped, branch_ranking):
    connected_ids = [int(idx) for idx in clipped["connected_branch_ids"]]
    upstream_id = int(clipped["upstream_id"])
    base_ids = [upstream_id] + [idx for idx in connected_ids if idx != upstream_id]
    ranking = np.asarray(branch_ranking, dtype=np.int64)

    if len(ranking) != len(base_ids):
        raise ValueError(f"branch_ranking length {len(ranking)} does not match branch count {len(base_ids)}")
    if sorted(ranking.tolist()) != list(range(len(base_ids))):
        raise ValueError(f"branch_ranking is not a permutation of branch positions: {ranking}")
    return connected_ids, upstream_id, np.asarray(base_ids, dtype=np.int64), [base_ids[int(i)] for i in ranking]


def _align_branch_points(points, affine):
    return ghd_world_alignment(
        np.asarray(points, dtype=np.float32),
        R=affine["R"],
        s=float(affine["scale"]),
        t_vec=affine["_t_vec"],
        s_can=float(affine["s_can"]),
        R_frame=affine["R_frame"],
        neck_pt=affine["neck_pt_world"],
        style="w2c_no_scale",
    ).astype(np.float32)


def _build_centerline(clipped, branch_ranking, affine):
    connected_ids, upstream_id, base_ids, branch_ids = _ordered_branch_ids(clipped, branch_ranking)
    branches = []
    for branch_id in branch_ids:
        branch = clipped["branches"][branch_id]
        if branch.get("pts") is None:
            raise ValueError(f"Branch {branch_id} has no clipped centerline points")
        branches.append({
            "branch_id": int(branch_id),
            "group": int(branch.get("group", branch_id)),
            "pts": _align_branch_points(branch["pts"], affine),
        })

    return {
        "coordinate_system": "ghd",
        "source_coordinate_system": "world",
        "upstream_id": upstream_id,
        "connected_branch_ids": np.asarray(connected_ids, dtype=np.int64),
        "base_branch_ids": base_ids,
        "branch_ranking": np.asarray(branch_ranking, dtype=np.int64),
        "branch_ids": np.asarray(branch_ids, dtype=np.int64),
        "branch_points": [branch["pts"] for branch in branches],
        "branches": branches,
    }


def get_ghd_reconstruct(device=None):
    """
    reconstruct mesh from ghd:
    Mesh = ghd_reconstruct.forward_as_meshes(
        ghd.unsqueeze(0),
        aneurysm_type=aneurysm_type,
        denormalize_shape=True,
    )
    
    """
    from models.multi_canonical_ghd_reconstruct import MultiCanonicalGHDReconstruct
    return MultiCanonicalGHDReconstruct(canonical_root=CANONICAL_ROOT, device=device)


def _canonical_paths_for_aneurysm_type(aneurysm_type):
    _, root = _canonical_info(aneurysm_type)
    return root / "mesh.obj", root / "eigenvectors.npy"


def _load_obj_mesh(path):
    verts = []
    faces = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                idxs = [int(part.split("/")[0]) - 1 for part in line.split()[1:]]
                for i in range(1, len(idxs) - 1):
                    faces.append([idxs[0], idxs[i], idxs[i + 1]])
    return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.int64)


def _reconstruct_ghd_numpy(checkpoint, denormalize_shape=True):
    mesh_path, eigen_path = _canonical_paths_for_aneurysm_type(checkpoint["aneurysm_type"])
    verts, faces = _load_obj_mesh(mesh_path)
    eigvec = np.load(eigen_path).astype(np.float32, copy=False)
    phi = np.asarray(checkpoint["ghd"]["phi"], dtype=np.float32)

    if eigvec.shape[0] != verts.shape[0]:
        raise ValueError(
            f"Canonical vertex/eigenvector mismatch for type {checkpoint['aneurysm_type']}: "
            f"{verts.shape[0]} verts vs {eigvec.shape[0]} eigen rows"
        )
    if eigvec.shape[1] != phi.shape[0]:
        raise ValueError(
            f"GHD basis mismatch for {checkpoint.get('case', 'case')}: "
            f"{phi.shape[0]} coefficients vs {eigvec.shape[1]} eigen columns"
        )

    norm_canonical = np.linalg.norm(verts, axis=-1).max() * 1.10 * 2.50
    verts_norm = verts / norm_canonical
    verts_recon = verts_norm + eigvec @ phi
    if denormalize_shape:
        verts_recon = verts_recon * norm_canonical
        # Processed centerlines use w2c_no_scale, which preserves the fitted
        # physical case scale. Apply the same scale to the reconstructed mesh.
        fitted_scale = float(checkpoint.get("affine", {}).get("scale", 1.0))
        verts_recon = verts_recon * fitted_scale
    return verts_recon.astype(np.float32), faces


def _set_axes_equal(ax, *point_sets):
    points = [pts for pts in point_sets if len(pts) > 0]
    if not points:
        return
    points = np.concatenate([pts for pts in points if len(pts) > 0], axis=0)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = (mins + maxs) / 2.0
    radius = float((maxs - mins).max() / 2.0)
    if radius <= 0:
        radius = 1.0
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass


def _load_clipped_reconstruction_ghd(checkpoint):
    """Load the source clipped PLY and transform world vertices into GHD space."""
    clipped_path = checkpoint.get("sources", {}).get("clipped_reconstruction")
    if clipped_path is None:
        return None, None

    import trimesh

    loaded = trimesh.load(clipped_path, process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = tuple(loaded.geometry.values())
        if not geometries:
            raise ValueError(f"No mesh geometry found in {clipped_path}")
        loaded = trimesh.util.concatenate(geometries)

    verts_world = np.asarray(loaded.vertices, dtype=np.float32)
    faces = np.asarray(loaded.faces, dtype=np.int64)
    affine = checkpoint["affine"]
    verts_ghd = ghd_world_alignment(
        verts_world,
        R=np.asarray(affine["R"]),
        s=float(affine["scale"]),
        t_vec=np.asarray(checkpoint["ghd"]["t_vec"]),
        s_can=float(affine["s_can"]),
        R_frame=np.asarray(affine["R_frame"]),
        neck_pt=np.asarray(affine["neck_pt_world"]),
        style="w2c_no_scale",
    )
    return verts_ghd.astype(np.float32), faces


def sanity_check_processed_checkpoint(
    processed_path,
    ghd_reconstruct=None,
    save_path=None,
    device=None,
    denormalize_shape=True,
):
    """
    Overlay the reconstructed GHD mesh, source clipped mesh, and centerlines.

    Parameters
    ----------
    processed_path : str or Path
        Path to one processed case_name.npy checkpoint.
    ghd_reconstruct : MultiCanonicalGHDReconstruct or GHD_Reconstruct, optional
        Optional legacy reconstructor. If None, a NumPy-only reconstructor is used
        to avoid importing the PyTorch3D/Open3D stack during preprocessing.
    save_path : str or Path, optional
        If provided, save the sanity-check figure here.
    device : torch.device or str, optional
        Device used for reconstruction. Defaults to ghd_reconstruct's device.
    denormalize_shape : bool
        Passed to ghd_reconstruct.ghd_forward_as_Meshes.

    Returns
    -------
    fig, ax, mesh
        Matplotlib figure/axes and the reconstructed PyTorch3D Meshes object.
    """
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    checkpoint = np.load(processed_path, allow_pickle=True).item()
    aneurysm_type = int(checkpoint["aneurysm_type"])
    if ghd_reconstruct is None:
        mesh = None
        verts, faces = _reconstruct_ghd_numpy(checkpoint, denormalize_shape=denormalize_shape)
    else:
        mesh = _reconstruct_with_backend(checkpoint, ghd_reconstruct, aneurysm_type, device, denormalize_shape)
        verts, faces = (item.detach().cpu().numpy() for item in (mesh.verts_list()[0], mesh.faces_list()[0]))

    centerline = checkpoint["clipped_centerline"]
    if centerline.get("coordinate_system") != "ghd":
        raise ValueError(f"Expected GHD-space centerlines, got {centerline.get('coordinate_system')}")
    branch_points = [np.asarray(pts, dtype=np.float32) for pts in centerline["branch_points"]]

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

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(branch_points), 1)))
    branch_ids = centerline["branch_ids"]
    for idx, pts in enumerate(branch_points):
        color = colors[idx % len(colors)]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], linewidth=2.0, color=color,
                label=f"branch {int(branch_ids[idx])}")
        ax.scatter(*pts[0], s=18, color=color)

    case = checkpoint.get("case", Path(processed_path).stem)
    ax.set_title(
        f"{case} (type {aneurysm_type}: {checkpoint.get('canonical_type', 'unknown')})\n"
        "GHD reconstruction: gray | clipped reconstruction: orange"
    )
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    geometry_points = [verts, *branch_points]
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


def _reconstruct_with_backend(checkpoint, ghd_reconstruct, aneurysm_type, device, denormalize_shape):
    import torch

    case_reconstruct = ghd_reconstruct.get(aneurysm_type) if hasattr(ghd_reconstruct, "get") else ghd_reconstruct
    device = torch.device(device or getattr(case_reconstruct.canonical_Meshes, "device", torch.device("cpu")))
    ghd = torch.as_tensor(checkpoint["ghd"]["phi"], dtype=torch.float32, device=device).unsqueeze(0)
    if hasattr(ghd_reconstruct, "forward_as_meshes"):
        mesh = ghd_reconstruct.forward_as_meshes(
            ghd, aneurysm_type=aneurysm_type, denormalize_shape=denormalize_shape
        )
    else:
        mesh = case_reconstruct.ghd_forward_as_Meshes(
            ghd, denormalize_shape=denormalize_shape
        )

    if denormalize_shape:
        fitted_scale = float(checkpoint.get("affine", {}).get("scale", 1.0))
        mesh = mesh.update_padded(mesh.verts_padded() * fitted_scale)
    return mesh


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess ImperialNHS or AneuX GHD checkpoints into one .npy file per case."
    )
    parser.add_argument(
        "--dataset",
        choices=tuple(DATASET_CONFIGS),
        default=DEFAULT_DATASET,
        help="Select default GHD/geometry roots (default: imperialnhs).",
    )
    parser.add_argument("--ghd-root", default=None, help="Override the selected dataset's GHD root.")
    parser.add_argument(
        "--geometry-root",
        "--post-root",
        dest="geometry_root",
        default=None,
        help="Override the selected dataset's geometry root (--post-root is a compatibility alias).",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument(
        "--aneurysm-types",
        nargs="+",
        type=int,
        choices=(0, 1, 2),
        default=None,
        metavar="TYPE",
        help="Only process these labels. AneuX defaults to 1 2; ImperialNHS defaults to all.",
    )
    parser.add_argument("--overwrite", action="store_true", default=True)
    parser.add_argument(
        "--save-sanity",
        action="store_true",
        help="Also render sanity PNGs. This imports the reconstruction stack and may be environment-sensitive.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    dataset_config = DATASET_CONFIGS[args.dataset]
    aneurysm_types = (
        args.aneurysm_types
        if args.aneurysm_types is not None
        else dataset_config["default_aneurysm_types"]
    )
    if args.dataset == "aneux" and aneurysm_types is not None:
        unsupported = sorted(set(aneurysm_types) - {1, 2})
        if unsupported:
            raise ValueError(f"AneuX preprocessing is restricted to aneurysm types 1 and 2; got {unsupported}")

    preprocessor = V2GHDPreprocessor(
        ghd_root=args.ghd_root or dataset_config["ghd_root"],
        post_root=args.geometry_root or dataset_config["geometry_root"],
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        save_sanity=args.save_sanity,
        aneurysm_types=aneurysm_types,
        dataset_name=args.dataset,
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
python dataset/preprocess_AneuX.py --dataset imperialnhs
python dataset/preprocess_AneuX.py --dataset aneux --aneurysm-types 1 2 \
    --output-dir "/media/yaplab2/HDD Storage/wenhao/AneuG/runtime/dataset/temp" \
    --save-sanity

"""
