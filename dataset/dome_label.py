"""
Dome-region ground truth for the warped canonical mesh.

Companion to dataset/auto_uncap.py (which labels the end caps). Together they
give the per-vertex segmentation target for the endcap/dome predictor:

    class 0  vessel wall
    class 1  end cap        (auto_uncap.compute_caps, one region per opening)
    class 2  aneurysm dome  (this module)

Because the canonical topology is FIXED under GHD deformation, a label is just
a boolean of length V_canonical (4143 bifurcated / 2590 sidewall), directly
comparable across cases.

TWO DATASET CONVENTIONS, both covering the assembled corpus completely
(283/283 AneuX, 242/242 ImperialNHS):

  ImperialNHS  label.nrrd, voxel label == 2 is the segmented dome mask (the
               same mask AneuSeg's vessel_clipping.py protects during clipping).
               A world->voxel lookup per vertex gives the mask directly.

  AneuX        dome_sac.ply, a dedicated pre-clipped sac SURFACE PATCH in the
               same world frame. It is NOT watertight, so there is no
               inside-test.

THE ANEUX TEST IS RELATIVE, NOT AN ABSOLUTE DISTANCE. The obvious approach --
"dome if within d mm of the sac" -- needs a threshold that does not transfer:
the median vertex-to-sac distance ranges 0.5-2.9 mm across cases, mixing sac
size with GHD fitting error, so any fixed d over-labels small sacs and
under-labels poorly fitted ones. Instead, note that the sac patch is a SUBSET
of the same vessel surface the full clipped mesh represents, so for every
vertex

    dist_to_sac  >=  dist_to_full_mesh

with equality exactly where the nearest surface IS the sac. Labelling on the
GAP between the two distances cancels the fitting error, which is common to
both, and leaves a quantity that means the same thing on every case.
"""

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Gap (mm) below which a vertex counts as sitting ON the sac patch. Small and
# absolute is fine HERE -- unlike a raw distance this is a residual after the
# common fitting error cancels, so it does not scale with sac size.
DEFAULT_GAP_TOL = 0.2
# Voxel dilation for the nrrd mask: a fitted vertex can fall just outside the
# segmented dome purely from fitting error, which leaves a ragged boundary.
DEFAULT_DILATE = 1


def world_from_aligned(verts_aligned, R, s, t):
    """Inverse of Stage 1's similarity: aligned = s*(world @ R.T) + t."""
    return ((np.asarray(verts_aligned, dtype=np.float64) - np.asarray(t)) @ np.asarray(R)) / float(s)


def dome_mask_nrrd(case_dir, verts_world, dilate=DEFAULT_DILATE):
    """Per-vertex dome mask from label.nrrd's voxel label == 2 (ImperialNHS)."""
    try:
        import SimpleITK as sitk
    except ImportError:
        return None
    p = Path(case_dir) / "label.nrrd"
    if not p.exists():
        return None
    itk = sitk.ReadImage(str(p))
    mask = sitk.GetArrayFromImage(itk) == 2
    if not mask.any():
        return None
    if dilate > 0:
        try:
            from scipy.ndimage import binary_dilation
            mask = binary_dilation(mask, iterations=int(dilate))
        except ImportError:
            pass
    sp = np.array(itk.GetSpacing())
    og = np.array(itk.GetOrigin())
    vc = np.round((np.asarray(verts_world) - og) / sp).astype(int)
    sh = mask.shape
    return mask[np.clip(vc[:, 2], 0, sh[0] - 1),
                np.clip(vc[:, 1], 0, sh[1] - 1),
                np.clip(vc[:, 0], 0, sh[2] - 1)].astype(bool)


def dome_mask_sac(case_dir, verts_world, full_mesh_path, gap_tol=DEFAULT_GAP_TOL):
    """Per-vertex dome mask from dome_sac.ply (AneuX), via the relative test.

    dome  <=>  dist(v, sac) - dist(v, full clipped mesh)  <  gap_tol
    """
    import trimesh

    sac_p = Path(case_dir) / "dome_sac.ply"
    if not sac_p.exists() or not Path(full_mesh_path).exists():
        return None
    sac = trimesh.load(sac_p, file_type="ply", process=False)
    full = trimesh.load(full_mesh_path, process=False)
    if isinstance(full, trimesh.Scene):
        full = trimesh.util.concatenate(tuple(full.geometry.values()))
    if len(sac.faces) == 0 or len(full.faces) == 0:
        return None
    V = np.asarray(verts_world, dtype=np.float64)
    d_sac = np.abs(trimesh.proximity.ProximityQuery(sac).signed_distance(V))
    d_full = np.abs(trimesh.proximity.ProximityQuery(full).signed_distance(V))
    return (d_sac - d_full) < gap_tol


def dome_mask_for_case(dataset, case_dir, verts_world, full_mesh_path=None, **kw):
    """Dispatch on dataset convention. Returns a bool mask or None."""
    if dataset == "AneuX":
        return dome_mask_sac(case_dir, verts_world, full_mesh_path,
                             gap_tol=kw.get("gap_tol", DEFAULT_GAP_TOL))
    return dome_mask_nrrd(case_dir, verts_world, dilate=kw.get("dilate", DEFAULT_DILATE))


# ── proposal plumbing ────────────────────────────────────────────────────────
# The processed record is deliberately lean (no Stage-1 frame, no geometry
# paths), so an automatic dome proposal has to reach back to the assembled case
# directory for landmarks.npz / alignment_result.npy / metrics.json. A shape
# with no such directory -- a GENERATED one -- simply has no proposal, which is
# why every entry point here returns None rather than raising: the caller falls
# through to manual brushing.

def stage1_transform(assembled_case_dir):
    """(R, s, t) of Stage 1 for an assembled case, or None if unavailable."""
    import json

    cd = Path(assembled_case_dir)
    lm_p = cd / "landmarks.npz"
    if not lm_p.exists():
        return None
    lm = np.load(lm_p, allow_pickle=True)
    R = np.asarray(lm["R_frame"], dtype=np.float64)
    t = np.asarray(lm["t_vec"], dtype=np.float64)

    ar = cd / "alignment_result.npy"
    if ar.exists():
        res = np.load(ar, allow_pickle=True).item()
        if "s" in res:
            return R, float(res["s"]), t
    # arap_init runs write no alignment_result.npy -- recover s from the
    # aneurysm-centroid correspondence (world in endpoints_manual.npy, aligned
    # in landmarks.npz). Reproduces the stored value to ~4e-9 where both exist.
    mj = cd / "metrics.json"
    if not mj.exists():
        return None
    geo = Path(json.loads(mj.read_text()).get("case_dir", ""))
    ep_p = geo / "endpoints_manual.npy"
    if not ep_p.exists():
        return None
    c_world = np.asarray(np.load(ep_p, allow_pickle=True).item()["aneurysm_centroid"], dtype=np.float64)
    x = c_world @ R.T
    if float(x @ x) < 1e-12:
        return None
    return R, float((np.asarray(lm["aneurysm_centroid"], dtype=np.float64) - t) @ x / (x @ x)), t


def geometry_dir_for(assembled_case_dir):
    """The case's geometry directory, per its own metrics.json."""
    import json

    mj = Path(assembled_case_dir) / "metrics.json"
    if not mj.exists():
        return None
    g = json.loads(mj.read_text()).get("case_dir", "")
    return Path(g) if g else None


def dome_proposal(dataset, case, verts_aligned, assembled_root, **kw):
    """Automatic dome mask for one case, or None when no source exists.

    None is the normal answer for a generated shape -- it has no geometry
    directory, no dome_sac.ply and no label.nrrd -- and the caller is expected
    to fall through to manual labelling rather than treat it as an error.
    """
    cd = Path(assembled_root) / dataset / case
    if not cd.is_dir():
        return None
    tf = stage1_transform(cd)
    geo = geometry_dir_for(cd)
    if tf is None or geo is None or not geo.is_dir():
        return None
    R, s, t = tf
    verts_world = world_from_aligned(verts_aligned, R, s, t)
    try:
        from ghd.fitting.alignment import resolve_case_paths
        full = Path(resolve_case_paths(geo)["clipped_mesh"])
    except Exception:
        full = None
    return dome_mask_for_case(dataset, geo, verts_world, full_mesh_path=full, **kw)
