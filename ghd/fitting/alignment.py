"""
alignment.py -- fit a similarity transform (rotation + uniform scale +
translation) taking a real case's shape INTO the canonical template's frame,
as a standalone, testable first step before adding GHD-coefficient fitting
on top (which will reuse this alignment, not reimplement it).

Inputs, per case (from preprocesser_manual_stable.py's pipeline):
  - merged_centerline.npy + clipped_centerline.npy (or the _fallback pair,
    preferred when present -- see resolve_case_paths) for the REAL per-branch
    centerline geometry.
  - clipped_reconstruction.ply (or _fallback) for the surface.
  - branch_ranking.npy (or _fallback) for branch correspondence order.
Inputs, canonical: dataset/canonical/<Type>/canonical_topology.npy (from
dataset/canonical/record_topology.py) + mesh.obj.

Why NOT merged_reconstruction.ply / clipped_centerline.npy's own points directly:
merged_reconstruction.ply is the clipped mesh with a SYNTHETIC tube grafted on
at the clip plane (for cross-section-propagation / GHD-input purposes) --
clipped_centerline.npy's stored branch points are exactly that synthetic
tube's guide centerline, not real anatomy. The canonical mesh has no such
synthetic parts (it's the full organic template), so aligning against the
synthetic extension would be comparing two geometrically different things.
Instead this script uses the REAL anatomical portion of each branch -- from
the neck/dome end out to where the case was actually clipped -- recovered by
finding where clipped_centerline.npy's remaining (post-clip) points begin
within merged_centerline.npy's full branch (same nearest-position-match
technique already used in AneuSeg/reclip.py to recover historical clip
distances), and clipped_reconstruction.ply (the real kept mesh, no
extension) for the surface term.

Branch correspondence: case rank k (branch_ranking.npy, inlet-first, already
confirmed during preprocessing) <-> canonical opening k (record_topology.py's
picked sequence, same inlet-first convention). branch_ranking's values are
RENDER indices into forward_fusion_info.npz's cpcd_glo, i.e. positions in
[upstream_id] + rest-of-connected_branch_ids (see compute_branch_ids_order) --
NOT raw connected_branch_ids values -- so mapping rank -> actual branch id
needs that reindirection.

Fitting: target (case) -> canonical. Loss = mean per-branch centerline
chamfer + surface chamfer + mean per-branch endpoint L2, EQUAL weight by
default (--w-* to override). Closed-form Umeyama similarity-transform fit
(rotation + scale + translation from corresponding points: each branch's
endpoint plus the neck/dome centroid) gives a deterministic initial guess --
chamfer terms have no closed-form solution (nearest-neighbour correspondence
changes with the transform, same reason ICP is iterative), so gradient
descent (pytorch3d.loss.chamfer_distance, autograd) refines from there.

conda activate new
python ghd/fitting/alignment.py --case-dir "/media/yaplab2/wd8tb/wenhao/angioflow/data/geometry/ImperialNHS/0bjVkRg2pM_aneurysm1" --sidewall-canonical-dir dataset/canonical/Sidewall --bifurcated-canonical-dir dataset/canonical/Bifurcated
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import trimesh
from pytorch3d.loss import chamfer_distance
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Repair tooling for the AneuX_stable-style gap: a fallback (short) clip mesh
# exists but its paired fallback centerline was never generated (older
# pipeline version, see resolve_case_paths' docstring). Lives in a different
# conda env (VMTK isn't installed in this repo's own 'new' env), so it's
# invoked as a subprocess, not imported in-process.
VMTK_PYTHON = "/home/yaplab2/miniconda3/envs/vmtk_autogen/bin/python3"
REPAIR_SCRIPT = ROOT / "AneuSeg" / "repair_centerline_with_complex.py"


def ensure_fallback_centerline(case_dir):
    """No-op unless this case has a fallback mesh (clipped_reconstruction_fallback.ply)
    but no matching fallback centerline (clipped_centerline_fallback.npy) -- the exact
    gap AneuX_stable's older preprocessing pipeline left in ~143 sidewall cases
    (confirmed via dataset scan, see the AneuX_stable support plan). When it applies,
    auto-repairs the case IN PLACE by running repair_centerline_with_complex.py
    --auto (unattended: no popups/prompts, branch ranking left untouched since it
    already exists for every one of these cases) in the vmtk_autogen env, before
    this pipeline reads anything. Falls back to a warning (proceeding with the
    long/main clip's centerline, the previous -- known imprecise -- behavior) if the
    repair tool/env isn't available or the repair itself fails, rather than crashing."""
    case_dir = Path(case_dir)
    if not (case_dir / "clipped_reconstruction_fallback.ply").exists():
        return
    if (case_dir / "clipped_centerline_fallback.npy").exists():
        return
    if not Path(VMTK_PYTHON).exists() or not REPAIR_SCRIPT.exists():
        print(f"  Warning: {case_dir.name} is missing its fallback centerline and the "
              f"auto-repair tool/env isn't available here -- branch endpoints will be "
              f"derived from the long/main clip's centerline instead (known imprecise "
              f"for this case).")
        return
    try:
        endpoints_ = np.load(case_dir / "endpoints_manual.npy", allow_pickle=True).item()
        aneurysm_type = int(endpoints_["aneurysm_type"])
    except Exception:
        aneurysm_type = 1
    scenario = "type2_incomplete_fallback" if aneurysm_type == 2 else "type1_partial_fallback"
    print(f"  Auto-repairing missing fallback centerline for {case_dir.name} "
          f"(scenario={scenario}, unattended)...", flush=True)
    result = subprocess.run(
        [VMTK_PYTHON, str(REPAIR_SCRIPT),
         "--post-root-dir", str(case_dir.parent),
         "--scenario", scenario, "--case", case_dir.name, "--auto"],
        stdin=subprocess.DEVNULL, capture_output=True, text=True,
    )
    if not (case_dir / "clipped_centerline_fallback.npy").exists():
        print(f"  Warning: auto-repair did not produce a fallback centerline for "
              f"{case_dir.name} (returncode={result.returncode}) -- proceeding with the "
              f"long/main clip's centerline instead. Repair output tail:\n"
              f"{(result.stdout or '')[-1500:]}\n{(result.stderr or '')[-1500:]}")
    else:
        print(f"  Auto-repair succeeded for {case_dir.name}.")


# ── case-side loading (fallback-aware, real-anatomy-only) ──────────────────

def resolve_case_paths(case_dir):
    """Fallback files (type-1/2 cases' short/sac-protected clip) are
    preferred over the main/long CFD clip whenever present -- same
    convention established for AneuSeg/reclip.py, now applied to which
    files alignment reads from. Gated on clipped_reconstruction_fallback.ply
    itself (the file that actually matters), NOT on
    clipped_centerline_fallback.npy -- some datasets (e.g. AneuX_stable,
    processed by an older pipeline version) have the former without the
    latter. Centerline geometry doesn't diverge between clip variants
    (only mesh extent + branch_ranking do), so its suffix is gated
    independently and falls back to the plain centerline file when no
    fallback-specific one exists."""
    case_dir = Path(case_dir)
    use_fallback = (case_dir / "clipped_reconstruction_fallback.ply").exists()
    mesh_suffix = "_fallback" if use_fallback else ""
    cl_suffix = "_fallback" if (use_fallback and
                                (case_dir / "clipped_centerline_fallback.npy").exists()) else ""
    return {
        "use_fallback": use_fallback,
        "merged_centerline": case_dir / f"merged_centerline{cl_suffix}.npy",
        "clipped_centerline": case_dir / f"clipped_centerline{cl_suffix}.npy",
        "clipped_mesh": case_dir / f"clipped_reconstruction{mesh_suffix}.ply",
        "branch_ranking": case_dir / f"branch_ranking{mesh_suffix}.npy",
        "endpoints": case_dir / "endpoints_manual.npy",
    }


def _ensure_outward(pts, ref_point):
    """Orient so pts[0] is the end nearest ref_point (neck/dome side) --
    same convention as AneuSeg/IAgents/tools/vessel_clipping.py's
    _ensure_outward, dataset/preprocess_endcaps.py, etc."""
    pts = np.asarray(pts)
    if np.linalg.norm(pts[-1] - ref_point) < np.linalg.norm(pts[0] - ref_point):
        return pts[::-1]
    return pts


def compute_branch_ids_order(clipped_cl):
    """branch_ranking.npy's values are indices into this order (render/
    cpcd_glo index space), NOT raw connected_branch_ids values -- matches
    forward_mesh_fusion_v2's own branch_ids construction exactly, so
    branch_ids[branch_ranking[k]] recovers the true branches[] index."""
    upstream_id = clipped_cl["upstream_id"]
    connected = clipped_cl["connected_branch_ids"]
    return [upstream_id] + [i for i in connected if i != upstream_id]


def extract_real_branch_segment(merged_cl, clipped_cl, branch_id, aneurysm_centroid):
    """The REAL anatomical portion of a branch: merged_centerline.npy's full
    branch, oriented dome-first, truncated at the point where
    clipped_centerline.npy's synthetic-extension remainder begins (i.e. where
    the case was actually clipped). clipped_centerline.npy doesn't store the
    cut arc-length directly, so it's recovered by nearest-position match --
    same technique as AneuSeg/reclip.py's _recover_old_distances."""
    full_pts = _ensure_outward(merged_cl["branches"][branch_id]["pts"], aneurysm_centroid)
    remaining = clipped_cl["branches"][branch_id].get("pts")
    if remaining is None or len(remaining) == 0:
        return full_pts  # never clipped away / fully consumed -- use the whole real branch
    remaining = np.asarray(remaining)
    d = np.linalg.norm(full_pts - remaining[0], axis=1)
    cut_idx = int(np.argmin(d))
    if cut_idx < 2:
        # degenerate: clip point resolved almost at the dome end -- keep a minimal
        # 2-point stub rather than an empty/1-point branch that breaks chamfer/endpoint math.
        cut_idx = min(2, len(full_pts) - 1)
    return full_pts[:cut_idx + 1]


# ── closing the case's open mesh, for volume computation only ──────────────
# Real cases' clipped_reconstruction.ply is open at every branch (that's what
# "clipped" means) -- trimesh.volume on an open mesh isn't physically
# meaningful, and trimesh.repair.fill_holes silently no-ops on holes this
# large. Close it properly by reusing this pipeline's OWN tangent-based
# extrusion (cfd_mesher.vessel_reconstruct.forward_mesh_fusion_v2, the same
# mechanism preprocesser_manual_stable.py's real pipeline already uses to
# build merged_reconstruction.ply -- tuned here to a short, fixed extrusion
# instead of a full CFD-length tube) + planarize_openings (flattens the new
# ring) + a simple centroid fan-cap (safe once the ring is flat and regular
# -- planarize_openings only repositions the ring, it doesn't add cap faces).
# Nothing is written to disk; purely in-memory, read-only w.r.t. the case folder.

def _find_boundary_loops(faces):
    """Ordered walk per boundary loop (edges used by exactly one face) --
    same algorithm as dataset/canonical/record_topology.py's
    find_boundary_loops, duplicated locally to keep this script self-contained."""
    from collections import Counter, defaultdict
    edge_count = Counter(
        tuple(sorted(e))
        for f in faces
        for e in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])]
    )
    boundary_edges = [(a, b) for (a, b), c in edge_count.items() if c == 1]
    adj = defaultdict(list)
    for a, b in boundary_edges:
        adj[a].append(b)
        adj[b].append(a)
    visited, loops = set(), []
    for start in list(adj):
        if start in visited:
            continue
        loop, cur, prev = [start], start, None
        visited.add(start)
        while True:
            nxt = next((v for v in adj[cur] if v != prev and v not in visited), None)
            if nxt is None:
                break
            loop.append(nxt)
            visited.add(nxt)
            prev, cur = cur, nxt
        loops.append(np.array(loop))
    return loops


def _fan_cap_open_mesh(mesh_pv, return_opening_info=False):
    """Closes a mesh's open boundary loops with a simple centroid fan cap.
    Coarse (fine for a volume estimate, not a precise reconstruction) but
    robust here specifically because it's called AFTER planarize_openings,
    so each ring is already flat and evenly spaced.

    If return_opening_info=True, also returns (opening_loops, cap_face_loop_id):
    opening_loops is the list of per-loop ring vertex-index arrays (into the
    OUTPUT mesh's vertex array -- same indices as input, since capping only
    appends new verts/faces), and cap_face_loop_id is an int array over the
    OUTPUT face array, -1 for original (body) faces and the loop's index
    (into opening_loops) for its newly-added fan-cap triangles -- lets
    callers tell EVERY detected hole apart, not just cap-vs-body, since not
    every closed loop is a real anatomical opening (see
    _real_opening_loops below: small mesh-defect holes get closed here same
    as real openings, for watertightness, but must be filtered out before
    being treated as opening/ring data for the experimental loss terms)."""
    import pyvista as pv

    verts = np.asarray(mesh_pv.points)
    faces = mesh_pv.faces.reshape(-1, 4)[:, 1:]
    loops = _find_boundary_loops(faces)
    if not loops:
        if return_opening_info:
            return mesh_pv, [], np.full(len(faces), -1, dtype=np.int64)
        return mesh_pv
    extra_verts, extra_faces = [verts], [faces]
    cap_loop_id_parts = [np.full(len(faces), -1, dtype=np.int64)]
    offset = len(verts)
    for loop_idx, loop in enumerate(loops):
        centroid = verts[loop].mean(axis=0)
        extra_verts.append(centroid[None])
        n = len(loop)
        tri = np.stack([loop, np.roll(loop, -1), np.full(n, offset)], axis=1)
        extra_faces.append(tri)
        cap_loop_id_parts.append(np.full(n, loop_idx, dtype=np.int64))
        offset += 1
    all_verts = np.vstack(extra_verts)
    all_faces = np.vstack(extra_faces)
    faces_pv = np.hstack([np.full((len(all_faces), 1), 3), all_faces]).ravel()
    result = pv.PolyData(all_verts, faces_pv)
    if return_opening_info:
        return result, loops, np.concatenate(cap_loop_id_parts)
    return result


def _build_closed_case_mesh(paths, extrude_length=0.25):
    """Closes the case's open clipped mesh (extrude ~extrude_length mm along
    each branch's centerline tangent, planarize, fan-cap) and returns it as a
    trimesh.Trimesh. extrude_length is deliberately uniform across every case
    (same value regardless of vessel size) so the closure itself doesn't bias
    the volume-matching loss.

    "Always after existing nodes" (never folds back into the mesh): the
    extrusion is pure linear extrapolation forward along the branch's own
    end tangent from its last real point (forward_mesh_fusion_v2's own
    construction, unchanged) -- monotonically increasing arc-length by
    construction, not a heuristic that could fail. max_cl_length/min_cl_length
    are set far below any real branch's length, so effectively none of the
    REAL centerline is used before the controlled extrude_length extrusion
    kicks in -- the two don't stack into more than ~extrude_length total.

    This is also what gets saved as final_aligned.obj (transformed) -- the
    old prep_case_for_ghd.py's final_aligned.obj was likewise the capped/
    watertight mesh (VMTK capping there), not the raw open clip, since
    downstream GHD fitting's occupancy/volume terms need a closed target.

    If the direct extrude->planarize->cap order leaves the result non-
    watertight (seen on some AneuX_stable short-clip meshes with irregular/
    jagged boundary loops -- extruding from a jagged ring propagates the
    jaggedness into the tube, which can then trip up planarize_openings/
    capping), automatically retries with an extra planarize pass on the RAW
    open mesh's own boundary loops FIRST, before extrusion -- regularizing
    the ring extrusion starts from, rather than the one it ends on.
    Confirmed empirically to recover watertightness on a case the direct
    order left open (see AneuX_stable support work). Only pays this extra
    cost when the direct order actually fails.
    """
    from cfd_mesher.vessel_reconstruct import forward_mesh_fusion_v2
    from cfd_mesher.patching import planarize_openings

    def _extrude_planarize_cap(mesh_path):
        merged = forward_mesh_fusion_v2(
            save_dir=None,
            r_clipped_mesh_filename=str(mesh_path),
            r_clipped_cl_filename=str(paths["clipped_centerline"]),
            r_branch_ranking_filename=str(paths["branch_ranking"]),
            w_forward_fusion_info_filename=None,
            w_merged_mesh_filename=None,
            max_cl_length=0.01,     # ~none of the real centerline -- extrude_length below does the work
            min_cl_length=1.0,      # safely below any real branch's true length -- doesn't auto-extend
            extrude_length=extrude_length,
            extrude_outlets=True,
            extrude_inlet=True,
        )
        flat = planarize_openings(merged, r_forward_fusion_info_filename=None, smooth=True)
        capped, opening_loops, cap_face_loop_id = _fan_cap_open_mesh(flat, return_opening_info=True)
        capped_tm = trimesh.Trimesh(vertices=np.asarray(capped.points),
                                    faces=capped.faces.reshape(-1, 4)[:, 1:], process=False)
        return capped_tm, opening_loops, cap_face_loop_id

    capped_tm, opening_loops, cap_face_loop_id = _extrude_planarize_cap(paths["clipped_mesh"])

    if not capped_tm.is_watertight:
        import pyvista as pv
        raw = pv.read(str(paths["clipped_mesh"]))
        pre_flat = planarize_openings(raw, r_forward_fusion_info_filename=None, smooth=True)
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".ply")
        os.close(tmp_fd)
        try:
            pre_flat.save(tmp_path)
            capped_tm2, opening_loops2, cap_face_loop_id2 = _extrude_planarize_cap(tmp_path)
        finally:
            os.unlink(tmp_path)
        if capped_tm2.is_watertight:
            print(f"  Recovered watertightness via pre-planarize-before-extrude fallback -- "
                  f"{paths['clipped_mesh']}")
            capped_tm, opening_loops, cap_face_loop_id = capped_tm2, opening_loops2, cap_face_loop_id2

    if not capped_tm.is_watertight:
        print(f"  Warning: case volume-closure mesh not fully watertight even after the "
              f"pre-planarize fallback (volume estimate may be inaccurate) -- {paths['clipped_mesh']}")
    capped_tm.opening_loops = opening_loops
    capped_tm.cap_face_loop_id = cap_face_loop_id
    return capped_tm


def load_dome_points_from_nrrd(case_dir, mesh_vertices, n_samples=2000):
    """DATASET-SPECIFIC dome-region loader for the convention used HERE
    (ImperialNHS via AneuSeg/preprocesser_manual_stable.py): label.nrrd's
    voxel label==2 is the segmented aneurysm dome mask -- the exact same
    mask AneuSeg/IAgents/tools/vessel_clipping.py already reads to protect
    dome vertices during clipping (same world->voxel lookup, reused here
    verbatim). Returns the subset of `mesh_vertices` that fall inside it
    (randomly subsampled to n_samples if larger), or None if label.nrrd is
    missing/empty -- so callers skip the dome loss term gracefully instead
    of crashing, rather than assuming every dataset has this file.

    THIS IS A PLACEHOLDER FOR THIS DATASET, NOT A FIXED INTERFACE. A future
    dataset may have no voxel label at all and instead provide dome
    location via, say, a manually-picked dome-region mesh/point-cloud or a
    different segmentation convention. To support that, write a new
    function with the same signature -- (case_dir, mesh_vertices, n_samples)
    -> (n, 3) float32 array | None -- and pass it as load_case_data's
    dome_points_loader argument; nothing else in this file (alignment_loss,
    fit_alignment, render_sanity_images) needs to change."""
    try:
        import SimpleITK as sitk
    except ImportError:
        return None
    nrrd_path = Path(case_dir) / "label.nrrd"
    if not nrrd_path.exists():
        return None
    label_itk = sitk.ReadImage(str(nrrd_path))
    label_array = sitk.GetArrayFromImage(label_itk)
    dome_mask = (label_array == 2)
    if not dome_mask.any():
        return None
    spacing = np.array(label_itk.GetSpacing())
    origin = np.array(label_itk.GetOrigin())
    voxel_coords = np.round((mesh_vertices - origin) / spacing).astype(int)
    shape = dome_mask.shape
    vx = np.clip(voxel_coords[:, 0], 0, shape[2] - 1)
    vy = np.clip(voxel_coords[:, 1], 0, shape[1] - 1)
    vz = np.clip(voxel_coords[:, 2], 0, shape[0] - 1)
    in_dome = dome_mask[vz, vy, vx].astype(bool)
    dome_pts = np.asarray(mesh_vertices)[in_dome]
    if len(dome_pts) < 10:
        return None
    if len(dome_pts) > n_samples:
        idx = np.random.default_rng(0).choice(len(dome_pts), n_samples, replace=False)
        dome_pts = dome_pts[idx]
    return dome_pts.astype(np.float32)


def load_dome_points_from_mesh(case_dir, mesh_vertices, n_samples=2000):
    """DATASET-SPECIFIC dome-region loader for AneuX_stable's convention:
    dome_sac.ply is a dedicated, pre-clipped aneurysm-sac mesh, confirmed to
    sit in the same world coordinate frame as clipped_reconstruction.ply (no
    extra registration needed). Unlike load_dome_points_from_nrrd (which
    must subset mesh_vertices via a voxel mask, since a mask is all it has),
    this loader samples directly from dome_sac.ply's own surface -- higher
    fidelity than a masked-vertex subset, and simpler. mesh_vertices is
    unused here but kept in the signature to match the dome_points_loader
    contract load_case_data expects (see load_dome_points_from_nrrd's own
    docstring for that contract)."""
    path = Path(case_dir) / "dome_sac.ply"
    if not path.exists():
        return None
    mesh = trimesh.load(path, process=False)
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        return None
    pts, _ = trimesh.sample.sample_surface(mesh, min(n_samples, len(mesh.faces) * 4))
    return np.asarray(pts, dtype=np.float32)


DOME_LOADERS = {
    "nrrd": load_dome_points_from_nrrd,
    "mesh": load_dome_points_from_mesh,
    "none": None,
}


def load_case_data(case_dir, n_surface_samples=4000, extrude_length=0.25,
                   dome_points_loader=load_dome_points_from_nrrd,
                   auto_repair_fallback_centerline=True):
    """Returns dict: branch_segments (list, rank-ordered, dome-first real
    anatomy only), branch_endpoints ([n_branches,3], the far/clip end of each
    segment), aneurysm_centroid, surface_points ([n_surface_samples,3], from
    the real clipped mesh -- no synthetic extension, used for the ALIGNMENT
    loss only), volume (mm^3, from closed_mesh), closed_mesh (trimesh.Trimesh,
    watertight -- extruded/planarized/fan-capped once here and reused for
    BOTH the volume figure and, transformed, as the final_aligned.obj
    checkpoint export -- see save_stage1_checkpoints), aneurysm_type.

    auto_repair_fallback_centerline: see ensure_fallback_centerline -- on by
    default, no-ops instantly for any case that doesn't need it."""
    if auto_repair_fallback_centerline:
        ensure_fallback_centerline(case_dir)
    paths = resolve_case_paths(case_dir)
    merged_cl = np.load(paths["merged_centerline"], allow_pickle=True).item()
    clipped_cl = np.load(paths["clipped_centerline"], allow_pickle=True).item()
    branch_ranking = np.load(paths["branch_ranking"], allow_pickle=True)
    endpoints_ = np.load(paths["endpoints"], allow_pickle=True).item()
    aneurysm_centroid = np.asarray(endpoints_["aneurysm_centroid"], dtype=np.float64)
    aneurysm_type = int(endpoints_["aneurysm_type"])

    branch_ids = compute_branch_ids_order(clipped_cl)
    branch_segments, branch_endpoints = [], []
    for render_idx in branch_ranking:
        branch_id = branch_ids[int(render_idx)]
        seg = extract_real_branch_segment(merged_cl, clipped_cl, branch_id, aneurysm_centroid)
        branch_segments.append(seg.astype(np.float32))
        branch_endpoints.append(seg[-1].astype(np.float32))

    # Raw open clip -- real anatomy only, no synthetic extrusion -- used ONLY
    # for the alignment loss's surface chamfer term (see module docstring for
    # why: comparing against the synthetic extension would be wrong).
    mesh = trimesh.load(paths["clipped_mesh"], process=False)
    surface_points, _ = trimesh.sample.sample_surface(mesh, n_surface_samples)

    closed_mesh = _build_closed_case_mesh(paths, extrude_length=extrude_length)
    volume = abs(float(closed_mesh.volume))

    dome_points = None
    if dome_points_loader is not None:
        dome_points = dome_points_loader(case_dir, np.asarray(mesh.vertices))
        if dome_points is None:
            print(f"  Note: no dome-region data available for this case ({case_dir}) -- "
                  f"the dome alignment term will be skipped for it.")

    return {
        "case_dir": str(case_dir),
        "paths": paths,
        "use_fallback": paths["use_fallback"],
        "aneurysm_type": aneurysm_type,
        "aneurysm_centroid": aneurysm_centroid.astype(np.float32),
        "dome_points": dome_points,
        "branch_segments": branch_segments,
        "branch_endpoints": np.stack(branch_endpoints, axis=0),
        "surface_points": np.asarray(surface_points, dtype=np.float32),
        "volume": volume,
        "closed_mesh": closed_mesh,
    }


# ── canonical-side loading ──────────────────────────────────────────────────

def load_canonical_data(canonical_dir, n_surface_samples=4000):
    canonical_dir = Path(canonical_dir)
    topology = np.load(canonical_dir / "canonical_topology.npy", allow_pickle=True).item()
    mesh = trimesh.load(canonical_dir / "mesh.obj", process=False)
    surface_points, _ = trimesh.sample.sample_surface(mesh, n_surface_samples)
    if not mesh.is_watertight:
        print(f"  Warning: canonical mesh {canonical_dir / 'mesh.obj'} is not watertight -- "
              f"volume estimate may be inaccurate.")

    branch_segments = [o["branch_points"].astype(np.float32) for o in topology["openings"]]
    branch_endpoints = np.stack([seg[-1] for seg in branch_segments], axis=0)

    # dome_vertex_indices comes from dataset/canonical/record_topology.py's
    # own picking (canonical-side counterpart to load_case_data's
    # dome_points_loader) -- always present for this repo's canonical
    # templates, no placeholder needed here (unlike the per-case target
    # side, the canonical template itself doesn't change per dataset).
    dome_idx = topology.get("dome_vertex_indices")
    dome_points = (np.asarray(mesh.vertices)[np.asarray(dome_idx)].astype(np.float32)
                  if dome_idx is not None and len(dome_idx) > 0 else None)

    return {
        "aneurysm_type": topology["aneurysm_type"],
        "neck_centroid": topology["neck_centroid"].astype(np.float32),
        "dome_points": dome_points,
        "branch_segments": branch_segments,
        "branch_endpoints": branch_endpoints,
        "surface_points": np.asarray(surface_points, dtype=np.float32),
        "volume": abs(float(mesh.volume)),
    }


# ── closed-form initialization (Umeyama similarity transform) ──────────────

def umeyama_similarity(src, dst):
    """Closed-form least-squares similarity transform (rotation + uniform
    scale + translation) from N >= 3 corresponding point pairs, s.t.
    dst_i ~= s * (R @ src_i) + t. Standard Umeyama (1991) algorithm -- the
    deterministic part of this fit; chamfer terms have no closed-form
    equivalent (nearest-neighbour correspondence depends on the transform
    itself), so this is used purely as an initialization for the gradient-
    based refinement that follows, not the final answer.

    Returns R [3,3], s (float), t [3], all numpy float64.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    mu_src, mu_dst = src.mean(0), dst.mean(0)
    src_c, dst_c = src - mu_src, dst - mu_dst

    cov = (dst_c.T @ src_c) / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[-1, -1] = -1.0
    R = U @ S @ Vt

    var_src = (src_c ** 2).sum() / len(src)
    s = float(np.trace(np.diag(D) @ S) / var_src) if var_src > 1e-12 else 1.0
    t = mu_dst - s * (R @ mu_src)
    return R, s, t


# ── learnable similarity transform ──────────────────────────────────────────

class SimilarityTransform(nn.Module):
    """s * (R @ x) + t, with R via axis-angle (pytorch3d) so gradient descent
    stays on a valid rotation manifold, and scale in log-space so it can't
    collapse through/past zero or flip sign during optimization."""

    def __init__(self, R_init=None, s_init=1.0, t_init=None, device="cpu"):
        super().__init__()
        if R_init is not None:
            axis_angle = matrix_to_axis_angle(
                torch.as_tensor(R_init, dtype=torch.float32, device=device).unsqueeze(0)
            ).squeeze(0)
        else:
            axis_angle = torch.zeros(3, device=device)
        self.axis_angle = nn.Parameter(axis_angle.clone())
        self.log_s = nn.Parameter(torch.tensor(float(np.log(max(s_init, 1e-6))), device=device))
        t_init = torch.zeros(3) if t_init is None else torch.as_tensor(t_init, dtype=torch.float32)
        self.t = nn.Parameter(t_init.clone().to(device))

    def rotation_matrix(self):
        return axis_angle_to_matrix(self.axis_angle.unsqueeze(0)).squeeze(0)

    def scale(self):
        return torch.exp(self.log_s)

    def forward(self, x):
        """x: [..., 3] -> transformed [..., 3]."""
        R = self.rotation_matrix()
        return self.scale() * (x @ R.T) + self.t


# ── loss ─────────────────────────────────────────────────────────────────────

def _resample_branch_uniform(pts, n_points=64):
    """Resample an ORDERED curve (a branch centerline, dome-first: index 0
    near the dome/neck, last index the far endpoint -- see load_case_data/
    load_canonical_data's own docstrings, both use this same convention) to
    exactly n_points, uniformly spaced by arc-length from start to end.

    Exploits the fact that branch order/direction is already known and
    consistent on both sides -- unlike chamfer (order-agnostic nearest-
    neighbor matching, which can match multiple points to the same
    counterpart or skip around along the curve), this gives a direct,
    unambiguous point-to-point correspondence at matched fractional arc-
    length, meant to be compared via a plain MSE rather than chamfer.
    Pure numpy, computed once on the untransformed points -- arc-length-
    uniform resampling commutes with any later rigid+scale transform, so
    there's no need to redo this every optimization step."""
    pts64 = np.asarray(pts, dtype=np.float64)
    if len(pts64) == 1:
        out = np.tile(pts64[0], (n_points, 1))
    else:
        seg_lens = np.linalg.norm(np.diff(pts64, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(seg_lens)])
        total = arc[-1]
        if total < 1e-9:
            out = np.tile(pts64[0], (n_points, 1))
        else:
            targets = np.linspace(0.0, total, n_points)
            out = np.stack([np.interp(targets, arc, pts64[:, d]) for d in range(3)], axis=1)
    return out.astype(np.float32)  # match branch_segments' own float32 convention


def _weighted_branch_mean(values, n_branches, focus_id=None, focus_weight=1.0):
    """Weighted mean over one scalar per branch (a list of 0-d tensors, or
    already a 1-D tensor of length n_branches) -- focus_id (if given) counts
    focus_weight times as much as every other branch, normalized so the
    total weight still sums to n_branches (keeps the overall loss magnitude
    comparable to a plain mean). focus_id=None or focus_weight=1.0 reduces
    to an exact plain mean -- the pre-existing, still-default behavior."""
    stacked = torch.stack(values) if isinstance(values, list) else values
    if focus_id is None or focus_weight == 1.0:
        return stacked.mean()
    weights = torch.ones(n_branches, device=stacked.device)
    weights[focus_id] = focus_weight
    weights = weights / weights.sum() * n_branches
    return (stacked * weights).mean()


def alignment_loss(transform, case, canonical, w_centerline=1.0, w_surface=1.0,
                   w_endpoint=1.0, w_volume=1.0, w_dome=1.0,
                   branch_focus_id=None, branch_focus_centerline_weight=1.0,
                   branch_focus_endpoint_weight=1.0,
                   centerline_matching="chamfer",
                   case_branch_resampled=None, canon_branch_resampled=None):
    """mean per-branch centerline term + surface chamfer + mean per-branch
    endpoint L2 + volume match + dome-region chamfer, all between the
    TRANSFORMED case and the fixed canonical.

    centerline_matching: "chamfer" (default, unchanged behavior) does order-
    agnostic nearest-neighbor chamfer distance per branch. "point2point"
    instead uses case_branch_resampled/canon_branch_resampled -- lists of
    (n_resample, 3) arrays, ALREADY resampled to uniform arc-length via
    _resample_branch_uniform (see fit_alignment, which precomputes these
    once) -- and computes a plain per-point MSE at matched indices, since
    branch order/direction (dome-first) is already known and consistent on
    both sides. Meant to give a sharper, unambiguous gradient signal than
    chamfer for a specific mismatched branch (see branch_focus_id below),
    which chamfer's many-to-one nearest-neighbor matching can dilute.

    branch_focus_id: if a case's alignment is dominated by one specific
    branch being mismatched (confirmed by eyeballing sanity images branch-
    by-branch), upweight that branch_ranking index's contribution to the
    centerline term (by branch_focus_centerline_weight) and the endpoint
    term (by branch_focus_endpoint_weight) INDEPENDENTLY -- every other
    branch stays at weight 1 on both terms -- rather than the plain
    unweighted per-branch mean. Leaves every other loss term (surface/
    volume/dome) untouched -- this is deliberately narrow, not a general
    reweighting knob.

    Dome term: chamfer between case["dome_points"] (this dataset's convention:
    label.nrrd voxel label==2, see load_dome_points_from_nrrd) and
    canonical["dome_points"] (canonical_topology.npy's dome_vertex_indices).
    Chosen over a plain centroid-to-centroid distance because it also
    constrains dome SIZE/SHAPE, not just position, at roughly the same cost
    as the existing surface chamfer term (same chamfer_distance call, on a
    subset of points). Skipped (contributes 0, not an error) whenever either
    side's dome_points is None -- keeps this robust to cases/datasets
    without dome-region data, see load_case_data's dome_points_loader.

    Volume is handled without re-meshing every step: rotation and translation
    don't change enclosed volume, only the uniform scale does, and it does so
    as the CUBE of the scale factor (V(s*X) = s^3 * V(X)) -- so
    case["volume"] (computed once, from the closed mesh in the case's OWN
    frame, at data-loading time) times transform.scale()**3 gives the
    transformed volume directly, as a cheap differentiable function of the
    scale parameter alone. Expressed as a RELATIVE (scale-invariant) error so
    its magnitude is comparable to the other terms for the equal weighting
    to mean something -- an absolute mm^3 difference would be a very
    different scale (hundreds) from the mm-scale distance terms."""
    device = transform.t.device
    n_branches = len(case["branch_segments"])

    centerline_terms = []
    if centerline_matching == "point2point":
        for k in range(n_branches):
            case_pts = transform(torch.as_tensor(case_branch_resampled[k], device=device))
            canon_pts = torch.as_tensor(canon_branch_resampled[k], device=device)
            centerline_terms.append(((case_pts - canon_pts) ** 2).sum(-1).mean())
    else:
        for k in range(n_branches):
            case_pts = transform(torch.as_tensor(case["branch_segments"][k], device=device)).unsqueeze(0)
            canon_pts = torch.as_tensor(canonical["branch_segments"][k], device=device).unsqueeze(0)
            cd, _ = chamfer_distance(case_pts, canon_pts)
            centerline_terms.append(cd)
    loss_centerline = _weighted_branch_mean(centerline_terms, n_branches, branch_focus_id, branch_focus_centerline_weight)

    case_surf = transform(torch.as_tensor(case["surface_points"], device=device)).unsqueeze(0)
    canon_surf = torch.as_tensor(canonical["surface_points"], device=device).unsqueeze(0)
    loss_surface, _ = chamfer_distance(case_surf, canon_surf)

    case_ep = transform(torch.as_tensor(case["branch_endpoints"], device=device))
    canon_ep = torch.as_tensor(canonical["branch_endpoints"], device=device)
    ep_dists = (case_ep - canon_ep).norm(dim=-1)
    loss_endpoint = _weighted_branch_mean(ep_dists, n_branches, branch_focus_id, branch_focus_endpoint_weight)

    transformed_volume = (transform.scale() ** 3) * case["volume"]
    loss_volume = ((transformed_volume - canonical["volume"]) / canonical["volume"]) ** 2

    loss_dome = torch.zeros((), device=device)
    if case.get("dome_points") is not None and canonical.get("dome_points") is not None:
        case_dome = transform(torch.as_tensor(case["dome_points"], device=device)).unsqueeze(0)
        canon_dome = torch.as_tensor(canonical["dome_points"], device=device).unsqueeze(0)
        loss_dome, _ = chamfer_distance(case_dome, canon_dome)

    total = (w_centerline * loss_centerline + w_surface * loss_surface
            + w_endpoint * loss_endpoint + w_volume * loss_volume + w_dome * loss_dome)
    return total, {
        "loss_centerline": loss_centerline.item(),
        "loss_surface": loss_surface.item(),
        "loss_endpoint": loss_endpoint.item(),
        "loss_volume": loss_volume.item(),
        "loss_dome": loss_dome.item(),
        "loss_total": total.item(),
    }


# ── fitting driver ───────────────────────────────────────────────────────────

def fit_alignment(case, canonical, epochs=800, lr=1e-2, w_centerline=1.0, w_surface=1.0,
                  w_endpoint=1.0, w_volume=1.0, w_dome=1.0, device="cpu", log_every=100,
                  branch_focus_id=None, branch_focus_centerline_weight=1.0,
                  branch_focus_endpoint_weight=1.0,
                  centerline_matching="chamfer", centerline_n_resample=64):
    if case["aneurysm_type"] != canonical["aneurysm_type"] and not (
        case["aneurysm_type"] in (1, 2) and canonical["aneurysm_type"] in (1, 2)
    ):
        raise ValueError(f"Case aneurysm_type={case['aneurysm_type']} doesn't match "
                         f"canonical aneurysm_type={canonical['aneurysm_type']}.")
    if len(case["branch_segments"]) != len(canonical["branch_segments"]):
        raise ValueError(f"Case has {len(case['branch_segments'])} branch(es), canonical has "
                         f"{len(canonical['branch_segments'])} -- can't establish correspondence.")

    # closed-form init from corresponding points: each branch's endpoint, plus the
    # case's own aneurysm centroid <-> canonical's neck centroid (both are "center of
    # the aneurysm/neck region", the same physical reference point on each side).
    src_pts = np.vstack([case["branch_endpoints"], case["aneurysm_centroid"][None]])
    dst_pts = np.vstack([canonical["branch_endpoints"], canonical["neck_centroid"][None]])
    R0, s0, t0 = umeyama_similarity(src_pts, dst_pts)

    # Precomputed ONCE (not per-iteration): arc-length-uniform resampling is
    # independent of the transform being optimized -- see
    # _resample_branch_uniform's own docstring.
    case_branch_resampled = canon_branch_resampled = None
    if centerline_matching == "point2point":
        case_branch_resampled = [_resample_branch_uniform(seg, centerline_n_resample)
                                 for seg in case["branch_segments"]]
        canon_branch_resampled = [_resample_branch_uniform(seg, centerline_n_resample)
                                  for seg in canonical["branch_segments"]]

    transform = SimilarityTransform(R_init=R0, s_init=s0, t_init=t0, device=device)
    init_loss, init_log = alignment_loss(transform, case, canonical, w_centerline, w_surface,
                                         w_endpoint, w_volume, w_dome,
                                         branch_focus_id, branch_focus_centerline_weight,
                                         branch_focus_endpoint_weight,
                                         centerline_matching, case_branch_resampled, canon_branch_resampled)
    print(f"  [init, closed-form Umeyama] {init_log}", flush=True)

    optimizer = torch.optim.Adam(transform.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss, log = alignment_loss(transform, case, canonical, w_centerline, w_surface,
                                   w_endpoint, w_volume, w_dome,
                                   branch_focus_id, branch_focus_centerline_weight,
                                   branch_focus_endpoint_weight,
                                   centerline_matching, case_branch_resampled, canon_branch_resampled)
        loss.backward()
        optimizer.step()
        history.append(log)
        if epoch % log_every == 0 or epoch == epochs - 1:
            print(f"  [{epoch:5d}/{epochs}] {log}", flush=True)

    result = {
        "R": transform.rotation_matrix().detach().cpu().numpy(),
        "s": transform.scale().detach().cpu().item(),
        "t": transform.t.detach().cpu().numpy(),
        "R_init": R0, "s_init": s0, "t_init": t0,
        "history": history,
        "final_loss": history[-1],
        "case_dir": case["case_dir"],
        "use_fallback": case["use_fallback"],
    }
    return transform, result


# ── sanity render ────────────────────────────────────────────────────────────

def render_sanity_images(save_dir, transform, case, canonical, n_angles=6,
                         filename="alignment_sanity.png"):
    """One combined image, multiple angles as a grid of subplots (not
    separate per-angle files) -- meant for a quick eyeball during testing."""
    import pyvista as pv

    if pv.system_supports_plotting() is False or not os.environ.get("DISPLAY"):
        # Headless machine, no real X server: PyVista/VTK still needs SOME
        # display target even for off_screen=True rendering (older PyVista/VTK
        # builds error out instead of falling back automatically -- hit this
        # exact crash testing on this workstation's 'new' env, pyvista 0.42,
        # whereas 'vmtk_autogen's pyvista 0.47 apparently handles it fine).
        # start_xvfb() launches a virtual framebuffer and points $DISPLAY at
        # it, which is enough for off-screen rendering with no real display.
        pv.start_xvfb()

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = transform.t.device

    with torch.no_grad():
        case_surf_t = transform(torch.as_tensor(case["surface_points"], device=device)).cpu().numpy()
        case_branches_t = [
            transform(torch.as_tensor(seg, device=device)).cpu().numpy()
            for seg in case["branch_segments"]
        ]
        case_dome_t = (transform(torch.as_tensor(case["dome_points"], device=device)).cpu().numpy()
                      if case.get("dome_points") is not None else None)

    canon_surf = canonical["surface_points"]
    canon_dome = canonical.get("dome_points")
    colors = ["red", "blue", "green"]

    all_pts = np.vstack([case_surf_t, canon_surf])
    focal = all_pts.mean(axis=0)
    diag = np.linalg.norm(all_pts.max(0) - all_pts.min(0))
    cam_dist = diag * 1.6

    n_cols = min(3, n_angles)
    n_rows = int(np.ceil(n_angles / n_cols))
    plotter = pv.Plotter(off_screen=True, shape=(n_rows, n_cols),
                         window_size=(480 * n_cols, 480 * n_rows), border=True)

    for i in range(n_angles):
        row, col = divmod(i, n_cols)
        plotter.subplot(row, col)

        angle_deg = round(360 * i / n_angles)
        angle_rad = np.deg2rad(angle_deg)
        cam_pos = focal + cam_dist * np.array([np.cos(angle_rad), np.sin(angle_rad), 0.3])

        plotter.add_points(canon_surf, color="lightgrey", point_size=3, opacity=0.4, label="canonical")
        plotter.add_points(case_surf_t, color="black", point_size=3, opacity=0.5, label="case (aligned)")
        if canon_dome is not None:
            plotter.add_points(canon_dome, color="orange", point_size=5, opacity=0.7,
                               label="canonical dome" if i == 0 else None)
        if case_dome_t is not None:
            plotter.add_points(case_dome_t, color="magenta", point_size=5, opacity=0.7,
                               label="case dome (aligned)" if i == 0 else None)
        for k, seg in enumerate(case_branches_t):
            c = colors[k % len(colors)]
            plotter.add_mesh(pv.Spline(seg), color=c, line_width=6,
                             label=f"case branch {k}" if i == 0 else None)
        for k, seg in enumerate(canonical["branch_segments"]):
            c = colors[k % len(colors)]
            plotter.add_mesh(pv.Spline(seg), color=c, line_width=2, opacity=0.5,
                             label=f"canonical branch {k}" if i == 0 else None)

        plotter.add_text(f"{angle_deg} deg", font_size=10, position="upper_edge")
        if i == 0:
            plotter.add_legend(size=(0.35, 0.3), bcolor="white")
        plotter.set_background("white")
        plotter.camera.position = cam_pos
        plotter.camera.focal_point = focal
        plotter.camera.up = (0.0, 0.0, 1.0)

    out_path = save_dir / filename
    plotter.screenshot(str(out_path))
    plotter.close()
    print(f"  Sanity render saved: {out_path}")
    return out_path


# ── stage-1 checkpoint export ───────────────────────────────────────────────

def save_stage1_checkpoints(save_dir, transform, case, canonical):
    """Export final_aligned.obj / landmarks.npz / centerline.vtp -- Stage 1's
    checkpoint set for Stage 2 (GHD-coefficient fitting) and any other
    downstream tool, matching AneuSeg/prep_case_for_ghd.py's file set in
    spirit (same three files, same roles), not byte-identical keys -- that
    script derived a rigid-only transform from anatomical landmarks; this
    one is a similarity transform fit by chamfer+volume gradient descent, and
    has no per-case opening-ring indices (nothing downstream consumes those
    here -- see alignment.py's module docstring / the plan this implements).

    final_aligned.obj is case["closed_mesh"] (the extruded/planarized/fan-
    capped, watertight mesh already built once in load_case_data -- NOT the
    raw open clip), rigidly transformed into canonical space -- this matches
    the old prep_case_for_ghd.py's own final_aligned.obj, which was likewise
    the capped/watertight mesh (via VMTK there), not the raw clip. Stage 2's
    occupancy/volume loss terms need a closed target, so this avoids Stage 2
    having to (unreliably) re-close an open mesh itself.
    """
    import pyvista as pv

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = transform.t.device

    with torch.no_grad():
        mesh = case["closed_mesh"]
        verts_t = transform(
            torch.as_tensor(mesh.vertices, dtype=torch.float32, device=device)
        ).cpu().numpy()
    aligned_mesh = trimesh.Trimesh(vertices=verts_t, faces=mesh.faces, process=False)
    aligned_path = save_dir / "final_aligned.obj"
    aligned_mesh.export(aligned_path)

    with torch.no_grad():
        aneurysm_centroid_t = transform(
            torch.as_tensor(case["aneurysm_centroid"], device=device)
        ).cpu().numpy()
        branch_endpoints_t = transform(
            torch.as_tensor(case["branch_endpoints"], device=device)
        ).cpu().numpy()

    # No `scale` key here -- matches the old prep_case_for_ghd.py format
    # exactly (rigid-only: R_frame + t_vec, no scale saved), even though this
    # transform *is* a similarity transform internally. Its scale is already
    # baked into final_aligned.obj's vertex positions; Stage 2 (ghd_fit.py)
    # re-derives its own residual-scale init directly from geometry (target
    # vs. canonical extents in the shared normalized space) rather than
    # depending on a persisted value here -- see ghd_fit.py's module notes.
    #
    # `volume` here is aligned_mesh.volume (the TRANSFORMED, post-scale
    # mesh's own volume) -- NOT case["volume"] (the untransformed volume in
    # the case's own original frame). Volume scales as scale**3, so using
    # case["volume"] directly would be off by transform.scale()**3 relative
    # to what final_aligned.obj's geometry actually is -- a real bug found
    # by inspection (Stage 2's VolumeLoss was fitting toward a target ~2.6x
    # too large for a scale~0.73 case, visibly inflating the fitted mesh).
    # Opening ring vertex indices (into final_aligned.obj's vertex array,
    # unchanged by the rigid transform), rank-matched to case["branch_endpoints"]'
    # order so they align with canonical's own rank-ordered opening indices
    # (canonical_topology.npy). EXPERIMENTAL -- for the opening-chamfer /
    # ring-roundness / ring-normal-alignment / geodesic-correspondence loss
    # terms only; not used by the default (opening-index-free) loss set.
    #
    # Not every boundary loop _find_boundary_loops finds is a real
    # anatomical opening: small mesh defects (a tear/hole from imperfect
    # reconstruction) get closed by _fan_cap_open_mesh exactly like real
    # openings (needed for watertightness), but must NOT be treated as
    # opening/ring data. Real openings are identified by proximity to a
    # real branch's centerline endpoint (case["branch_endpoints"], already
    # known from Stage 1's centerline loading) -- a defect hole sits
    # somewhere else on the surface and won't be near any branch endpoint.
    # A branch with no loop within the threshold is left WITHOUT saved
    # ring data (this case's opening-chamfer/roundness/normal-alignment/
    # geodesic loss setup will just skip it or, if too incomplete, disable
    # itself with a warning -- see ghd_fit.py) rather than guessing wrong;
    # per explicit instruction, some cases being unusable for these
    # EXPERIMENTAL terms is fine and expected, not something to force
    # through -- report and move on, manual mesh repair is a separate step.
    # ALL-OR-NOTHING: opening_ring_idx_k must be a DENSE 0..n-1 range for
    # ghd_fit.py's rank-correspondence with canonical_topology.npy's openings
    # to be meaningful, so a partial match (some branches matched, others
    # not) is not saved as partial data -- that would either misalign ranks
    # or need sparse-index handling everywhere downstream. Simpler and more
    # robust: either every real branch gets a matched loop (full opening
    # data saved) or none do (n_openings=0, this case is just unusable for
    # the EXPERIMENTAL opening-chamfer/roundness/normal-alignment/geodesic
    # terms -- ghd_fit.py already disables them gracefully when n_openings
    # is 0). Per explicit instruction this is fine; report and move on.
    OPENING_MATCH_THRESHOLD_MM = 10.0
    opening_loops = getattr(mesh, "opening_loops", [])
    cap_face_loop_id = getattr(mesh, "cap_face_loop_id", np.full(len(mesh.faces), -1, dtype=np.int64))
    landmarks_extra = {}
    real_loop_ids = set()
    all_matched = False
    if opening_loops:
        loop_centroids = np.stack([case["closed_mesh"].vertices[loop].mean(axis=0) for loop in opening_loops])
        matches = {}
        failed = []
        for k, ep in enumerate(case["branch_endpoints"]):
            dists = np.linalg.norm(loop_centroids - ep[None], axis=1)
            best = int(np.argmin(dists))
            if dists[best] > OPENING_MATCH_THRESHOLD_MM:
                failed.append((k, dists[best]))
            else:
                matches[k] = best
        duplicated = len(set(matches.values())) != len(matches)
        all_matched = not failed and not duplicated and len(matches) == len(case["branch_endpoints"])
        if all_matched:
            for k, best in matches.items():
                landmarks_extra[f"opening_ring_idx_{k}"] = np.asarray(opening_loops[best], dtype=np.int64)
                real_loop_ids.add(best)
            n_defect_loops = len(opening_loops) - len(real_loop_ids)
            if n_defect_loops > 0:
                print(f"  Note: {n_defect_loops} boundary loop(s) closed for watertightness but not "
                      f"matched to any branch (small mesh defect, not a real opening) -- excluded "
                      f"from opening-ring/cap data.")
        else:
            if failed:
                print(f"  WARNING: no boundary loop found within {OPENING_MATCH_THRESHOLD_MM}mm of "
                      f"branch endpoint(s) {[k for k, _ in failed]} "
                      f"(nearest: {[f'{d:.1f}mm' for _, d in failed]}) -- likely a mesh defect/"
                      f"closing failure needing manual repair.")
            if duplicated:
                print(f"  WARNING: two or more branches matched the SAME boundary loop -- too few "
                      f"loops found for the branch count.")
            print(f"  Opening-ring data NOT saved for this case (all-or-nothing) -- it will be "
                  f"unusable for the EXPERIMENTAL opening-chamfer/roundness/normal-alignment/"
                  f"geodesic loss terms, but Stage 1/2 otherwise proceed normally.")
    # cap_face_mask covers ONLY real (branch-matched) openings' cap faces --
    # defect-hole caps stay in the mesh (still watertight) but are excluded
    # here so they never pollute the pooled opening-chamfer point sampling.
    landmarks_extra["cap_face_mask"] = np.isin(cap_face_loop_id, list(real_loop_ids)) if real_loop_ids \
        else np.zeros(len(mesh.faces), dtype=bool)
    # n_openings MUST equal the number of opening_ring_idx_k keys actually
    # written above -- NOT len(opening_loops) or len(branch_endpoints),
    # either of which can exceed the real saved-key count and cause a
    # downstream KeyError in ghd_fit.py::load_target.
    landmarks_extra["n_openings"] = np.array(len(case["branch_endpoints"]) if all_matched else 0)

    landmarks_path = save_dir / "landmarks.npz"
    np.savez(
        landmarks_path,
        R_frame=transform.rotation_matrix().detach().cpu().numpy(),
        t_vec=transform.t.detach().cpu().numpy(),
        aneurysm_type=np.array(case["aneurysm_type"]),
        aneurysm_centroid=aneurysm_centroid_t.astype(np.float32),
        volume=np.array(abs(float(aligned_mesh.volume)), dtype=np.float64),
        branch_endpoints=branch_endpoints_t.astype(np.float32),
        **landmarks_extra,
    )

    with torch.no_grad():
        branches_t = [
            transform(torch.as_tensor(seg, device=device)).cpu().numpy()
            for seg in case["branch_segments"]
        ]
    points = np.vstack(branches_t).astype(np.float32)
    lines, offset = [], 0
    for seg in branches_t:
        n = len(seg)
        lines.append(n)
        lines.extend(range(offset, offset + n))
        offset += n
    cl_poly = pv.PolyData(points, lines=np.array(lines, dtype=np.int64))
    centerline_path = save_dir / "centerline.vtp"
    cl_poly.save(centerline_path)

    print(f"  Stage 1 checkpoints saved -> {aligned_path.name}, "
          f"{landmarks_path.name}, {centerline_path.name}")
    return {"final_aligned": aligned_path, "landmarks": landmarks_path,
            "centerline": centerline_path}


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--sidewall-canonical-dir", default=str(ROOT / "dataset" / "canonical" / "Sidewall"))
    parser.add_argument("--bifurcated-canonical-dir", default=str(ROOT / "dataset" / "canonical" / "Bifurcated"))
    parser.add_argument("--save-dir", default=None,
                        help="Default: runtime/alignment_test/<case-name>/ -- kept OUT of the shared "
                             "case geometry folder since this is a testing script (runtime/ is the "
                             "existing gitignored convention for local, regenerable outputs).")
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--w-centerline", type=float, default=1.0)
    parser.add_argument("--w-surface", type=float, default=1.0)
    parser.add_argument("--w-endpoint", type=float, default=1.0)
    parser.add_argument("--w-volume", type=float, default=1.0)
    parser.add_argument("--w-dome", type=float, default=1.0,
                        help="Weight for the dome-region chamfer term. Auto-skipped (contributes "
                             "0) for any case/canonical missing dome-region data.")
    parser.add_argument("--dome-source", choices=list(DOME_LOADERS), default="nrrd",
                        help="Which per-case dome-region loader to use: nrrd (label.nrrd "
                             "voxel label==2, ImperialNHS-style), mesh (dome_sac.ply, "
                             "AneuX_stable-style), or none (disable the dome term).")
    parser.add_argument("--branch-focus-id", type=int, default=None,
                        help="branch_ranking index (0-based) to upweight in the centerline+"
                             "endpoint terms, e.g. when one specific branch is confirmed "
                             "mismatched by eyeball. None (default) = plain unweighted mean, "
                             "unchanged behavior.")
    parser.add_argument("--branch-focus-centerline-weight", type=float, default=1.0,
                        help="Relative weight for --branch-focus-id's CENTERLINE term vs. "
                             "every other branch (each still at weight 1). 1.0 = no effect.")
    parser.add_argument("--branch-focus-endpoint-weight", type=float, default=1.0,
                        help="Relative weight for --branch-focus-id's ENDPOINT term vs. "
                             "every other branch (each still at weight 1). 1.0 = no effect.")
    parser.add_argument("--centerline-matching", choices=["chamfer", "point2point"], default="chamfer",
                        help="chamfer (default): order-agnostic nearest-neighbor chamfer per "
                             "branch. point2point: resample both sides to --centerline-n-resample "
                             "points at uniform arc-length (dome-first order already known) and "
                             "use per-point MSE instead -- sharper, unambiguous correspondence.")
    parser.add_argument("--centerline-n-resample", type=int, default=64,
                        help="Points per branch when --centerline-matching point2point.")
    parser.add_argument("--n-surface-samples", type=int, default=4000)
    parser.add_argument("--extrude-length", type=float, default=0.25,
                        help="mm, uniform across every case -- how far each case's open "
                             "boundaries are extruded (real tangent direction) before "
                             "capping, purely to get a watertight mesh for the volume term.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Default: cuda if available (chamfer distance is the expensive "
                             "part of every step -- GPU is a large speedup here).")
    args = parser.parse_args()

    case_dir = Path(args.case_dir)
    save_dir = Path(args.save_dir) if args.save_dir else ROOT / "runtime" / "alignment_test" / case_dir.name

    case = load_case_data(case_dir, n_surface_samples=args.n_surface_samples,
                          extrude_length=args.extrude_length,
                          dome_points_loader=DOME_LOADERS[args.dome_source])
    canonical_dir = (args.bifurcated_canonical_dir if case["aneurysm_type"] == 0
                     else args.sidewall_canonical_dir)
    print(f"Case: {case_dir.name}  aneurysm_type={case['aneurysm_type']}  "
          f"use_fallback={case['use_fallback']}  canonical={canonical_dir}  device={args.device}")
    canonical = load_canonical_data(canonical_dir, n_surface_samples=args.n_surface_samples)
    print(f"  case volume={case['volume']:.2f} mm^3  canonical volume={canonical['volume']:.2f} mm^3")

    transform, result = fit_alignment(
        case, canonical, epochs=args.epochs, lr=args.lr,
        w_centerline=args.w_centerline, w_surface=args.w_surface, w_endpoint=args.w_endpoint,
        w_volume=args.w_volume, w_dome=args.w_dome, device=args.device,
        branch_focus_id=args.branch_focus_id,
        branch_focus_centerline_weight=args.branch_focus_centerline_weight,
        branch_focus_endpoint_weight=args.branch_focus_endpoint_weight,
        centerline_matching=args.centerline_matching,
        centerline_n_resample=args.centerline_n_resample,
    )

    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / "alignment_result.npy", result, allow_pickle=True)
    print(f"Saved alignment result -> {save_dir / 'alignment_result.npy'}")

    save_stage1_checkpoints(save_dir, transform, case, canonical)
    render_sanity_images(save_dir, transform, case, canonical)


if __name__ == "__main__":
    main()
