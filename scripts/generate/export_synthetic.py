"""
Export spec-compliant synthetic-aneurysm training bundles from the
Fourier branch MLP-VAE generator.

This is the data-exporting counterpart to `eval_branch_mlp_vae.py` (which is a
visual evaluator only). It reuses that script's model-loading and generation
helpers — GHD VAE → phi, then BranchFourierVAE → per-branch (chord vector,
Fourier coeffs, presence) → centerlines — then writes, per sample, the file
bundle defined in `SYNTHETIC_DATA_SPEC.md` so the existing 8-view preprocessing
pipeline can run on synthetic data unchanged.

Per-sample folder ``{out}/synth_{NNNNN}_aneurysm1/`` contains:

    ghd_coefficients.npz     phi[144,3] f32, w_rot[3] f32, log_scale () f64, t_vec[3] f32
    ghd_fitted.obj           dome GHD mesh, NORMALIZED (canonical) frame  (full faces)
    ghd_fitted_world.obj     same mesh in SYNTH-WORLD mm  (== "cropped" dome-only mesh)
    merge_mesh_world.obj     full fused dome+vessel mesh in SYNTH-WORLD mm
    dome_face_ids.npy        (K,) int64 face indices (into the .obj face list) of the dome
    landmarks.npz            neck_pt_world, R_frame, dome_centroid_world, aneu_type
    metrics.json             {"s_can": ...}
    merged_centerline.npy    {"branches":[{group,pts,radii,arc_length}], "connected_branch_ids":[...]}

Coordinate model (see SYNTHETIC_DATA_SPEC.md "two frames"):

    verts_world = (verts_canonical * s_can) @ R_frame + neck_pt_world

For synthetics we use ``R_frame = I``, ``neck_pt_world = 0``, and
``s_can = recon.norm_canonical``. The transformer pipeline already produces
geometry in *physical GHD space* (canonical x norm_canonical), so:

    verts_world      = multi_recon._reconstruct_verts_np(phi, atype)   # physical == world
    verts_canonical  = verts_world / s_can
    branch curves    = already in physical/world space (no extra scaling)

GHD fit-corrections are left at identity (w_rot=0, log_scale=0, t_vec=0): the
new fused-mesh path does not apply the VAE scalar, so baking it into only the
dome mesh would desync it from the merged mesh. This keeps every mesh in one
frame and satisfies the spec round-trip ``world == canonical * s_can`` exactly.

Usage:
    conda activate new
    python scripts/generate/export_synthetic.py --out v2/synthetic_v1 --n 50 --seed 1
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh

ROOT = Path(__file__).resolve().parents[2]
sys.path = [path for path in sys.path if Path(path or ".").resolve() != ROOT]
sys.path.insert(0, str(ROOT))

from models.mesh_plugins import seperate_mesh
from models.multi_canonical_ghd_reconstruct import MultiCanonicalGHDReconstruct
# Vessel generator: the Fourier branch MLP-VAE evaluation helper.
# model.sample(phi, cond, z) -> (vec, coeff, presence); branches are
# reconstructed per opening with branch_centerlines() and pruned by presence.
from scripts.evaluate.eval_branch_mlp_vae import (
    CHECKPOINT, GHD_VAE_CKPT, DEVICE, NUM_TYPES, MAX_BRANCHES, GHD_INPUT_DIM,
    PRESENCE_THRESH, TYPE_N_OPEN,
    EXTRUDE_LENGTH, MIN_BRANCH_ARC, FUSE_SMOOTH,
    load_branch_model, load_ghd_vae, generate_ghd, build_conditions,
    branch_centerlines, polydata_tris,
)

CANONICAL_ROOT = ROOT / "dataset" / "canonical"

# Spec landmark string for `aneu_type` (the preprocessors do
# `str(lm["aneu_type"]) in ("bifurcation", "sidewall")`). NB: the old reference
# data stored an int code here, which fails that very check; we store the string.
ANEU_TYPE_STR = {0: "bifurcation", 1: "sidewall", 2: "sidewall"}


# ── dome extraction ─────────────────────────────────────────────────────────

def _load_neck_indices(spec_root):
    """Canonical-mesh vertex indices forming the neck ring for one type.

    Accepts either ``neck_size_checkpoint.pkl`` (key ``neck_v_indices`` — a list
    whose first element is the ring, as used by the bifurcated canonical) or a
    plain ``neck_v_indices.npy`` (1-D int array). To add a new type, pick the
    neck-ring vertices in ParaView on that type's ``mesh.obj`` (its vertex
    indexing equals the canonical indexing) and save them as ``neck_v_indices.npy``.

    Returns:
        np.ndarray (M,) int64 — neck vertex indices.
    """
    root = Path(spec_root)
    pkl = root / "neck_size_checkpoint.pkl"
    npy = root / "neck_v_indices.npy"
    if pkl.exists():
        chk = torch.load(pkl, map_location="cpu", weights_only=False)
        return np.asarray(chk["neck_v_indices"][0], dtype=np.int64).ravel()
    if npy.exists():
        return np.load(npy).astype(np.int64).ravel()
    raise FileNotFoundError(
        f"No neck definition for {root.name}: expected {pkl.name} or {npy.name}. "
        f"Pick the neck-ring vertices in ParaView on {root / 'mesh.obj'} and save "
        f"their indices to {npy}."
    )


def _build_dome_face_id_lookup(full_faces, dome_faces):
    """Map each dome face (vertex triple) to its row index in the full face list.

    Faces are matched order-independently via their sorted vertex tuple, so the
    returned ids index `full_faces` exactly (the face list shared by
    ghd_fitted.obj / ghd_fitted_world.obj). Ported from the old
    create_merged_meshes.py.
    """
    key_full = {tuple(np.sort(f)): i for i, f in enumerate(full_faces)}
    return np.array([key_full[tuple(np.sort(f))] for f in dome_faces], dtype=np.int64)


def compute_dome_face_ids(multi_recon, atype, _cache={}):
    """Dome face indices into the FULL canonical face list, for one type (cached).

    Strategy (identical for every type, matching the old pipeline):
      1. Build the trimmed-topology mesh in canonical vertex indices.
      2. Split it with the neck ring via `seperate_mesh`.
      3. Pick the dome side as the region containing the FEWER opening vertices
         (robust to seperate_mesh's seed-dependent region ordering).
      4. Remap the chosen faces onto the full canonical face list.

    Returns:
        np.ndarray (K,) int64 — dome face indices into `ghd_fitted.obj`'s faces.
    """
    atype = int(atype)
    if atype in _cache:
        return _cache[atype]

    recon       = multi_recon.get(atype)
    full_faces  = recon.canonical_Meshes.faces_packed().detach().cpu().numpy().astype(np.int64)
    canon_verts = recon.canonical_Meshes.verts_packed().detach().cpu().numpy()  # topology only
    trimmed     = multi_recon._trimmed_faces(atype)                              # [F',3] canon vidx
    neck_idx    = set(int(i) for i in _load_neck_indices(multi_recon.specs[atype].root))

    tmesh = trimesh.Trimesh(vertices=canon_verts, faces=trimmed, process=False)
    part_a, part_b = seperate_mesh(tmesh, np.asarray(sorted(neck_idx), dtype=np.int64))

    opening_idx = np.unique(np.concatenate(
        [idx.detach().cpu().numpy() for idx in multi_recon._load_openings(atype)]))
    opening_set = set(int(i) for i in opening_idx)

    def _n_opening_verts(faces):
        return len(set(int(v) for v in np.asarray(faces).reshape(-1)) & opening_set)

    # Dome = the part touching the openings the least.
    dome_faces = part_a if _n_opening_verts(part_a) <= _n_opening_verts(part_b) else part_b
    dome_face_ids = _build_dome_face_id_lookup(full_faces, dome_faces)

    _validate_dome(full_faces, dome_face_ids, atype)
    _cache[atype] = dome_face_ids
    return dome_face_ids


def _validate_dome(full_faces, dome_face_ids, atype):
    """Sanity-check the dome region: in range, non-trivial, single connected blob."""
    F = len(full_faces)
    assert dome_face_ids.ndim == 1 and dome_face_ids.size > 0, "empty dome"
    assert dome_face_ids.min() >= 0 and dome_face_ids.max() < F, "dome face id out of range"
    frac = dome_face_ids.size / F
    if not (0.02 < frac < 0.95):
        print(f"[warn] type {atype}: dome is {frac:.1%} of faces ({dome_face_ids.size}/{F}); "
              "check the neck ring.")
    # connectivity: the dome submesh must be one component
    sub = trimesh.Trimesh(
        vertices=np.zeros((full_faces.max() + 1, 3)),
        faces=full_faces[dome_face_ids], process=False,
    )
    n_comp = len(trimesh.graph.connected_components(sub.face_adjacency,
                                                    nodes=np.arange(len(sub.faces))))
    if n_comp != 1:
        print(f"[warn] type {atype}: dome has {n_comp} connected components (expected 1).")


# ── geometry helpers ────────────────────────────────────────────────────────

def _save_obj(path, verts, faces):
    """Minimal OBJ writer preserving exact vertex order and 0-based face list."""
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in verts]
    lines += [f"f {a+1} {b+1} {c+1}" for a, b, c in faces]
    Path(path).write_text("\n".join(lines) + "\n")


def reconstruct_world_dome(multi_recon, phi_i, atype):
    """Dome (ghd_fitted) geometry for one sample.

    Returns:
        verts_world (V,3) f64 — physical/synth-world mm
        verts_canon (V,3) f64 — normalized canonical frame (== world / s_can)
        full_faces  (F,3) i64 — shared face list
        s_can       float      — canonical->world scale (recon.norm_canonical)
    """
    recon = multi_recon.get(atype)
    s_can = float(recon.norm_canonical)
    verts_world = multi_recon._reconstruct_verts_np(phi_i, atype).astype(np.float64)
    verts_canon = verts_world / s_can
    full_faces  = recon.canonical_Meshes.faces_packed().detach().cpu().numpy().astype(np.int64)
    return verts_world, verts_canon, full_faces, s_can


def reconstruct_world_merged(multi_recon, phi_i, atype, curves, branch_mask, save_path=None):
    """Full fused dome+vessel mesh for one sample, in synth-world mm.

    `curves` and the fused mesh are already in physical/world space, so no extra
    scaling is applied. Returns (verts (V,3) f64, faces (F,3) i64).
    """
    merged = multi_recon.reconstruct_fused_mesh(
        phi_i, atype,
        branch_points=curves,
        branch_mask=np.asarray(branch_mask).astype(bool).ravel(),
        extrude_length=EXTRUDE_LENGTH, min_branch_arc=MIN_BRANCH_ARC,
        smooth=FUSE_SMOOTH, save_path=save_path,
    )
    verts, faces = polydata_tris(merged)
    return verts.astype(np.float64), faces.astype(np.int64)


# ── mesh self-intersection (defect detector) ─────────────────────────────────

def _seg_tri_hits(p0, p1, V0, V1, V2):
    """Möller–Trumbore: does segment p0->p1 cross triangle (V0,V1,V2)? bool[M]."""
    d = p1 - p0; e1 = V1 - V0; e2 = V2 - V0
    h = np.cross(d, e2); a = np.einsum("ij,ij->i", e1, h)
    eps = 1e-9; ok = np.abs(a) > eps
    f = np.zeros(len(a)); f[ok] = 1.0 / a[ok]
    s = p0 - V0; u = f * np.einsum("ij,ij->i", s, h)
    q = np.cross(s, e1); v = f * np.einsum("ij,ij->i", d, q)
    t = f * np.einsum("ij,ij->i", e2, q)
    return ok & (u >= -eps) & (u <= 1 + eps) & (v >= -eps) & (u + v <= 1 + eps) \
              & (t >= eps) & (t <= 1 - eps)


def count_self_intersections(verts, faces):
    """Number of self-intersecting (non-vertex-sharing) triangle pairs in a mesh.

    Detects true geometric overlap — tube-vs-tube crossings on tight bends AND
    tube-vs-dome poke-through at a misaligned graft — independent of curvature.
    A valid hard bend with radius-of-curvature ≥ tube radius scores 0.

    cKDTree broadphase over face centroids (radius = sum of circumradii) → exact
    edge/triangle tests on surviving candidate pairs.
    """
    from scipy.spatial import cKDTree
    verts = np.asarray(verts, float); faces = np.asarray(faces)
    tris = verts[faces]
    cen = tris.mean(1)
    rad = np.linalg.norm(tris - cen[:, None], axis=2).max(1)
    pairs = cKDTree(cen).query_pairs(r=2 * rad.max(), output_type="ndarray")
    if len(pairs) == 0:
        return 0
    close = (np.linalg.norm(cen[pairs[:, 0]] - cen[pairs[:, 1]], axis=1)
             <= rad[pairs[:, 0]] + rad[pairs[:, 1]])
    pairs = pairs[close]
    if len(pairs) == 0:
        return 0
    # drop pairs that share any vertex (topological neighbours), vectorized
    shares = (faces[pairs[:, 0]][:, :, None] == faces[pairs[:, 1]][:, None, :]).any((1, 2))
    pairs = pairs[~shares]
    if len(pairs) == 0:
        return 0
    A, B = tris[pairs[:, 0]], tris[pairs[:, 1]]
    hit = np.zeros(len(pairs), bool)
    for a, b in [(0, 1), (1, 2), (2, 0)]:
        hit |= _seg_tri_hits(A[:, a], A[:, b], B[:, 0], B[:, 1], B[:, 2])
        hit |= _seg_tri_hits(B[:, a], B[:, b], A[:, 0], A[:, 1], A[:, 2])
    return int(hit.sum())


def self_intersection_frac(mesh_path):
    """selfX fraction (intersecting pairs / faces) for a saved .obj, or None."""
    p = Path(mesh_path)
    if not p.exists():
        return None
    m = trimesh.load(p, process=False)
    F = len(m.faces)
    return count_self_intersections(m.vertices, m.faces) / max(F, 1)


# ── bundle writer ───────────────────────────────────────────────────────────

def save_training_sample(sample_dir, phi_i, atype, curves, branch_mask,
                         multi_recon, save_merged=True, estimate_radii=False):
    """Assemble and write the full spec bundle for one synthetic sample.

    Args:
        sample_dir:   output folder (created if absent).
        phi_i:        (144,3) GHD coefficients (tensor or array) for this sample.
        atype:        int aneurysm type.
        curves:       list of [Ni,3] absolute branch centerlines (physical/world space),
                      length MAX_BRANCHES; entry o pairs with opening o.
        branch_mask:  bool [MAX_BRANCHES] — which openings have a valid branch.
        multi_recon:  MultiCanonicalGHDReconstruct.
        save_merged:  also build + write merge_mesh_world.obj (the fused full mesh).
        estimate_radii: store a per-branch constant radius (opening-ring radius)
                      instead of None.

    Returns the sample_dir path.
    """
    sample_dir = Path(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)
    atype = int(atype)

    phi_np = (phi_i.detach().cpu().numpy() if torch.is_tensor(phi_i) else np.asarray(phi_i))
    phi_np = phi_np.reshape(144, 3).astype(np.float32)
    branch_mask = np.asarray(branch_mask).astype(bool).ravel()

    # geometry
    verts_world, verts_canon, full_faces, s_can = reconstruct_world_dome(multi_recon, phi_np, atype)
    dome_face_ids = compute_dome_face_ids(multi_recon, atype)

    # 1. ghd_coefficients.npz  (identity fit-corrections — see module docstring)
    np.savez(
        sample_dir / "ghd_coefficients.npz",
        phi=phi_np,
        w_rot=np.zeros(3, dtype=np.float32),
        log_scale=np.float64(0.0),
        t_vec=np.zeros(3, dtype=np.float32),
    )

    # 2 & 3. ghd_fitted.obj (canonical) and ghd_fitted_world.obj (world, "cropped" dome)
    _save_obj(sample_dir / "ghd_fitted.obj",       verts_canon, full_faces)
    _save_obj(sample_dir / "ghd_fitted_world.obj", verts_world, full_faces)

    # 4. dome_face_ids.npy
    np.save(sample_dir / "dome_face_ids.npy", dome_face_ids.astype(np.int64))

    # 5. landmarks.npz  (zero-pose recipe; dome_centroid in world)
    dome_v_idx = np.unique(full_faces[dome_face_ids].reshape(-1))
    dome_centroid_world = verts_world[dome_v_idx].mean(axis=0).astype(np.float64)
    np.savez(
        sample_dir / "landmarks.npz",
        neck_pt_world=np.zeros(3, dtype=np.float64),
        R_frame=np.eye(3, dtype=np.float64),
        dome_centroid_world=dome_centroid_world,
        aneu_type=np.array(ANEU_TYPE_STR[atype]),
    )

    # 6. metrics.json
    (sample_dir / "metrics.json").write_text(json.dumps({"s_can": s_can}))

    # 7. merged_centerline.npy  (synth-world mm, same frame as ghd_fitted_world.obj)
    recon = multi_recon.get(atype)
    opening_idx = [idx.detach().cpu().numpy() for idx in multi_recon._load_openings(atype)]
    branches = []
    for o in range(min(len(curves), MAX_BRANCHES)):
        if not branch_mask[o]:
            continue
        pts = np.asarray(curves[o], dtype=np.float32)
        if pts.shape[0] < 2:
            continue
        radii = None
        if estimate_radii:
            ring = verts_world[opening_idx[o]]
            r = float(np.linalg.norm(ring - ring.mean(0), axis=1).mean())
            radii = np.full(pts.shape[0], r, dtype=np.float64)
        branches.append({
            "group": 0,
            "pts": pts,
            "radii": radii,
            "arc_length": float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()),
        })
    np.save(
        sample_dir / "merged_centerline.npy",
        {"branches": branches, "connected_branch_ids": list(range(len(branches)))},
        allow_pickle=True,
    )

    # 8. merge_mesh_world.obj  (full fused dome+vessel mesh)
    if save_merged:
        try:
            reconstruct_world_merged(multi_recon, phi_np, atype, curves, branch_mask,
                                     save_path=sample_dir / "merge_mesh_world.obj")
        except Exception as exc:
            print(f"[warn] {sample_dir.name}: fusion failed ({exc}); merge_mesh_world.obj skipped")

    return sample_dir


# ── per-folder sanity visuals (Step 5) ───────────────────────────────────────

def _equal_box(ax, *vert_sets):
    pts = np.concatenate([np.asarray(v).reshape(-1, 3) for v in vert_sets if v is not None and len(v)])
    mn, mx = pts.min(0), pts.max(0)
    c, h = (mn + mx) / 2, (mx - mn).max() / 2
    ax.set_xlim(c[0] - h, c[0] + h)
    ax.set_ylim(c[1] - h, c[1] + h)
    ax.set_zlim(c[2] - h, c[2] + h)
    ax.set_box_aspect((1, 1, 1))


def _remap_dome_to_merged(dome_tri_centroids, merged_verts, merged_faces):
    """Nearest-centroid map of dome faces onto the merged mesh's face list."""
    from scipy.spatial import cKDTree
    merged_centroids = merged_verts[merged_faces].mean(axis=1)
    _, idx = cKDTree(merged_centroids).query(dome_tri_centroids)
    return np.unique(idx)


def visualize_sample(sample_dir, out_png=None, views=((20, 30), (20, 210))):
    """Render a per-folder sanity figure from the SAVED bundle files.

    Left block  : ghd_fitted_world.obj with dome faces (red) vs vessel (gray) +
                  dome_centroid_world (star). Confirms dome_face_ids.
    Right block : merge_mesh_world.obj (gray) with each merged_centerline branch
                  drawn as a colored polyline + start marker. Confirms the
                  centerline sits in the same world frame as the mesh, and that
                  the dome region (red, remapped by centroid) matches.

    Reads only saved files, so it validates exactly what shipped.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    sample_dir = Path(sample_dir)
    dome = trimesh.load(sample_dir / "ghd_fitted_world.obj", process=False)
    V, Fc = np.asarray(dome.vertices), np.asarray(dome.faces)
    fids = np.load(sample_dir / "dome_face_ids.npy")
    lm   = np.load(sample_dir / "landmarks.npz", allow_pickle=True)
    cl   = np.load(sample_dir / "merged_centerline.npy", allow_pickle=True).item()
    centroid = np.asarray(lm["dome_centroid_world"])

    mask = np.zeros(len(Fc), bool); mask[fids] = True
    tri_dome, tri_other = V[Fc[fids]], V[Fc[~mask]]
    branches = cl["branches"]
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(branches), 1)))

    mm_path = sample_dir / "merge_mesh_world.obj"
    merged = trimesh.load(mm_path, process=False) if mm_path.exists() else None

    nv = len(views)
    fig = plt.figure(figsize=(5 * nv * 2, 5))

    # left: dome overlay
    for j, (el, az) in enumerate(views):
        ax = fig.add_subplot(1, nv * 2, j + 1, projection="3d")
        ax.add_collection3d(Poly3DCollection(tri_other, facecolor="lightgray",
                            edgecolor=(0, 0, 0, 0.10), linewidth=0.1, alpha=0.5))
        ax.add_collection3d(Poly3DCollection(tri_dome, facecolor="crimson",
                            edgecolor=(0, 0, 0, 0.15), linewidth=0.1, alpha=0.9))
        ax.scatter(*centroid, c="gold", marker="*", s=160, edgecolor="k", depthshade=False)
        _equal_box(ax, V)
        ax.view_init(elev=el, azim=az); ax.set_axis_off()
        ax.set_title(f"dome {len(fids)}/{len(Fc)} faces" if j == 0 else "", fontsize=9)

    # right: merged mesh + centerlines
    dome_centroids = tri_dome.mean(axis=1)
    for j, (el, az) in enumerate(views):
        ax = fig.add_subplot(1, nv * 2, nv + j + 1, projection="3d")
        cl_pts = [b["pts"] for b in branches]
        if merged is not None:
            Vm, Fm = np.asarray(merged.vertices), np.asarray(merged.faces)
            dome_m = _remap_dome_to_merged(dome_centroids, Vm, Fm)
            mm = np.zeros(len(Fm), bool); mm[dome_m] = True
            ax.add_collection3d(Poly3DCollection(Vm[Fm[~mm]], facecolor="lightgray",
                                edgecolor="none", alpha=0.18))
            ax.add_collection3d(Poly3DCollection(Vm[Fm[mm]], facecolor="crimson",
                                edgecolor="none", alpha=0.18))
            box_sets = [Vm] + cl_pts
        else:
            box_sets = [V] + cl_pts
        for i, b in enumerate(branches):
            p = np.asarray(b["pts"]); c = colors[i % len(colors)]
            ax.plot(p[:, 0], p[:, 1], p[:, 2], color=c, linewidth=2)
            ax.scatter(*p[0], color=c, marker="x", s=50)
        _equal_box(ax, *box_sets)
        ax.view_init(elev=el, azim=az); ax.set_axis_off()
        ax.set_title(f"merged + {len(branches)} centerlines" if j == 0 else "", fontsize=9)

    fig.suptitle(f"{sample_dir.name}  —  aneu_type={str(lm['aneu_type'])}", fontsize=11)
    plt.tight_layout()
    if out_png is None:
        out_png = sample_dir / "sanity_check.png"
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_png


# ── validation (Step 5) ──────────────────────────────────────────────────────

def validate_sample(sample_dir, atol=1e-3):
    """Run the spec's per-sample assertions + extra checks. Returns list of issues."""
    sample_dir = Path(sample_dir)
    issues = []

    def check(cond, msg):
        if not cond:
            issues.append(msg)

    m_canon = trimesh.load(sample_dir / "ghd_fitted.obj",       process=False)
    m_world = trimesh.load(sample_dir / "ghd_fitted_world.obj", process=False)
    fids    = np.load(sample_dir / "dome_face_ids.npy")
    cl      = np.load(sample_dir / "merged_centerline.npy", allow_pickle=True).item()
    lm      = np.load(sample_dir / "landmarks.npz", allow_pickle=True)
    ghd     = np.load(sample_dir / "ghd_coefficients.npz")
    metrics = json.loads((sample_dir / "metrics.json").read_text())

    # topology / shared faces
    check(np.array_equal(m_world.faces, m_canon.faces), "face indexing differs canon vs world")
    check(fids.max() < len(m_world.faces), "dome face id out of range")
    check(m_world.is_watertight, "world mesh not watertight")

    # ghd schema
    check(ghd["phi"].shape == (144, 3) and ghd["phi"].dtype == np.float32, "phi shape/dtype")
    check(ghd["w_rot"].shape == (3,) and ghd["log_scale"].shape == () and ghd["t_vec"].shape == (3,),
          "ghd correction shapes")

    # landmarks
    check(lm["R_frame"].shape == (3, 3) and lm["neck_pt_world"].shape == (3,), "landmark shapes")
    check(str(lm["aneu_type"]) in ("bifurcation", "sidewall"), "aneu_type not a spec string")

    # canonical -> world round-trip
    s_can = float(metrics["s_can"])
    expected = (m_canon.vertices * s_can) @ lm["R_frame"] + lm["neck_pt_world"]
    check(np.allclose(m_world.vertices, expected, atol=atol), "canonical->world transform mismatch")

    # dome centroid consistency
    dome_v = np.unique(m_world.faces[fids].reshape(-1))
    check(np.allclose(m_world.vertices[dome_v].mean(0), lm["dome_centroid_world"], atol=atol),
          "dome_centroid_world != centroid of dome submesh")

    # centerlines
    check(all(b["pts"].shape[1] == 3 for b in cl["branches"]), "centerline pts not (N,3)")
    check(all(b["pts"].dtype == np.float32 for b in cl["branches"]), "centerline pts not float32")
    check(len(cl["connected_branch_ids"]) > 0, "no connected branches")

    # merged mesh present + geometrically valid
    mm = sample_dir / "merge_mesh_world.obj"
    if mm.exists():
        merged = trimesh.load(mm, process=False)
        check(len(merged.faces) > len(m_world.faces),
              "merged mesh not larger than dome mesh (vessels missing?)")
        check(merged.vertices.shape[0] > 0, "merged mesh empty")
    else:
        issues.append("merge_mesh_world.obj missing")

    return issues


# ── phi sampling ─────────────────────────────────────────────────────────────

@torch.no_grad()
def _generate_ghd_amp(ghd_vae, ghd_mean, ghd_std, types, amp):
    """generate_ghd with a latent temperature `amp` on z_ghd (amp=1 == prior)."""
    B = types.size(0)
    z_ghd = torch.randn(B, ghd_vae.latent_dim, device=DEVICE) * amp
    ghd_n, scale_n = ghd_vae.decode(z_ghd, types)
    phi   = (ghd_n * ghd_std[:, :GHD_INPUT_DIM] + ghd_mean[:, :GHD_INPUT_DIM]).reshape(B, -1, 3)
    scale = (scale_n * ghd_std[:, GHD_INPUT_DIM:] + ghd_mean[:, GHD_INPUT_DIM:]).squeeze(1)
    return phi, scale


# canonical group per aneurysm type (real phi is fit per shared canonical)
_CANON_GROUP = {0: "bifurcated", 1: "sidewall", 2: "sidewall"}


class RealPhiGMM:
    """Sample phi from a Gaussian fit to the REAL phi distribution, per canonical
    group, in PCA-reduced space.

    The VAE prior under-covers the real phi spread (~0.70 of real along the real
    PCs) and raising its temperature only adds off-manifold shapes. Fitting a
    Gaussian directly to the real phi and sampling from it reproduces the real
    covariance by construction (spread ≈ 1.0), staying realistic. Broken domes
    that fall in the Gaussian tail are caught downstream by the dome filter.
    """

    def __init__(self, real_root="dataset/processed", k=40):
        self.real_root = real_root
        self.k = k
        self._fit = {}   # group -> (mu, comp[k,432], chol[k,k])

    def _fit_group(self, group):
        if group not in self._fit:
            phis = []
            for f in glob.glob(os.path.join(self.real_root, "*.npy")):
                d = np.load(f, allow_pickle=True).item()
                if str(d.get("canonical_type", "")).lower() == group:
                    phis.append(np.asarray(d["ghd"]["phi"], np.float64).ravel())
            if len(phis) < 8:
                raise ValueError(f"too few real '{group}' phi ({len(phis)}) to fit a GMM")
            X = np.stack(phis)
            k = min(self.k, len(X) // 3)
            mu = X.mean(0)
            _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
            comp = Vt[:k]
            proj = (X - mu) @ comp.T
            chol = np.linalg.cholesky(np.cov(proj.T) + 1e-9 * np.eye(k))
            self._fit[group] = (mu, comp, chol)
            print(f"[gmm] fit real '{group}': n={len(X)} k={k}")
        return self._fit[group]

    def sample(self, atype, n, rng):
        """Return (n, 144, 3) float32 phi for one aneurysm type."""
        mu, comp, chol = self._fit_group(_CANON_GROUP[int(atype)])
        z = rng.standard_normal((n, chol.shape[0])) @ chol.T
        return (z @ comp + mu).reshape(n, 144, 3).astype(np.float32)


@torch.no_grad()
def export(out_dir, n=50, batch=12, seed=1, save_merged=True, validate=True,
           estimate_radii=False, viz=True, sample_z=True,
           min_branches=1, reject_invalid=True, max_attempts_factor=5,
           compute_selfx=True, max_selfx_frac=None, max_dome_selfx=0,
           force_type=None, start_index=0, ghd_amp=1.0, branch_amp=1.0,
           phi_source="vae", gmm_k=40, real_root="dataset/processed"):
    """Generate `n` ACCEPTED samples and write spec bundles under `out_dir`.

    sample_z:    True  → branch latent z ~ N(0,I) per sample (diverse vessels);
                 False → z = 0 (deterministic mean/most-likely branches).
    force_type:  None → random type per sample; int → all samples this aneurysm
                 type (0=bifurcation, 1/2=sidewall). Use to build per-type batches.
    start_index: folder numbering offset, so per-type runs can share one out_dir
                 without name collisions (synth_{start_index+i:05d}_aneurysm1).

    Rejection-resampling (a rejected candidate is not written and not counted;
    a new one is drawn until `n` are accepted or the attempt cap is hit):
      min_branches:   reject (cheaply, before any mesh work) candidates with
                      fewer than this many present branches — a sample with no
                      vessel reaching the dome is unusable (`connected_branch_ids`
                      would be empty, failing the spec).
      max_dome_selfx: reject (before fusion) GHD domes whose own surface self-
                      intersects beyond this many triangle pairs. Clean domes
                      score exactly 0, so the default 0 drops only broken shapes
                      with no false positives. None disables the check.
      max_selfx_frac: if set, reject samples whose self-intersecting-face fraction
                      exceeds this (calibrate against a labelled subset first).
      reject_invalid: also delete + resample any written sample that fails
                      `validate_sample` (e.g. fusion produced a non-watertight or
                      missing merged mesh). Requires `validate=True`.
      max_attempts_factor: cap total draws at `n * factor` to avoid looping
                      forever if the model rarely produces valid samples.
    """
    import shutil

    torch.manual_seed(seed)
    np.random.seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    multi_recon = MultiCanonicalGHDReconstruct(CANONICAL_ROOT, device=DEVICE)
    model       = load_branch_model(CHECKPOINT, multi_recon)
    ghd_vae, ghd_mean, ghd_std = load_ghd_vae(GHD_VAE_CKPT, DEVICE)
    tstr = "random" if force_type is None else f"{force_type} ({ANEU_TYPE_STR.get(force_type, '?')})"
    print(f"Branch MLP-VAE: {CHECKPOINT}\nGHD VAE: {GHD_VAE_CKPT}\nType: {tstr}\nphi_source: {phi_source}")
    gmm = RealPhiGMM(real_root, k=gmm_k) if phi_source == "gmm" else None
    gmm_rng = np.random.default_rng(seed)

    idx = 0                 # accepted-sample counter (drives folder names)
    attempts = 0            # total candidates drawn
    attempt_cap = max(n * max_attempts_factor, n + 50)
    rejected = {"no_branch": 0, "dome_defect": 0, "self_intersect": 0, "invalid": 0, "error": 0}
    all_issues = {}

    while idx < n and attempts < attempt_cap:
        B = min(batch, attempt_cap - attempts)
        if force_type is None:
            types = torch.randint(0, NUM_TYPES, (B,), device=DEVICE)
        else:
            types = torch.full((B,), int(force_type), dtype=torch.long, device=DEVICE)
        # GHD dome shape.
        if phi_source == "gmm":
            # phi sampled from the real distribution (matches real spread); scale
            # still from the VAE (unit-correct for branch conditioning).
            _, scale = generate_ghd(ghd_vae, ghd_mean, ghd_std, types)
            phi_np = np.concatenate([gmm.sample(int(t), 1, gmm_rng) for t in types.tolist()], 0)
            phi = torch.from_numpy(phi_np).to(DEVICE)
        elif ghd_amp == 1.0:
            phi, scale = generate_ghd(ghd_vae, ghd_mean, ghd_std, types)
        else:
            # ghd_amp scales the VAE latent (temperature). NB: >1 adds off-manifold
            # shapes and lowers real-axis spread — prefer phi_source=gmm for variety.
            phi, scale = _generate_ghd_amp(ghd_vae, ghd_mean, ghd_std, types, ghd_amp)
        cond = build_conditions(multi_recon, phi, types, scale, MAX_BRANCHES)
        # Fourier MLP-VAE: per-branch chord vector, Fourier coeffs, presence prob.
        # branch_amp scales the branch latent likewise.
        if sample_z:
            z = torch.randn(B, model.latent_dim, device=DEVICE) * branch_amp
        else:
            z = torch.zeros(B, model.latent_dim, device=DEVICE)
        vec, coeff, pres = model.sample(phi, cond, z=z)
        starts = cond.start_points.cpu().numpy()
        vec, coeff, pres = vec.cpu().numpy(), coeff.cpu().numpy(), pres.cpu().numpy()

        for i in range(B):
            if idx >= n:
                break
            attempts += 1
            atype  = int(types[i])
            n_open = TYPE_N_OPEN.get(atype, MAX_BRANCHES)
            # present openings → Fourier centerline; absent (pres ≤ thresh) → None (3mm stub)
            curves = branch_centerlines(starts[i], vec[i], coeff[i], pres[i],
                                        n_open, PRESENCE_THRESH)
            branch_mask = [c is not None for c in curves]

            # cheap pre-save reject: a sample with no vessel is unusable
            if sum(branch_mask) < min_branches:
                rejected["no_branch"] += 1
                continue

            # dome-quality reject: the GHD dome itself must not self-intersect
            # (a broken training target). Clean domes score exactly 0, so this is
            # a binary, false-positive-free filter. Checked before fusion to skip
            # the expensive merge on a doomed sample.
            dome_sx = None
            if max_dome_selfx is not None:
                dverts, _, dfaces, _ = reconstruct_world_dome(multi_recon, phi[i], atype)
                dome_sx = count_self_intersections(dverts, dfaces)
                if dome_sx > max_dome_selfx:
                    rejected["dome_defect"] += 1
                    continue

            name = f"synth_{start_index + idx:05d}_aneurysm1"
            sample_dir = out_dir / name
            try:
                save_training_sample(
                    sample_dir, phi[i], atype, curves, branch_mask,
                    multi_recon, save_merged=save_merged, estimate_radii=estimate_radii,
                )
                if viz:
                    try:
                        visualize_sample(sample_dir)
                    except Exception as exc:
                        print(f"[warn] {name}: viz failed ({exc})")
                # ground-truth defect: mesh self-intersection (tube-tube crossing
                # or tube-dome poke-through). Recorded for every sample so the
                # threshold can be calibrated against a manually-labelled subset.
                selfx = None
                if compute_selfx:
                    selfx = self_intersection_frac(sample_dir / "merge_mesh_world.obj")
                if selfx is not None or dome_sx is not None:
                    meta = json.loads((sample_dir / "metrics.json").read_text())
                    if selfx is not None:
                        meta["selfx_frac"] = selfx
                    if dome_sx is not None:
                        meta["dome_selfx"] = int(dome_sx)
                    (sample_dir / "metrics.json").write_text(json.dumps(meta))
                if (max_selfx_frac is not None and selfx is not None
                        and selfx > max_selfx_frac):
                    shutil.rmtree(sample_dir, ignore_errors=True)
                    rejected["self_intersect"] += 1
                    print(f"[reject] {name}: selfx_frac={selfx:.3f} > {max_selfx_frac}")
                    continue

                issues = validate_sample(sample_dir) if validate else []
                if issues and reject_invalid:
                    shutil.rmtree(sample_dir, ignore_errors=True)
                    rejected["invalid"] += 1
                    print(f"[reject] {name}: {issues}")
                    continue
                sx = f"  selfx={selfx:.3f}" if selfx is not None else ""
                if issues:
                    all_issues[name] = issues
                    print(f"[FAIL] {name}: {issues}{sx}")
                else:
                    print(f"[ok]   {name}  (type {atype}){sx}")
            except Exception as exc:
                shutil.rmtree(sample_dir, ignore_errors=True)
                rejected["error"] += 1
                print(f"[reject] {name}: exception: {exc}")
                continue
            idx += 1

    print(f"\nDone: {idx}/{n} accepted in {attempts} draws → {out_dir}")
    print(f"Rejected — no-branch: {rejected['no_branch']}, "
          f"dome-defect: {rejected['dome_defect']}, "
          f"self-intersect: {rejected['self_intersect']}, "
          f"invalid: {rejected['invalid']}, error: {rejected['error']}")
    if idx < n:
        print(f"[warn] hit attempt cap ({attempt_cap}); only {idx}/{n} accepted. "
              "Lower PRESENCE_THRESH, raise max_attempts_factor, or check the model.")
    if all_issues:
        print(f"{len(all_issues)} written sample(s) still flagged (reject_invalid off).")
    return all_issues


def _parse_args():
    p = argparse.ArgumentParser(description="Export synthetic aneurysm training bundles.")
    p.add_argument("--out", type=str, default=str(ROOT / "v2" / "synthetic_v1"))
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--batch", type=int, default=12)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--no-merged", action="store_true", help="skip merge_mesh_world.obj")
    p.add_argument("--no-validate", action="store_true", help="skip per-sample validation")
    p.add_argument("--no-viz", action="store_true", help="skip per-sample sanity_check.png")
    p.add_argument("--estimate-radii", action="store_true", help="store opening-ring radii")
    p.add_argument("--mean-branches", action="store_true",
                   help="z=0 deterministic branches instead of z~N(0,I)")
    p.add_argument("--min-branches", type=int, default=1,
                   help="reject+resample candidates with fewer present branches")
    p.add_argument("--keep-invalid", action="store_true",
                   help="keep (don't resample) samples that fail validation")
    p.add_argument("--max-attempts-factor", type=int, default=5,
                   help="cap total draws at n * this factor")
    p.add_argument("--type", type=int, default=None,
                   help="fix aneurysm type for all samples (0=bifurcation, 1/2=sidewall); "
                        "default random")
    p.add_argument("--start-index", type=int, default=0,
                   help="folder numbering offset (for per-type runs into one out dir)")
    p.add_argument("--ghd-amp", type=float, default=1.0,
                   help="VAE dome latent temperature (>1 adds off-manifold shapes; "
                        "prefer --phi-source gmm for variety)")
    p.add_argument("--branch-amp", type=float, default=1.0,
                   help="branch latent temperature (>1 = more diverse vessels, more defects)")
    p.add_argument("--phi-source", choices=["vae", "gmm"], default="vae",
                   help="vae = N(0,I) prior (under-covers real); gmm = Gaussian fit to "
                        "real phi (matches real spread)")
    p.add_argument("--gmm-k", type=int, default=40, help="PCA dims for the real-phi Gaussian")
    p.add_argument("--real-root", type=str, default="dataset/processed",
                   help="real phi source for --phi-source gmm")
    p.add_argument("--no-selfx", action="store_true",
                   help="skip mesh self-intersection measurement (recorded in metrics.json)")
    p.add_argument("--max-selfx-frac", type=float, default=None,
                   help="reject samples whose self-intersecting-face fraction exceeds this "
                        "(calibrate against a manually-labelled subset first; e.g. 0.005)")
    p.add_argument("--max-dome-selfx", type=int, default=0,
                   help="reject GHD domes with more than this many self-intersecting "
                        "triangle pairs (0 = drop any broken dome; -1 disables)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    export(args.out, n=args.n, batch=args.batch, seed=args.seed,
           save_merged=not args.no_merged, validate=not args.no_validate,
           estimate_radii=args.estimate_radii, viz=not args.no_viz,
           sample_z=not args.mean_branches,
           min_branches=args.min_branches, reject_invalid=not args.keep_invalid,
           max_attempts_factor=args.max_attempts_factor,
           compute_selfx=not args.no_selfx, max_selfx_frac=args.max_selfx_frac,
           max_dome_selfx=(None if args.max_dome_selfx < 0 else args.max_dome_selfx),
           force_type=args.type, start_index=args.start_index,
           ghd_amp=args.ghd_amp, branch_amp=args.branch_amp,
           phi_source=args.phi_source, gmm_k=args.gmm_k, real_root=args.real_root)
