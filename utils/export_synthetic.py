"""
Helpers for scripts/generate/generate_synthetic.py.

Dome extraction, spec-bundle writing, self-intersection detection, per-folder
sanity visualization, and per-sample validation for synthetic aneurysm
training bundles (see SYNTHETIC_DATA_SPEC.md). Also RealPhiGMM, which samples
GHD coefficients from a Gaussian fit to the real phi distribution rather than
the VAE prior (the prior under-covers the real spread; see its docstring).

The orchestration loop (sampling, rejection, batching) stays in the script;
this module holds everything it calls into.
"""

import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
import trimesh

from models.multi_canonical_ghd_reconstruct import MultiCanonicalGHDReconstruct
from models.mesh_plugins import seperate_mesh
from utils.generate_synthetic import polydata_tris

ANEU_TYPE_STR = {0: "bifurcation", 1: "sidewall", 2: "sidewall"}

# canonical group per aneurysm type (real phi is fit per shared canonical)
_CANON_GROUP = {0: "bifurcated", 1: "sidewall", 2: "sidewall"}


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


def reconstruct_world_merged(multi_recon: MultiCanonicalGHDReconstruct, phi_i, atype, curves, branch_mask, save_path=None,
                             extrude_length=3.0, min_branch_arc=3.0, fuse_smooth=True,
                             min_torsion=True, smooth_n_rings=3, smooth_n_iter=10, smooth_lam=0.5,
                             planarize_n_iter=10, planarize_lam=0.5):
    """Full fused dome+vessel mesh for one sample, in synth-world mm.

    `curves` and the fused mesh are already in physical/world space, so no extra
    scaling is applied. Returns (verts (V,3) f64, faces (F,3) i64).

    min_torsion, smooth_n_rings/n_iter/lam, planarize_n_iter/lam are forwarded
    to reconstruct_fused_mesh as-is; see its docstring for what each controls.
    """
    merged = multi_recon.reconstruct_fused_mesh(
        phi_i, atype,
        branch_points=curves,
        branch_mask=np.asarray(branch_mask).astype(bool).ravel(),
        extrude_length=extrude_length, min_branch_arc=min_branch_arc,
        min_torsion=min_torsion,
        smooth=fuse_smooth, smooth_n_rings=smooth_n_rings, smooth_n_iter=smooth_n_iter,
        smooth_lam=smooth_lam, planarize_n_iter=planarize_n_iter, planarize_lam=planarize_lam,
        save_path=save_path,
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
                         multi_recon, save_merged=True, estimate_radii=False,
                         extrude_length=3.0, min_branch_arc=3.0, fuse_smooth=True,
                         min_torsion=True, smooth_n_rings=3, smooth_n_iter=10, smooth_lam=0.5,
                         planarize_n_iter=10, planarize_lam=0.5):
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
        extrude_length, min_branch_arc, fuse_smooth, min_torsion, smooth_n_rings,
        smooth_n_iter, smooth_lam, planarize_n_iter, planarize_lam: passed to
                      reconstruct_fused_mesh for the merged mesh.

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
    for o in range(min(len(curves), len(branch_mask))):
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
                                     save_path=sample_dir / "merge_mesh_world.obj",
                                     extrude_length=extrude_length, min_branch_arc=min_branch_arc,
                                     fuse_smooth=fuse_smooth, min_torsion=min_torsion,
                                     smooth_n_rings=smooth_n_rings, smooth_n_iter=smooth_n_iter,
                                     smooth_lam=smooth_lam, planarize_n_iter=planarize_n_iter,
                                     planarize_lam=planarize_lam)
        except Exception as exc:
            print(f"[warn] {sample_dir.name}: fusion failed ({exc}); merge_mesh_world.obj skipped")

    return sample_dir


# ── per-folder sanity visuals ─────────────────────────────────────────────────

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


# ── validation ─────────────────────────────────────────────────────────────

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
