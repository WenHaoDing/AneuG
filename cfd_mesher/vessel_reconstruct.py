"""
From an opening mesh, reconstruct tubular mesh through propagating open vertices

"""

import numpy as np
import pyvista as pv
import networkx as nx
import os
from .patching import remove_orphan_vertices


def detect_openings(mesh, flaw_opening_min_size=10):
    """
    Detect the open boundary rings of a surface mesh.

    Parameters
    ----------
    mesh : str or pyvista.PolyData
        Path to mesh file or a loaded PolyData object.

    Returns
    -------
    openings : list of np.ndarray, shape (N_i, 3)
        One array per opening, containing the 3-D coordinates of its boundary nodes.
    vertex_ids : list of np.ndarray, shape (N_i,)
        One array per opening, containing the original vertex indices into mesh.points.
    """
    mesh = pv.read(mesh) if isinstance(mesh, str) else mesh

    # Boundary edges: edges shared by exactly one face
    faces = mesh.faces.reshape(-1, 4)[:, 1:]  # (F, 3)
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges_sorted = np.sort(edges, axis=1)
    unique_edges, counts = np.unique(edges_sorted, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]

    # Connected components on the boundary edge graph = one component per opening
    G = nx.Graph()
    G.add_edges_from(boundary_edges.tolist())
    components = list(nx.connected_components(G))

    vertex_ids = [np.array(sorted(comp)) for comp in components if len(comp) >= flaw_opening_min_size]
    openings = [mesh.points[ids] for ids in vertex_ids]

    return openings, vertex_ids



def avg_edge_length(mesh):
    """Return the mean edge length of a triangulated mesh."""
    faces = mesh.faces.reshape(-1, 4)[:, 1:]  # (F, 3)
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges_sorted = np.sort(edges, axis=1)
    unique_edges = np.unique(edges_sorted, axis=0)
    lengths = np.linalg.norm(mesh.points[unique_edges[:, 0]] - mesh.points[unique_edges[:, 1]], axis=1)
    return 0.85 * np.median(lengths)


def resample_branch(pts, step, min_n=5):
    """
    Resample a polyline (N, 3) at uniform arc-length intervals of `step`.

    Returns an (M, 3) array starting at pts[0] and ending at pts[-1].
    """
    pts = np.asarray(pts, dtype=float)
    dists = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cumlen = np.concatenate([[0.0], np.cumsum(dists)])
    total = cumlen[-1]
    sample_s = np.arange(0.0, total, step)
    if sample_s[-1] < total:
        sample_s = np.append(sample_s, total)
    if len(sample_s) < min_n:
        sample_s = np.linspace(0.0, total, min_n)
    resampled = np.column_stack([
        np.interp(sample_s, cumlen, pts[:, dim]) for dim in range(3)
    ])
    return resampled

def get_cpcd_tangent(cpcd):
    """
    Compute per-point tangent vectors for a resampled centerline polyline.

    Uses central differences for interior points and forward/backward
    differences at the endpoints.

    Parameters
    ----------
    cpcd : np.ndarray, shape (N, 3)

    Returns
    -------
    tangents : np.ndarray, shape (N, 3)
        Unit tangent vectors at each point.
    """
    cpcd = np.asarray(cpcd, dtype=float)
    tangents = np.empty_like(cpcd)
    tangents[0]    = cpcd[1] - cpcd[0]
    tangents[-1]   = cpcd[-1] - cpcd[-2]
    tangents[1:-1] = cpcd[2:] - cpcd[:-2]
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    return tangents / norms



def get_tubular_l2w_trans(cpcd_tangent_glo, min_torsion=False):
    """
    Build a local-to-world rotation matrix at each centerline point.

    Mirrors the batch version in mesh_plugins.py but operates on a single
    branch (no batch or branch dimensions).

    The local frame axes are:
      x — along the tangent (travel direction)
      z — reference projected perpendicular to x (Gram-Schmidt)
      y — cross(z, x), completing the right-handed frame

    Local-to-world convention: p_world = p_local @ l2w_trans[i]

    Parameters
    ----------
    cpcd_tangent_glo : np.ndarray, shape (N, 3)
        Tangent vectors along the branch (need not be unit length).
    min_torsion : bool
        If False (default): each frame is built independently from a fixed
        world reference axis via Gram-Schmidt — fast and unchanged behaviour,
        but introduces spurious twist between adjacent rings on curved
        centerlines.
        If True: a rotation-minimizing frame (parallel transport) is used.
        Frame 0 is seeded by the same Gram-Schmidt rule (so the angular
        ordering at the neck is preserved), and each subsequent frame is
        obtained from the previous by the minimum rotation that maps
        t[i-1] -> t[i] (Rodrigues about t[i-1] x t[i]). This eliminates
        framing-induced torsion in the swept ring.

    Returns
    -------
    l2w_trans : np.ndarray, shape (N, 3, 3)
        Rotation matrices; rows are [x, y, z] local axes in world coordinates.
    """
    t = np.asarray(cpcd_tangent_glo, dtype=float)
    t = t / np.linalg.norm(t, axis=-1, keepdims=True)
    N = len(t)

    # Pick world reference axis (avoid near-collinearity with t[0]).
    ref = np.array([0.0, 0.0, 1.0])
    if np.abs(np.dot(t[0], ref)) > 0.99:
        ref = np.array([0.0, 1.0, 0.0])

    if not min_torsion:
        ref_b = np.tile(ref, (N, 1))
        loc_sys_x = t
        dot = np.einsum('ni,ni->n', ref_b, loc_sys_x)
        loc_sys_z = ref_b - dot[:, None] * loc_sys_x
        loc_sys_z /= np.linalg.norm(loc_sys_z, axis=-1, keepdims=True)
        loc_sys_y = np.cross(loc_sys_z, loc_sys_x)
        loc_sys_y /= np.linalg.norm(loc_sys_y, axis=-1, keepdims=True)
        return np.stack([loc_sys_x, loc_sys_y, loc_sys_z], axis=-2)

    # Rotation-minimizing frame via parallel transport.
    # Seed frame 0 with the same Gram-Schmidt rule so the angular sort at the
    # neck (computed in get_tubular_mesh_verts from l2w[0]) is identical.
    z0 = ref - np.dot(ref, t[0]) * t[0]
    z0 /= np.linalg.norm(z0)
    y0 = np.cross(z0, t[0])
    y0 /= np.linalg.norm(y0)

    x_axes = np.empty_like(t)
    y_axes = np.empty_like(t)
    z_axes = np.empty_like(t)
    x_axes[0], y_axes[0], z_axes[0] = t[0], y0, z0

    for i in range(1, N):
        t_prev = x_axes[i-1]
        t_curr = t[i]
        axis = np.cross(t_prev, t_curr)
        sin_a = np.linalg.norm(axis)
        cos_a = float(np.clip(np.dot(t_prev, t_curr), -1.0, 1.0))
        if sin_a < 1e-12:
            # Tangents collinear (or anti-parallel — unlikely on resampled
            # centerlines); carry the previous transverse axes through.
            x_axes[i] = t_curr
            y_axes[i] = y_axes[i-1]
            z_axes[i] = z_axes[i-1]
            continue
        axis /= sin_a
        # Rodrigues rotation by angle a about `axis`.
        def _rot(v):
            return v * cos_a + np.cross(axis, v) * sin_a + axis * (np.dot(axis, v) * (1.0 - cos_a))
        x_axes[i] = t_curr  # by construction _rot(t_prev) == t_curr
        y_new = _rot(y_axes[i-1])
        # Re-orthonormalize against t_curr to suppress numerical drift, then
        # rebuild z from x x y to maintain the y = z x x convention.
        y_new -= np.dot(y_new, t_curr) * t_curr
        y_new /= np.linalg.norm(y_new)
        y_axes[i] = y_new
        z_axes[i] = np.cross(x_axes[i], y_axes[i])

    return np.stack([x_axes, y_axes, z_axes], axis=-2)


def get_tubular_mesh_verts(opening_pcd, cpcd_glo, cpcd_tangent_glo, l2w_trans,
                           radius_map=True, c_transition=True):
    """
    Generate tubular mesh vertices from opening rings and centerline data.

    Numpy port of get_tubular_mesh_verts in mesh_plugins.py (no batch dim).

    For each branch:
      1. Project the opening ring onto the plane at cpcd_glo[0] (the neck cross-section).
      2. Compute per-node radii (optionally corrected for tangent projection).
      3. Sort nodes by angle in the local frame.
      4. Sweep the ring along the centerline with optional radius transition toward
         the RMS average radius (c_transition).
      5. Transform tube vertices back to world coordinates.

    Parameters
    ----------
    opening_pcd : list of np.ndarray, shape (N_i, 3)
        Boundary ring vertices for each branch opening.
    cpcd_glo : list of np.ndarray, shape (dpi, 3)
        Resampled centerline points for each branch.
    cpcd_tangent_glo : list of np.ndarray, shape (dpi, 3)
        Unit tangent vectors at each centerline point, per branch.
    l2w_trans : list of np.ndarray, shape (dpi, 3, 3)
        Local-to-world rotation matrices, per branch.
    radius_map : bool
        If True, project radii onto the cross-section plane (removes tangent component).
    c_transition : bool
        If True, transition ring radii toward RMS average along the centerline.

    Returns
    -------
    tube_verts_glo_list : list of np.ndarray, shape (dpi, N_i, 3)
        Tube vertex positions in world coordinates, per branch.
    sort_indices_list : list of np.ndarray, shape (N_i,)
        Indices that sort opening nodes by angle in the local frame, per branch.
    """
    tube_verts_glo_list = []
    sort_indices_list = []

    for opening, cpcd, tangents, l2w in zip(opening_pcd, cpcd_glo, cpcd_tangent_glo, l2w_trans):
        opening = np.asarray(opening, dtype=float)   # (N, 3)
        N   = len(opening)
        dpi = len(cpcd)

        centroid = cpcd[0]        # (3,) — neck centre
        tangent  = tangents[0]    # (3,) — neck tangent (unit)

        # ── 1. Project opening ring onto the neck plane ───────────────────────
        D = -np.dot(centroid, tangent)
        t = -(opening @ tangent + D) / np.dot(tangent, tangent)  # (N,)
        pop_glo = opening + t[:, None] * tangent                  # (N, 3)

        # ── 2. Compute radii ──────────────────────────────────────────────────
        rays = opening - centroid                                  # (N, 3)
        dist = np.linalg.norm(rays, axis=-1)                      # (N,)
        if radius_map:
            radius_set = np.sqrt(np.maximum(dist**2 - (rays @ tangent)**2, 0.0))
        else:
            radius_set = dist

        # ── 3. Sort by angle in local frame ──────────────────────────────────
        w2l_0 = l2w[0].T                                          # (3, 3) world→local at neck
        pop_loc = (pop_glo - centroid) @ w2l_0                    # (N, 3)
        loc_angle = np.arctan2(pop_loc[:, 2], pop_loc[:, 1])      # (N,)
        sort_idx = np.argsort(loc_angle)
        loc_angle_sorted = loc_angle[sort_idx]                     # (N,)
        radius_sorted    = radius_set[sort_idx]                    # (N,)
        sort_indices_list.append(sort_idx)

        # ── 4. Build local tube vertices (dpi, N, 3) ─────────────────────────
        tube_loc_x = np.zeros((dpi, N))

        if not c_transition:
            tube_loc_y = np.outer(np.ones(dpi), np.cos(loc_angle_sorted) * radius_sorted)
            tube_loc_z = np.outer(np.ones(dpi), np.sin(loc_angle_sorted) * radius_sorted)
        else:
            radius_avg   = np.linalg.norm(radius_sorted) / np.sqrt(N)    # RMS scalar
            transit      = np.linspace(0, 1, dpi)                         # (dpi,)
            radius_transit = (radius_sorted[:, None]
                              + (radius_avg - radius_sorted[:, None]) * transit[None, :])  # (N, dpi)
            tube_loc_y = (np.cos(loc_angle_sorted)[:, None] * radius_transit).T  # (dpi, N)
            tube_loc_z = (np.sin(loc_angle_sorted)[:, None] * radius_transit).T

        tube_verts_loc = np.stack([tube_loc_x, tube_loc_y, tube_loc_z], axis=-1)  # (dpi, N, 3)

        # ── 5. Transform to world coordinates ────────────────────────────────
        tube_verts_glo = (np.einsum('dnc,dcl->dnl', tube_verts_loc, l2w)
                          + cpcd[:, None, :])                              # (dpi, N, 3)
        tube_verts_glo_list.append(tube_verts_glo)

    return tube_verts_glo_list, sort_indices_list


def get_tubular_mesh_faces(tube_verts_glo_list, ds_r=2):
    """
    Build triangle faces for each tubular mesh.

    Numpy port of get_tubular_mesh_faces in mesh_plugins.py (no batch dim).

    The tube vertices are laid out as a flat array of rings:
      ring 0: indices [0 .. N-1]
      ring 1: indices [N .. 2N-1]  ...

    Each adjacent pair of rings forms a strip of quads, each quad split into
    two triangles (standard cylinder connectivity):

      ring d  :  n ── n+1          tri1: (d*N+n,   d*N+(n+1)%N, (d+1)*N+n)
                 |  / |            tri2: (d*N+(n+1)%N, (d+1)*N+(n+1)%N, (d+1)*N+n)
      ring d+1:  n ── n+1

    Parameters
    ----------
    tube_verts_glo_list : list of np.ndarray, shape (dpi, N, 3)
    ds_r : int
        Downsample rate along the centerline (keep every ds_r-th ring).

    Returns
    -------
    faces_list : list of np.ndarray, shape (2*(dpi'-1)*N, 3)
        Triangle face indices into the flattened (dpi'*N) vertex array, per branch.
    ds_tube_verts_glo_list : list of np.ndarray, shape (dpi', N, 3)
        Downsampled tube vertices, per branch.
    """
    faces_list = []
    ds_tube_verts_glo_list = []

    for tube_verts_glo in tube_verts_glo_list:
        tube_verts_glo = tube_verts_glo[::ds_r]          # (dpi', N, 3)
        ds_tube_verts_glo_list.append(tube_verts_glo)
        dpi, N, _ = tube_verts_glo.shape

        # faces_element[d, n] = d*N + n  — flat index of vertex (ring d, node n)
        d_idx = np.arange(dpi - 1)
        n_idx = np.arange(N)
        d_grid, n_grid = np.meshgrid(d_idx, n_idx, indexing='ij')  # (dpi-1, N)
        faces_element = d_grid * N + n_grid                         # (dpi-1, N)

        f1 = faces_element.ravel()                          # d*N + n
        f2 = np.roll(faces_element, -1, axis=1).ravel()    # d*N + (n+1)%N  (wraps around)
        f3 = f1 + N                                         # (d+1)*N + n

        tri1 = np.stack([f1, f2, f3],      axis=1)         # (dpi-1)*N, 3)
        tri2 = np.stack([f2, f2 + N, f3], axis=1)

        faces_list.append(np.concatenate([tri1, tri2], axis=0))  # (2*(dpi-1)*N, 3)

    return faces_list, ds_tube_verts_glo_list


def get_face_patch(opening_idx_sorted, n_count):
    """
    Build the collar triangles that stitch the original mesh's open boundary
    ring to the first ring of the tube.

    Numpy port of get_face_patch in mesh_plugins.py (no batch dim).

    The patch is a quad strip between two N-node rings:
      - "original" ring : opening_idx_sorted[n]   (indices into original mesh)
      - "tube" ring     : n_count + n             (indices into appended tube verts)

    Each quad is split into two triangles:
      tri1: (orig[n],      orig[n+1]%N, tube[n])
      tri2: (orig[n+1]%N,  tube[n+1]%N, tube[n])

    Parameters
    ----------
    opening_idx_sorted : np.ndarray, shape (N,)
        Original mesh vertex indices of the boundary ring, sorted by angle.
    n_count : int
        Vertex offset at which the first tube ring starts in the merged array.

    Returns
    -------
    patch_faces : np.ndarray, shape (2*N, 3)
    """
    N = len(opening_idx_sorted)
    orig      = opening_idx_sorted                    # (N,)
    orig_next = np.roll(opening_idx_sorted, -1)       # (N,) circular
    tube      = np.arange(N) + n_count                # (N,)
    tube_next = np.roll(tube, -1)                     # (N,) circular

    tri1 = np.stack([orig,      orig_next, tube],      axis=1)  # (N, 3)
    tri2 = np.stack([orig_next, tube_next, tube],      axis=1)  # (N, 3)
    return np.concatenate([tri1, tri2], axis=0)                  # (2*N, 3)


def merge_meshes(mesh, tube_faces_list, tube_verts_glo_list, opening_idx_sorted_list):
    """
    Merge the clipped vessel mesh with the generated tubular extensions.

    Numpy / PyVista port of merge_meshes in mesh_plugins.py (no batch dim).

    For each branch:
      1. Record current vertex count as n_count.
      2. Flatten tube verts (dpi, N, 3) → (dpi*N, 3) and append.
      3. Append tube faces (offset by n_count).
      4. Append collar patch faces that bridge the open boundary to the tube.

    Parameters
    ----------
    mesh : str or pyvista.PolyData
        The clipped vessel / aneurysm mesh.
    tube_faces_list : list of np.ndarray, shape (2*(dpi-1)*N_i, 3)
    tube_verts_glo_list : list of np.ndarray, shape (dpi, N_i, 3)
    opening_idx_sorted_list : list of np.ndarray, shape (N_i,)

    Returns
    -------
    merged : pyvista.PolyData
    """
    mesh = pv.read(mesh) if isinstance(mesh, str) else mesh
    verts = np.array(mesh.points, dtype=float)
    faces = mesh.faces.reshape(-1, 4)[:, 1:].copy()  # (F, 3)

    for tube_verts, tube_faces, opening_idx_sorted in zip(
            tube_verts_glo_list, tube_faces_list, opening_idx_sorted_list):
        n_count = len(verts)
        dpi, N, _ = tube_verts.shape

        verts = np.vstack([verts, tube_verts.reshape(dpi * N, 3)])
        faces = np.vstack([faces, tube_faces + n_count])
        faces = np.vstack([faces, get_face_patch(opening_idx_sorted, n_count)])

    faces_pv = np.hstack([np.full((len(faces), 1), 3), faces]).ravel()
    return pv.PolyData(verts, faces_pv)


def forward_mesh_fusion(post_dir,
                        flaw_opening_min_size=10,
                        init_step=1,
                        min_torsion=False):
    # load mesh & centerlines
    mesh_path = os.path.join(post_dir, "clipped_reconstruction.ply")
    mesh = pv.read(mesh_path)
    opening_verts_list, vertex_ids_list = detect_openings(mesh, flaw_opening_min_size=flaw_opening_min_size)
    cl_path = os.path.join(post_dir, "clipped_centerline.npy")
    cl = np.load(cl_path, allow_pickle=True).item()
    branch_ids = [cl['upstream_id']] + [id for id in cl['connected_branch_ids'] if id != cl['upstream_id']]
    branches = [cl['branches'][id]['pts'] for id in branch_ids if cl['branches'][id]['pts'] is not None]
    # interpolate branch pcd at the mesh's average edge length
    step = avg_edge_length(mesh)
    cpcd_glo = [resample_branch(b, step) for b in branches]

    cpcd_glo_tangent = [get_cpcd_tangent(cpcd) for cpcd in cpcd_glo]
    cpcd_glo = [branch[init_step:] for branch in cpcd_glo]  # skip first few points to avoid noise at opening
    cpcd_glo_tangent = [tangent[init_step:] for tangent in cpcd_glo_tangent]

    # Match each branch to its nearest opening using the branch start point (cut end)
    opening_centroids = np.array([o.mean(axis=0) for o in opening_verts_list])  # (num_openings, 3)
    matched_opening_verts = []
    matched_vertex_ids = []
    for cpcd in cpcd_glo:
        dists = np.linalg.norm(opening_centroids - cpcd[0], axis=1)
        best = int(np.argmin(dists))
        matched_opening_verts.append(opening_verts_list[best])
        matched_vertex_ids.append(vertex_ids_list[best])

    # get tubular mesh
    l2w_trans_list = [get_tubular_l2w_trans(tangent, min_torsion=min_torsion) for tangent in cpcd_glo_tangent]
    tube_verts_glo_list, sort_indices_list = get_tubular_mesh_verts(
        opening_pcd=matched_opening_verts,
        cpcd_glo=cpcd_glo,
        cpcd_tangent_glo=cpcd_glo_tangent,
        l2w_trans=l2w_trans_list,
    )
    # reorder opening vertex indices to match the angular sort applied inside get_tubular_mesh_verts
    opening_idx_sorted_list = [ids[sort_idx] for ids, sort_idx in zip(matched_vertex_ids, sort_indices_list)]
    tube_faces_list, tube_verts_glo_list = get_tubular_mesh_faces(tube_verts_glo_list, ds_r=3)
    merged = merge_meshes(mesh, tube_faces_list, tube_verts_glo_list, opening_idx_sorted_list)
    return merged




def ring_outward_normal_from_mesh(mesh_points, mesh_faces, ring_vertex_ids):
    """Ring plane normal by SVD, oriented outward using ONLY the mesh itself.

    ring_normal_and_centroid() needs a reference tangent to disambiguate SVD's
    arbitrary sign. That makes the extrusion direction depend on the centerline
    even though the ring alone already determines the plane. The mesh knows
    which side it is on: take the ring's one-ring neighbours that are NOT on the
    ring -- those lie in the body -- and orient the normal away from them.

    Returns (normal, centroid), or (None, None) if the ring is unusable or has
    no interior neighbours to orient against.
    """
    ring_vertex_ids = np.asarray(ring_vertex_ids).ravel()
    if ring_vertex_ids.size < 3:
        return None, None
    P = np.asarray(mesh_points, dtype=float)[ring_vertex_ids]
    centroid = P.mean(axis=0)
    normal = np.linalg.svd(P - centroid, full_matrices=False)[2][-1]

    ring = set(int(i) for i in ring_vertex_ids)
    F = np.asarray(mesh_faces).reshape(-1, 3) if np.asarray(mesh_faces).ndim == 2 \
        else np.asarray(mesh_faces).reshape(-1, 4)[:, 1:]
    touching = F[np.isin(F, list(ring)).any(axis=1)]
    interior = [int(v) for v in np.unique(touching) if int(v) not in ring]
    if interior:
        body = np.asarray(mesh_points, dtype=float)[interior].mean(axis=0)
        # outward = away from the body vertices adjacent to the ring
        if np.dot(normal, centroid - body) < 0:
            normal = -normal
    else:
        return None, None
    n = np.linalg.norm(normal)
    return (normal / n, centroid) if n > 1e-12 else (None, None)


def ring_normal_and_centroid(opening_verts, reference_tangent):
    """A boundary ring's own plane normal (via SVD) and centroid.

    OPT-IN helper for forward_mesh_fusion_v2(ring_normal_extrusion=True); the
    default extrusion path does not call it and is unchanged.

    Only the extrusion DIRECTION is taken from the ring; the start point stays
    cpcd[-1] as before. After truncation cpcd[-1] already sits on the opening
    (~0.1 mm from the ring centroid on p171), so moving the start would only
    risk a discontinuity in the path the tube is swept along.

    An AUTOMATIC clip cuts perpendicular to the centerline, so the ring's plane
    normal and the branch's end tangent agree and extruding along the tangent
    is fine. A HAND-CUT ("_manual") clip is oblique: the ring plane is tilted
    relative to the tangent, so tangent-directed rings intersect the opening
    instead of clearing it, and the stitching quads fold into inverted /
    zero-area triangles.

    Extruding along the RING's own normal, starting from the RING's own
    centroid, makes the first ring parallel to the opening by construction --
    no fold-back regardless of how oblique the cut is.

    SVD's normal has arbitrary sign, so it is flipped to agree with
    `reference_tangent` (the centerline end tangent, which always points out of
    the vessel); otherwise the tube would extrude backwards into the mesh.

    Returns (normal, centroid), or (None, None) if the ring is unusable.
    """
    if opening_verts is None or len(opening_verts) < 3:
        return None, None
    P = np.asarray(opening_verts, dtype=float)
    centroid = P.mean(axis=0)
    # smallest right-singular vector of the centred ring = plane normal
    normal = np.linalg.svd(P - centroid, full_matrices=False)[2][-1]
    ref = np.asarray(reference_tangent, dtype=float)
    if np.dot(normal, ref) < 0:
        normal = -normal
    n = np.linalg.norm(normal)
    if n < 1e-12:
        return None, None
    return normal / n, centroid


def forward_mesh_fusion_v2(save_dir,
                            flaw_opening_min_size=10,
                            init_step=3,
                            max_cl_length=10.0,
                            min_cl_length=8.0,
                            ring_downsample_ratio=1,
                            r_clipped_mesh_filename="clipped_reconstruction.ply",
                            r_clipped_cl_filename="clipped_centerline.npy",
                            w_forward_fusion_info_filename="forward_fusion_info.npz",
                            w_merged_mesh_filename="merged_reconstruction.ply",
                            branch_ids: list=None,
                            r_branch_ranking_filename="branch_ranking.npy",
                            extrude_outlets=False,
                            extrude_inlet=False,
                            extrude_length=2.5,
                            min_torsion=False,
                            ring_normal_extrusion=False):
    """
    Extend the openings of a clipped dome mesh with tubular vessel segments guided
    by the clipped centerlines. The openings are automatically deteceted and matched
    to centerline branches. Tubular meshes for the vessels are generated and merged
    with the dome mesh.

    Parameters
    ----------
    post_dir : str
        Case directory containing clipped_reconstruction.ply and clipped_centerline.npy.
    flaw_opening_min_size : int
        Detected openings with fewer boundary vertices than this are ignored (to filter out small flaws).
    init_step : int
        Number of resampled points to skip at the opening end to avoid noise. (usually 1-3).
        Larger values are needed for jagged openings.
    max_cl_length : float or list or None
        Maximum arc-length (mm) of centerline used per branch. If None,
        defaults to 10.0 for the first branch (upstream) and 2.0 for all others.
    ring_downsample_ratio : int
        Step size for downsampling cross-section rings along the centerline.
        Larger value is needed for mesh with collapse issues.
    Returns
    -------
    merged : pyvista.PolyData
        The dome mesh with tubular extensions merged at each opening.
    """
    # load mesh & centerlines
    mesh_path = os.path.join(r_clipped_mesh_filename)
    mesh = pv.read(mesh_path)
    # kept for ring_outward_normal_from_mesh: it orients each ring from the
    # mesh's own topology instead of a centerline tangent
    verts = np.asarray(mesh.points)
    faces = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]
    opening_verts_list, vertex_ids_list = detect_openings(mesh, flaw_opening_min_size=flaw_opening_min_size)
    cl_path = os.path.join(r_clipped_cl_filename)
    cl = np.load(cl_path, allow_pickle=True).item()
    if branch_ids is None:
        branch_ids = [cl['upstream_id']] + [id for id in cl['connected_branch_ids'] if id != cl['upstream_id']]
    if r_branch_ranking_filename is not None:
        branch_ranking = np.load(r_branch_ranking_filename)
        branch_ids = [branch_ids[i] for i in branch_ranking]
    branches = [cl['branches'][id]['pts'] for id in branch_ids if cl['branches'][id]['pts'] is not None]
    # interpolate branch pcd at the mesh's average edge length
    step = avg_edge_length(mesh)
    cpcd_glo = [resample_branch(b, step) for b in branches]
    cpcd_glo_tangent = [get_cpcd_tangent(cpcd) for cpcd in cpcd_glo]
    cpcd_glo = [branch[init_step:] for branch in cpcd_glo]  # skip first few points to avoid noise at opening
    cpcd_glo_tangent = [tangent[init_step:] for tangent in cpcd_glo_tangent]

    # Match each branch to its nearest opening using whichever endpoint is closer
    opening_centroids = np.array([o.mean(axis=0) for o in opening_verts_list])  # (num_openings, 3)
    matched_opening_verts = []
    matched_vertex_ids = []
    matched_opening_centroids = []
    for cpcd in cpcd_glo:
        d_start = np.linalg.norm(opening_centroids - cpcd[0], axis=1)
        d_end = np.linalg.norm(opening_centroids - cpcd[-1], axis=1)
        best = int(np.argmin(np.minimum(d_start, d_end)))
        matched_opening_verts.append(opening_verts_list[best])
        matched_vertex_ids.append(vertex_ids_list[best])
        matched_opening_centroids.append(opening_centroids[best])

    # Flip any branch whose first point is farther from its matched opening than its last point
    for i, (cpcd, tangent, oc) in enumerate(zip(cpcd_glo, cpcd_glo_tangent, matched_opening_centroids)):
        if np.linalg.norm(cpcd[-1] - oc) < np.linalg.norm(cpcd[0] - oc):
            cpcd_glo[i] = cpcd[::-1]
            cpcd_glo_tangent[i] = tangent[::-1]

    # Extrude all branches to at least min_cl_length along the end tangent
    # First compute average step size for extrusion.
    all_steps = []
    for cpcd in cpcd_glo:
        if len(cpcd) >= 2:
            all_steps.extend(np.linalg.norm(np.diff(cpcd, axis=0), axis=1).tolist())
    avg_pt_step = np.mean(all_steps) if all_steps else step

    for i, (cpcd, tangent) in enumerate(zip(cpcd_glo, cpcd_glo_tangent)):
        if len(cpcd) < 2:
            continue
        arc_dists = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(cpcd, axis=0), axis=1))])
        total_arc = arc_dists[-1]
        if total_arc < min_cl_length:
            print('detect short branch (<8mm), extruded.')
            pt_step = avg_pt_step
            if pt_step == 0:
                pt_step = step
            remaining = min_cl_length - total_arc
            n_extra = int(np.ceil(remaining / pt_step))
            last_tan = tangent[-1]
            extra_pts = cpcd[-1] + last_tan * (pt_step * np.arange(1, n_extra + 1))[:, None]
            extra_tan = np.tile(last_tan, (n_extra, 1))
            cpcd_glo[i] = np.vstack([cpcd, extra_pts])
            cpcd_glo_tangent[i] = np.vstack([tangent, extra_tan])

    # Truncate branches that exceed max_cl_length (arc distance from first to last point)
    truncated_cpcd_glo = []
    truncated_tangent_glo = []
    if max_cl_length is None:
        max_cl_length = [10.0] + [2.5] * (len(cpcd_glo) - 1)
    elif isinstance(max_cl_length, (int, float)):
        max_cl_length = [max_cl_length] * len(cpcd_glo)
    elif isinstance(max_cl_length, list):
        max_cl_length = max_cl_length[:len(cpcd_glo)]
    bid = 0 
    for cpcd, tangent in zip(cpcd_glo, cpcd_glo_tangent):
        arc_dists = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(cpcd, axis=0), axis=1))])
        cutoff = np.searchsorted(arc_dists, max_cl_length[bid], side='right')
        truncated_cpcd_glo.append(cpcd[:cutoff])
        truncated_tangent_glo.append(tangent[:cutoff])
        bid += 1
    cpcd_glo = truncated_cpcd_glo
    cpcd_glo_tangent = truncated_tangent_glo

    # Extend outlet branches by extrude_length along end tangent.
    # cpcd_glo[0] is the upstream inlet — skip it, extend indices 1+ only.
    if extrude_outlets:
        n_extra = max(1, int(np.ceil(extrude_length / avg_pt_step)))
        for i in range(1, len(cpcd_glo)):
            cpcd, tangent = cpcd_glo[i], cpcd_glo_tangent[i]
            last_tan = tangent[-1]
            # OPT-IN: march along the RING's own normal from the RING's own
            # centroid, so the first extruded ring is parallel to an oblique
            # (hand-cut) opening instead of slicing through it.
            ring_n = ring_c = None
            if ring_normal_extrusion and i < len(matched_opening_verts):
                # centerline-free first: the mesh itself says which side is out
                if i < len(matched_vertex_ids):
                    ring_n, ring_c = ring_outward_normal_from_mesh(
                        verts, faces, matched_vertex_ids[i])
                if ring_n is None:
                    ring_n, ring_c = ring_normal_and_centroid(matched_opening_verts[i], last_tan)
            if ring_n is None:
                ring_n = last_tan
            extra_pts = cpcd[-1] + ring_n * (avg_pt_step * np.arange(1, n_extra + 1))[:, None]
            extra_tan = np.tile(ring_n, (n_extra, 1))
            cpcd_glo[i] = np.vstack([cpcd, extra_pts])
            cpcd_glo_tangent[i] = np.vstack([tangent, extra_tan])

    # Extend the inlet branch (cpcd_glo[0]) by extrude_length along its end tangent.
    if extrude_inlet and len(cpcd_glo) > 0:
        n_extra = max(1, int(np.ceil(2 * extrude_length / avg_pt_step)))
        cpcd, tangent = cpcd_glo[0], cpcd_glo_tangent[0]
        last_tan = tangent[-1]
        ring_n = ring_c = None
        if ring_normal_extrusion and matched_opening_verts:
            if matched_vertex_ids:
                ring_n, ring_c = ring_outward_normal_from_mesh(verts, faces, matched_vertex_ids[0])
            if ring_n is None:
                ring_n, ring_c = ring_normal_and_centroid(matched_opening_verts[0], last_tan)
        if ring_n is None:
            ring_n = last_tan
        extra_pts = cpcd[-1] + ring_n * (avg_pt_step * np.arange(1, n_extra + 1))[:, None]
        extra_tan = np.tile(ring_n, (n_extra, 1))
        cpcd_glo[0] = np.vstack([cpcd, extra_pts])
        cpcd_glo_tangent[0] = np.vstack([tangent, extra_tan])

    # downsample rings along the centerline to reduce mesh complexity
    cpcd_glo = [cpcd[::ring_downsample_ratio] for cpcd in cpcd_glo]
    cpcd_glo_tangent = [tangent[::ring_downsample_ratio] for tangent in cpcd_glo_tangent]

    # get tubular mesh
    l2w_trans_list = [get_tubular_l2w_trans(tangent, min_torsion=min_torsion) for tangent in cpcd_glo_tangent]
    tube_verts_glo_list, sort_indices_list = get_tubular_mesh_verts(
        opening_pcd=matched_opening_verts,
        cpcd_glo=cpcd_glo,
        cpcd_tangent_glo=cpcd_glo_tangent,
        l2w_trans=l2w_trans_list,
    )
    # reorder opening vertex indices to match the angular sort applied inside get_tubular_mesh_verts
    opening_idx_sorted_list = [ids[sort_idx] for ids, sort_idx in zip(matched_vertex_ids, sort_indices_list)]
    tube_faces_list, tube_verts_glo_list = get_tubular_mesh_faces(tube_verts_glo_list, ds_r=3)
    merged = merge_meshes(mesh, tube_faces_list, tube_verts_glo_list, opening_idx_sorted_list)
    # save processed centerline and opening information
    if w_forward_fusion_info_filename is not None:
        np.savez(os.path.join(save_dir, w_forward_fusion_info_filename),
                cpcd_glo=np.array(cpcd_glo, dtype=object),
                cpcd_glo_tangent=np.array(cpcd_glo_tangent, dtype=object),
                opening_centroids=np.array([oc for oc in matched_opening_centroids]),
                opening_vertex_ids=np.array(matched_vertex_ids, dtype=object),
                allow_pickle=True)
    if w_merged_mesh_filename is not None:
        merged.save(os.path.join(save_dir, w_merged_mesh_filename))
    return merged



def ghd_forward_mesh_fusion(save_dir,
                            flaw_opening_min_size=10,
                            init_step=4,
                            max_cl_length=10.0,
                            ring_downsample_ratio=1,
                            r_ghd_mesh_filename="ghd_fitted_uncapped_world.obj",
                            r_clipped_cl_filename="clipped_centerline.npy",
                            r_forward_fusion_info_filename="forward_fusion_info.npz",
                            r_branch_ranking_filename="branch_ranking.npy",
                            w_ghd_reconstructed_filename="ghd_reconstructed.obj",
                            w_ghd_forward_fusion_info_filename="ghd_forward_fusion_info.npz",
                            w_merged_mesh_filename="ghd_merged_reconstruction.ply",
                            w_smoothed_mesh_filename="ghd_smoothed_reconstruction.ply",
                            planarize_n_iter=10,
                            planarize_lam=0.5,
                            smooth_n_rings=3,
                            smooth_n_iter=10,
                            smooth_lam=0.5,
                            extrude_outlets=False,
                            extrude_inlet=False,
                            min_torsion=True,):
    """
    Full GHD-to-CFD mesh pipeline for one case:

      1. Planarize the open boundary rings of the GHD-fitted mesh using the
         outward tangents stored in the forward-fusion info file.
      2. Merge the planarized mesh with tubular vessel extensions guided by
         the clipped centerlines.
      3. Laplacian-smooth the merged mesh in the neighbourhood of each opening
         to clean up the seam between the original surface and the tubes.

    Parameters
    ----------
    post_dir : str
        Case directory containing the centerline and forward-fusion info files.
    ghd_dir : str
        Directory containing the GHD-fitted uncapped mesh.
    flaw_opening_min_size : int
        Minimum boundary-ring size; smaller rings are treated as mesh flaws.
    init_step : int
        Centerline points to skip at the opening end (avoids noise).
    max_cl_length : float or list or None
        Maximum arc-length (mm) of centerline used per branch. If None,
        defaults to 10.0 for the first branch (upstream) and 2.0 for all others.
    ring_downsample_ratio : int
        Downsampling step along the centerline rings.
    r_ghd_mesh_filename : str
        GHD uncapped mesh filename inside ghd_dir.
    r_clipped_cl_filename : str
        Clipped centerline filename inside post_dir.
    r_forward_fusion_info_filename : str
        Forward-fusion info filename inside post_dir (used for planarization).
    w_ghd_reconstructed_filename : str
        Output filename for the planarized mesh (written to post_dir).
    w_ghd_forward_fusion_info_filename : str
        Output filename for the GHD forward-fusion info (written to post_dir).
    w_merged_mesh_filename : str
        Output filename for the merged mesh (written to post_dir).
    w_smoothed_mesh_filename : str
        Output filename for the smoothed mesh (written to post_dir).
        Set to None to skip saving.
    planarize_n_iter : int
        Laplacian iterations for opening planarization.
    planarize_lam : float
        Smoothing weight for opening planarization.
    smooth_n_rings : int
        Neighbourhood width (edge hops) for seam smoothing.
    smooth_n_iter : int
        Laplacian iterations for seam smoothing.
    smooth_lam : float
        Smoothing weight for seam smoothing.

    Returns
    -------
    smoothed : pyvista.PolyData
    """
    from .patching import planarize_openings, smooth_near_openings
    # --- Step 1: Planarize openings of the GHD mesh ---
    ghd_mesh_path = r_ghd_mesh_filename
    ghd_mesh      = pv.read(ghd_mesh_path)
    planarized    = planarize_openings(
        ghd_mesh,
        r_forward_fusion_info_filename=r_forward_fusion_info_filename,
        flaw_opening_min_size=flaw_opening_min_size,
        n_iter=planarize_n_iter,
        lam=planarize_lam,
    )
    w_ghd_reconstructed_path = w_ghd_reconstructed_filename
    pv.save_meshio(w_ghd_reconstructed_path, planarized)

    # --- Step 2: Merge with tubular extensions ---
    merged = forward_mesh_fusion_v2(
        save_dir=save_dir,
        flaw_opening_min_size=flaw_opening_min_size,
        init_step=init_step,
        max_cl_length=max_cl_length,
        ring_downsample_ratio=ring_downsample_ratio,
        r_clipped_mesh_filename=w_ghd_reconstructed_path,
        r_clipped_cl_filename=r_clipped_cl_filename,
        w_forward_fusion_info_filename=w_ghd_forward_fusion_info_filename,
        w_merged_mesh_filename=w_merged_mesh_filename,
        r_branch_ranking_filename=r_branch_ranking_filename,
        extrude_outlets=extrude_outlets,
        extrude_inlet=extrude_inlet,
        min_torsion=min_torsion,
    )

    # --- Step 3: Smooth the seam near each opening ---
    smoothed = smooth_near_openings(
        merged,
        r_forward_fusion_info_filename=w_ghd_forward_fusion_info_filename,
        n_rings=smooth_n_rings,
        n_iter=smooth_n_iter,
        lam=smooth_lam,
    )
    smoothed = remove_orphan_vertices(smoothed)  # clean up any disconnected verts from smoothing

    # final planarization
    smoothed    = planarize_openings(
        smoothed,
        r_forward_fusion_info_filename=None,
        flaw_opening_min_size=flaw_opening_min_size,
        smooth=False
    )
    if w_smoothed_mesh_filename is not None:
        smoothed.save(os.path.join(save_dir, w_smoothed_mesh_filename))

    return smoothed


def aneux_forward_mesh_fusion(save_dir,
                            flaw_opening_min_size=10,
                            init_step=4,
                            max_cl_length=10.0,
                            ring_downsample_ratio=1,
                            r_ghd_mesh_filename="clipped_reconstruction.obj",
                            r_clipped_cl_filename="clipped_centerline.npy",
                            r_forward_fusion_info_filename="forward_fusion_info.npz",
                            r_branch_ranking_filename="branch_ranking.npy",
                            w_ghd_reconstructed_filename="ghd_reconstructed.obj",
                            w_ghd_forward_fusion_info_filename="ghd_forward_fusion_info.npz",
                            w_merged_mesh_filename="ghd_merged_reconstruction.ply",
                            w_smoothed_mesh_filename="ghd_smoothed_reconstruction.ply",
                            planarize_n_iter=10,
                            planarize_lam=0.5,
                            smooth_n_rings=3,
                            smooth_n_iter=10,
                            smooth_lam=0.5,
                            extrude_outlets=False,
                            min_torsion=False):
    """
    Full GHD-to-CFD mesh pipeline for one case:

      1. Planarize the open boundary rings of the GHD-fitted mesh using the
         outward tangents stored in the forward-fusion info file.
      2. Merge the planarized mesh with tubular vessel extensions guided by
         the clipped centerlines.
      3. Laplacian-smooth the merged mesh in the neighbourhood of each opening
         to clean up the seam between the original surface and the tubes.

    Parameters
    ----------
    post_dir : str
        Case directory containing the centerline and forward-fusion info files.
    ghd_dir : str
        Directory containing the GHD-fitted uncapped mesh.
    flaw_opening_min_size : int
        Minimum boundary-ring size; smaller rings are treated as mesh flaws.
    init_step : int
        Centerline points to skip at the opening end (avoids noise).
    max_cl_length : float or list or None
        Maximum arc-length (mm) of centerline used per branch. If None,
        defaults to 10.0 for the first branch (upstream) and 2.0 for all others.
    ring_downsample_ratio : int
        Downsampling step along the centerline rings.
    r_ghd_mesh_filename : str
        GHD uncapped mesh filename inside ghd_dir.
    r_clipped_cl_filename : str
        Clipped centerline filename inside post_dir.
    r_forward_fusion_info_filename : str
        Forward-fusion info filename inside post_dir (used for planarization).
    w_ghd_reconstructed_filename : str
        Output filename for the planarized mesh (written to post_dir).
    w_ghd_forward_fusion_info_filename : str
        Output filename for the GHD forward-fusion info (written to post_dir).
    w_merged_mesh_filename : str
        Output filename for the merged mesh (written to post_dir).
    w_smoothed_mesh_filename : str
        Output filename for the smoothed mesh (written to post_dir).
        Set to None to skip saving.
    planarize_n_iter : int
        Laplacian iterations for opening planarization.
    planarize_lam : float
        Smoothing weight for opening planarization.
    smooth_n_rings : int
        Neighbourhood width (edge hops) for seam smoothing.
    smooth_n_iter : int
        Laplacian iterations for seam smoothing.
    smooth_lam : float
        Smoothing weight for seam smoothing.

    Returns
    -------
    smoothed : pyvista.PolyData
    """
    from .patching import planarize_openings, smooth_near_openings
    # --- Step 1: Planarize openings of the GHD mesh ---
    ghd_mesh_path = r_ghd_mesh_filename
    ghd_mesh      = pv.read(ghd_mesh_path)
    planarized    = planarize_openings(
        ghd_mesh,
        r_forward_fusion_info_filename=r_forward_fusion_info_filename,
        flaw_opening_min_size=flaw_opening_min_size,
        n_iter=planarize_n_iter,
        lam=planarize_lam,
    )
    w_ghd_reconstructed_path = w_ghd_reconstructed_filename
    pv.save_meshio(w_ghd_reconstructed_path, planarized)

    # --- Step 2: Merge with tubular extensions ---
    merged = forward_mesh_fusion_v2(
        save_dir=save_dir,
        flaw_opening_min_size=flaw_opening_min_size,
        init_step=init_step,
        max_cl_length=max_cl_length,
        ring_downsample_ratio=ring_downsample_ratio,
        r_clipped_mesh_filename=w_ghd_reconstructed_path,
        r_clipped_cl_filename=r_clipped_cl_filename,
        w_forward_fusion_info_filename=w_ghd_forward_fusion_info_filename,
        w_merged_mesh_filename=w_merged_mesh_filename,
        r_branch_ranking_filename=r_branch_ranking_filename,
        extrude_outlets=extrude_outlets,
        min_torsion=min_torsion,
    )

    # --- Step 3: Smooth the seam near each opening ---
    smoothed = smooth_near_openings(
        merged,
        r_forward_fusion_info_filename=w_ghd_forward_fusion_info_filename,
        n_rings=smooth_n_rings,
        n_iter=smooth_n_iter,
        lam=smooth_lam,
    )
    smoothed = remove_orphan_vertices(smoothed)  # clean up any disconnected verts from smoothing

    # final planarization
    smoothed    = planarize_openings(
        smoothed,
        r_forward_fusion_info_filename=None,
        flaw_opening_min_size=flaw_opening_min_size,
        smooth=False
    )
    if w_smoothed_mesh_filename is not None:
        smoothed.save(os.path.join(save_dir, w_smoothed_mesh_filename))

    return smoothed