"""
Self-contained tubular-mesh fusion helpers.

Copied (with light edits) from Aneu_GHD/utils/vessel_reconstruct.py and
Aneu_GHD/utils/patching.py so the v2 pipeline does not depend on the Aneu_GHD
package. Only numpy / pyvista / networkx are required, and they are imported
lazily where possible so importing this module stays cheap.

Functions
---------
avg_edge_length, resample_branch, get_cpcd_tangent, get_tubular_l2w_trans,
get_tubular_mesh_verts, get_tubular_mesh_faces, get_face_patch, merge_meshes
    — build tubular extensions from opening rings + centerlines and glue them
      onto a surface mesh.
remove_orphan_vertices, flatten_and_smooth_opening, planarize_openings,
smooth_near_openings
    — clean up / planarize / smooth the merged surface.
"""

import numpy as np


# ── tubular mesh generation ─────────────────────────────────────────────────

def avg_edge_length(mesh):
    """Return (0.85 *) the median edge length of a triangulated PolyData."""
    faces = mesh.faces.reshape(-1, 4)[:, 1:]  # (F, 3)
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges_sorted = np.sort(edges, axis=1)
    unique_edges = np.unique(edges_sorted, axis=0)
    lengths = np.linalg.norm(mesh.points[unique_edges[:, 0]] - mesh.points[unique_edges[:, 1]], axis=1)
    return 0.85 * np.median(lengths)


def resample_branch(pts, step, min_n=5):
    """Resample a polyline (N, 3) at uniform arc-length intervals of `step`."""
    pts = np.asarray(pts, dtype=float)
    dists = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cumlen = np.concatenate([[0.0], np.cumsum(dists)])
    total = cumlen[-1]
    sample_s = np.arange(0.0, total, step)
    if len(sample_s) == 0 or sample_s[-1] < total:
        sample_s = np.append(sample_s, total)
    if len(sample_s) < min_n:
        sample_s = np.linspace(0.0, total, min_n)
    return np.column_stack([np.interp(sample_s, cumlen, pts[:, dim]) for dim in range(3)])


def get_cpcd_tangent(cpcd):
    """Per-point unit tangent vectors for a resampled centerline polyline (N, 3)."""
    cpcd = np.asarray(cpcd, dtype=float)
    tangents = np.empty_like(cpcd)
    tangents[0] = cpcd[1] - cpcd[0]
    tangents[-1] = cpcd[-1] - cpcd[-2]
    tangents[1:-1] = cpcd[2:] - cpcd[:-2]
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    return tangents / norms


def get_tubular_l2w_trans(cpcd_tangent_glo, min_torsion=True):
    """Local-to-world rotation matrix at each centerline point.

    Local axes: x along tangent, z = reference projected perpendicular to x,
    y = cross(z, x). Convention: p_world = p_local @ l2w_trans[i].
    If min_torsion, use a rotation-minimizing (parallel transport) frame.
    """
    t = np.asarray(cpcd_tangent_glo, dtype=float)
    t = t / np.linalg.norm(t, axis=-1, keepdims=True)
    N = len(t)

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

    z0 = ref - np.dot(ref, t[0]) * t[0]
    z0 /= np.linalg.norm(z0)
    y0 = np.cross(z0, t[0])
    y0 /= np.linalg.norm(y0)

    x_axes = np.empty_like(t)
    y_axes = np.empty_like(t)
    z_axes = np.empty_like(t)
    x_axes[0], y_axes[0], z_axes[0] = t[0], y0, z0

    for i in range(1, N):
        t_prev = x_axes[i - 1]
        t_curr = t[i]
        axis = np.cross(t_prev, t_curr)
        sin_a = np.linalg.norm(axis)
        cos_a = float(np.clip(np.dot(t_prev, t_curr), -1.0, 1.0))
        if sin_a < 1e-12:
            x_axes[i] = t_curr
            y_axes[i] = y_axes[i - 1]
            z_axes[i] = z_axes[i - 1]
            continue
        axis /= sin_a

        def _rot(v):
            return v * cos_a + np.cross(axis, v) * sin_a + axis * (np.dot(axis, v) * (1.0 - cos_a))

        x_axes[i] = t_curr
        y_new = _rot(y_axes[i - 1])
        y_new -= np.dot(y_new, t_curr) * t_curr
        y_new /= np.linalg.norm(y_new)
        y_axes[i] = y_new
        z_axes[i] = np.cross(x_axes[i], y_axes[i])

    return np.stack([x_axes, y_axes, z_axes], axis=-2)


def get_tubular_mesh_verts(opening_pcd, cpcd_glo, cpcd_tangent_glo, l2w_trans,
                           radius_map=True, c_transition=True):
    """Generate tubular mesh vertices from opening rings and centerline data.

    Returns (tube_verts_glo_list, sort_indices_list).
    """
    tube_verts_glo_list = []
    sort_indices_list = []

    for opening, cpcd, tangents, l2w in zip(opening_pcd, cpcd_glo, cpcd_tangent_glo, l2w_trans):
        opening = np.asarray(opening, dtype=float)
        N = len(opening)
        dpi = len(cpcd)

        centroid = cpcd[0]
        tangent = tangents[0]

        # 1. Project opening ring onto the neck plane.
        D = -np.dot(centroid, tangent)
        tproj = -(opening @ tangent + D) / np.dot(tangent, tangent)
        pop_glo = opening + tproj[:, None] * tangent

        # 2. Radii.
        rays = opening - centroid
        dist = np.linalg.norm(rays, axis=-1)
        if radius_map:
            radius_set = np.sqrt(np.maximum(dist**2 - (rays @ tangent)**2, 0.0))
        else:
            radius_set = dist

        # 3. Sort by angle in local frame.
        w2l_0 = l2w[0].T
        pop_loc = (pop_glo - centroid) @ w2l_0
        loc_angle = np.arctan2(pop_loc[:, 2], pop_loc[:, 1])
        sort_idx = np.argsort(loc_angle)
        loc_angle_sorted = loc_angle[sort_idx]
        radius_sorted = radius_set[sort_idx]
        sort_indices_list.append(sort_idx)

        # 4. Build local tube vertices (dpi, N, 3).
        tube_loc_x = np.zeros((dpi, N))
        if not c_transition:
            tube_loc_y = np.outer(np.ones(dpi), np.cos(loc_angle_sorted) * radius_sorted)
            tube_loc_z = np.outer(np.ones(dpi), np.sin(loc_angle_sorted) * radius_sorted)
        else:
            radius_avg = np.linalg.norm(radius_sorted) / np.sqrt(N)
            transit = np.linspace(0, 1, dpi)
            radius_transit = (radius_sorted[:, None]
                              + (radius_avg - radius_sorted[:, None]) * transit[None, :])
            tube_loc_y = (np.cos(loc_angle_sorted)[:, None] * radius_transit).T
            tube_loc_z = (np.sin(loc_angle_sorted)[:, None] * radius_transit).T

        tube_verts_loc = np.stack([tube_loc_x, tube_loc_y, tube_loc_z], axis=-1)

        # 5. Transform to world coordinates.
        tube_verts_glo = (np.einsum('dnc,dcl->dnl', tube_verts_loc, l2w) + cpcd[:, None, :])
        tube_verts_glo_list.append(tube_verts_glo)

    return tube_verts_glo_list, sort_indices_list


def get_tubular_mesh_faces(tube_verts_glo_list, ds_r=2):
    """Build triangle faces for each tubular mesh; downsample rings by ds_r."""
    faces_list = []
    ds_tube_verts_glo_list = []

    for tube_verts_glo in tube_verts_glo_list:
        tube_verts_glo = tube_verts_glo[::ds_r]
        ds_tube_verts_glo_list.append(tube_verts_glo)
        dpi, N, _ = tube_verts_glo.shape

        d_idx = np.arange(dpi - 1)
        n_idx = np.arange(N)
        d_grid, n_grid = np.meshgrid(d_idx, n_idx, indexing='ij')
        faces_element = d_grid * N + n_grid

        f1 = faces_element.ravel()
        f2 = np.roll(faces_element, -1, axis=1).ravel()
        f3 = f1 + N

        tri1 = np.stack([f1, f2, f3], axis=1)
        tri2 = np.stack([f2, f2 + N, f3], axis=1)
        faces_list.append(np.concatenate([tri1, tri2], axis=0))

    return faces_list, ds_tube_verts_glo_list


def get_face_patch(opening_idx_sorted, n_count):
    """Collar triangles bridging the original boundary ring to the first tube ring."""
    N = len(opening_idx_sorted)
    orig = opening_idx_sorted
    orig_next = np.roll(opening_idx_sorted, -1)
    tube = np.arange(N) + n_count
    tube_next = np.roll(tube, -1)
    tri1 = np.stack([orig, orig_next, tube], axis=1)
    tri2 = np.stack([orig_next, tube_next, tube], axis=1)
    return np.concatenate([tri1, tri2], axis=0)


def merge_meshes(mesh, tube_faces_list, tube_verts_glo_list, opening_idx_sorted_list):
    """Merge a surface mesh with generated tubular extensions (PolyData out)."""
    import pyvista as pv
    mesh = pv.read(mesh) if isinstance(mesh, str) else mesh
    verts = np.array(mesh.points, dtype=float)
    faces = mesh.faces.reshape(-1, 4)[:, 1:].copy()

    for tube_verts, tube_faces, opening_idx_sorted in zip(
            tube_verts_glo_list, tube_faces_list, opening_idx_sorted_list):
        n_count = len(verts)
        dpi, N, _ = tube_verts.shape
        verts = np.vstack([verts, tube_verts.reshape(dpi * N, 3)])
        faces = np.vstack([faces, tube_faces + n_count])
        faces = np.vstack([faces, get_face_patch(opening_idx_sorted, n_count)])

    faces_pv = np.hstack([np.full((len(faces), 1), 3), faces]).ravel()
    return pv.PolyData(verts, faces_pv)


# ── surface cleanup / smoothing ─────────────────────────────────────────────

def remove_orphan_vertices(mesh):
    """Remove vertices not referenced by any face and remap face indices."""
    import pyvista as pv
    mesh = pv.read(mesh) if isinstance(mesh, str) else mesh
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    referenced = np.unique(faces)
    new_index = np.full(len(mesh.points), -1, dtype=int)
    new_index[referenced] = np.arange(len(referenced))
    new_points = mesh.points[referenced]
    new_faces = new_index[faces]
    faces_pv = np.hstack([np.full((len(new_faces), 1), 3), new_faces]).ravel()
    return pv.PolyData(new_points, faces_pv)


def flatten_and_smooth_opening(mesh_points, vertex_ids, component_edges,
                               normal=None, n_iter=10, lam=0.5):
    """Flatten a boundary ring onto a plane, then Laplacian-smooth node spacing."""
    points = mesh_points[vertex_ids]
    centroid = points.mean(axis=0)

    if normal is None:
        _, _, Vt = np.linalg.svd(points - centroid)
        normal = Vt[-1]
        u, v = Vt[0], Vt[1]
        plane_origin = centroid
    else:
        normal = np.asarray(normal, dtype=np.float64).reshape(3)
        normal = normal / np.linalg.norm(normal)
        ref = np.array([0.0, 0.0, 1.0])
        if abs(normal @ ref) > 0.99:
            ref = np.array([0.0, 1.0, 0.0])
        u = ref - (ref @ normal) * normal
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        v /= np.linalg.norm(v)
        signed_dists = (points - centroid) @ normal
        plane_origin = centroid + signed_dists.max() * normal

    proj = points - np.outer((points - plane_origin) @ normal, normal)
    coords = np.column_stack([(proj - plane_origin) @ u, (proj - plane_origin) @ v])

    global_to_local = {gid: lid for lid, gid in enumerate(vertex_ids)}
    N = len(vertex_ids)
    neighbors = [[] for _ in range(N)]
    for a, b in component_edges:
        la, lb = global_to_local[a], global_to_local[b]
        neighbors[la].append(lb)
        neighbors[lb].append(la)

    c = coords.copy()
    for _ in range(n_iter):
        c_new = c.copy()
        for i in range(N):
            nbrs = neighbors[i]
            if nbrs:
                c_new[i] = (1.0 - lam) * c[i] + lam * np.mean(c[nbrs], axis=0)
        c = c_new

    mean_r_before = np.mean(np.linalg.norm(coords, axis=1))
    mean_r_after = np.mean(np.linalg.norm(c, axis=1))
    if mean_r_after > 0:
        c *= mean_r_before / mean_r_after

    return plane_origin + c[:, 0:1] * u + c[:, 1:2] * v


def planarize_openings(mesh, r_forward_fusion_info_filename="forward_fusion_info.npz",
                       flaw_opening_min_size=10, n_iter=10, lam=0.5, smooth=True):
    """Nudge every open boundary ring onto a flat plane (+ optional smoothing)."""
    import pyvista as pv
    import networkx as nx
    mesh = pv.read(mesh) if isinstance(mesh, str) else mesh
    points = np.array(mesh.points, dtype=float)

    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    all_edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges_sorted = np.sort(all_edges, axis=1)
    unique_edges, counts = np.unique(edges_sorted, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]

    G = nx.Graph()
    G.add_edges_from(boundary_edges.tolist())
    components = [comp for comp in nx.connected_components(G)
                 if len(comp) >= flaw_opening_min_size]

    if r_forward_fusion_info_filename is not None:
        info = np.load(r_forward_fusion_info_filename, allow_pickle=True)
        cpcd_glo_tangent = info['cpcd_glo_tangent']
        saved_centroids = info['opening_centroids']
        branch_normals = [cpcd_glo_tangent[i][0] for i in range(len(saved_centroids))]
        opening_centroids = np.array([
            points[np.array(sorted(comp))].mean(axis=0) for comp in components
        ])
        opening_normals = []
        for oc in opening_centroids:
            dists = np.linalg.norm(saved_centroids - oc, axis=1)
            opening_normals.append(branch_normals[int(np.argmin(dists))])
    else:
        opening_normals = []
        for comp in components:
            ring_pts = points[np.array(sorted(comp))]
            centered = ring_pts - ring_pts.mean(axis=0)
            _, _, Vt = np.linalg.svd(centered, full_matrices=False)
            opening_normals.append(Vt[-1])

    for idx, comp in enumerate(components):
        vertex_ids = np.array(sorted(comp))
        comp_edges = np.array(list(G.subgraph(comp).edges()))
        normal = opening_normals[idx] if opening_normals is not None else None
        new_pos = flatten_and_smooth_opening(points, vertex_ids, comp_edges,
                                             normal=normal,
                                             n_iter=n_iter if smooth else 0, lam=lam)
        points[vertex_ids] = new_pos

    return pv.PolyData(points, mesh.faces.copy())


def smooth_near_openings(mesh, r_forward_fusion_info_filename="ghd_forward_fusion_info.npz",
                         n_rings=3, n_iter=10, lam=0.5):
    """Laplacian-smooth the merged mesh in a k-ring neighbourhood of each opening."""
    import pyvista as pv
    import networkx as nx
    mesh = pv.read(mesh) if isinstance(mesh, str) else mesh
    points = np.array(mesh.points, dtype=float)

    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    all_edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges_sorted = np.sort(all_edges, axis=1)
    unique_edges = np.unique(edges_sorted, axis=0)
    G = nx.Graph()
    G.add_edges_from(unique_edges.tolist())

    info = np.load(r_forward_fusion_info_filename, allow_pickle=True)
    opening_vids = info['opening_vertex_ids']

    for i in range(len(opening_vids)):
        seed_nodes = set(opening_vids[i].tolist())
        region = set(seed_nodes)
        frontier = set(seed_nodes)
        for _ in range(n_rings):
            next_frontier = set()
            for node in frontier:
                for nbr in G.neighbors(node):
                    if nbr not in region:
                        next_frontier.add(nbr)
            region |= next_frontier
            frontier = next_frontier

        anchors = frontier
        free_nodes = list(region - anchors)
        region_list = list(region)
        subgraph = G.subgraph(region_list)
        neighbors = {n: list(subgraph.neighbors(n)) for n in free_nodes}

        for _ in range(n_iter):
            new_pts = points.copy()
            for n in free_nodes:
                nbrs = neighbors[n]
                if nbrs:
                    new_pts[n] = (1.0 - lam) * points[n] + lam * points[nbrs].mean(axis=0)
            points = new_pts

    return pv.PolyData(points, mesh.faces.copy())
