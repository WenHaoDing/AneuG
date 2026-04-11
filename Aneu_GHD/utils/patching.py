import os
import numpy as np
import pyvista as pv
import networkx as nx


def flatten_and_smooth_opening(mesh_points, vertex_ids, component_edges,
                                normal=None, n_iter=10, lam=0.5):
    """
    Flatten a jagged boundary ring onto a plane, then regularise node spacing
    with Laplacian smoothing in that plane.

    The plane normal is either supplied (centerline tangent) or estimated via
    PCA.  When a normal is supplied the plane origin is shifted outward to the
    farthest node so that projected nodes are never pulled inward.

    Parameters
    ----------
    mesh_points : np.ndarray, shape (M, 3)
        Full mesh vertex positions.
    vertex_ids : np.ndarray, shape (N,)
        Global vertex indices of this opening's boundary ring.
    component_edges : np.ndarray, shape (E, 2)
        Boundary edges (global vertex indices) belonging to this ring.
    normal : np.ndarray, shape (3,) or None
        Prescribed plane normal (unit vector pointing AWAY from the vessel,
        i.e. cpcd_glo_tangent[i][0] from forward_mesh_fusion_v2).
        If None, the PCA best-fit normal is used and no outward shift is applied.
    n_iter : int
        Number of Laplacian smoothing iterations.
    lam : float
        Smoothing weight in [0, 1].

    Returns
    -------
    new_positions : np.ndarray, shape (N, 3)
        Updated 3-D positions for the nodes in vertex_ids order.
    """
    points   = mesh_points[vertex_ids]   # (N, 3)
    centroid = points.mean(axis=0)

    # --- Step 1: Determine plane normal and in-plane axes ---
    if normal is None:
        # PCA fallback: no outward shift
        _, _, Vt = np.linalg.svd(points - centroid)
        normal   = Vt[-1]
        u, v     = Vt[0], Vt[1]
        plane_origin = centroid
    else:
        normal = normal / np.linalg.norm(normal)
        # Gram-Schmidt: build u, v perpendicular to normal
        ref = np.array([0.0, 0.0, 1.0])
        if abs(normal @ ref) > 0.99:
            ref = np.array([0.0, 1.0, 0.0])
        u = ref - (ref @ normal) * normal
        u /= np.linalg.norm(u)
        v  = np.cross(normal, u)
        v /= np.linalg.norm(v)
        # Outward shift: normal faces away from the vessel, so outward nodes
        # have the largest positive signed distance.  Place the plane at the
        # farthest outward node so projected nodes are never pulled inward.
        signed_dists = (points - centroid) @ normal   # (N,)
        plane_origin = centroid + signed_dists.max() * normal

    # --- Step 2: Project onto plane, express in 2-D local coords ---
    proj   = points - np.outer((points - plane_origin) @ normal, normal)
    coords = np.column_stack([(proj - plane_origin) @ u,
                               (proj - plane_origin) @ v])   # (N, 2)

    # --- Step 3: Build neighbour lists in local index space ---
    global_to_local = {gid: lid for lid, gid in enumerate(vertex_ids)}
    N         = len(vertex_ids)
    neighbors = [[] for _ in range(N)]
    for a, b in component_edges:
        la, lb = global_to_local[a], global_to_local[b]
        neighbors[la].append(lb)
        neighbors[lb].append(la)

    # --- Step 4: Laplacian smoothing on 2-D coords ---
    c = coords.copy()
    for _ in range(n_iter):
        c_new = c.copy()
        for i in range(N):
            nbrs = neighbors[i]
            if nbrs:
                c_new[i] = (1.0 - lam) * c[i] + lam * np.mean(c[nbrs], axis=0)
        c = c_new

    # Rescale to preserve mean radius (undo Laplacian shrinkage)
    mean_r_before = np.mean(np.linalg.norm(coords, axis=1))
    mean_r_after  = np.mean(np.linalg.norm(c,      axis=1))
    c *= mean_r_before / mean_r_after

    # --- Step 5: Lift back to 3-D ---
    new_positions = plane_origin + c[:, 0:1] * u + c[:, 1:2] * v
    return new_positions


def planarize_openings(mesh, post_dir=None,
                       r_forward_fusion_info_filename="forward_fusion_info.npz",
                       flaw_opening_min_size=10, n_iter=10, lam=0.5):
    """
    Detect every open boundary ring of a surface mesh and nudge its nodes
    onto a flat plane, then regularise spacing with Laplacian smoothing.
    The rest of the mesh is untouched.

    When a forward-fusion info file is provided the plane normal for each
    opening is taken from cpcd_glo_tangent[i][0] — the outward-facing tangent
    already computed and stored by forward_mesh_fusion_v2.  The plane is
    shifted outward to the farthest node so projected nodes are never pulled
    inward.  Without the info file the normal falls back to PCA.

    Parameters
    ----------
    mesh : str or pyvista.PolyData
        Input mesh.
    post_dir : str or None
        Case directory containing the forward_fusion_info file.
        When None, the plane normal falls back to PCA.
    r_forward_fusion_info_filename : str
        Info filename relative to post_dir (written by forward_mesh_fusion_v2).
    flaw_opening_min_size : int
        Rings with fewer nodes than this are ignored.
    n_iter : int
        Laplacian smoothing iterations per ring.
    lam : float
        Smoothing weight per iteration (0 = no change, 1 = full neighbour mean).

    Returns
    -------
    mesh_out : pyvista.PolyData
        Copy of the input mesh with boundary-ring nodes repositioned.
    """
    mesh   = pv.read(mesh) if isinstance(mesh, str) else mesh
    points = np.array(mesh.points, dtype=float)

    # --- Boundary edges: shared by exactly one face ---
    faces          = mesh.faces.reshape(-1, 4)[:, 1:]
    all_edges      = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges_sorted   = np.sort(all_edges, axis=1)
    unique_edges, counts = np.unique(edges_sorted, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]

    # One connected component per opening
    G = nx.Graph()
    G.add_edges_from(boundary_edges.tolist())
    components = [comp for comp in nx.connected_components(G)
                  if len(comp) >= flaw_opening_min_size]

    # --- Optionally load outward tangents from the forward-fusion info file ---
    opening_normals = None
    if post_dir is not None:
        info              = np.load(os.path.join(post_dir, r_forward_fusion_info_filename),
                                    allow_pickle=True)
        cpcd_glo_tangent  = info['cpcd_glo_tangent']    # object array, one tang array per branch
        saved_centroids   = info['opening_centroids']   # (B, 3) centroid per matched opening

        # cpcd_glo_tangent[i][0] is the outward-facing tangent at the opening
        # of branch i (already flipped and oriented by forward_mesh_fusion_v2).
        branch_normals = [cpcd_glo_tangent[i][0] for i in range(len(saved_centroids))]

        # Match each detected opening to the nearest saved centroid
        opening_centroids = np.array([
            points[np.array(sorted(comp))].mean(axis=0) for comp in components
        ])
        opening_normals = []
        for oc in opening_centroids:
            dists = np.linalg.norm(saved_centroids - oc, axis=1)
            opening_normals.append(branch_normals[int(np.argmin(dists))])

    # --- Apply flatten + smooth per opening ---
    for idx, comp in enumerate(components):
        vertex_ids = np.array(sorted(comp))
        comp_edges = np.array(list(G.subgraph(comp).edges()))
        normal     = opening_normals[idx] if opening_normals is not None else None
        new_pos    = flatten_and_smooth_opening(points, vertex_ids, comp_edges,
                                                normal=normal, n_iter=n_iter, lam=lam)
        points[vertex_ids] = new_pos

    return pv.PolyData(points, mesh.faces.copy())


def smooth_near_openings(mesh, post_dir,
                         r_forward_fusion_info_filename="ghd_forward_fusion_info.npz",
                         n_rings=3, n_iter=10, lam=0.5):
    """
    Laplacian-smooth the merged mesh in a k-ring neighbourhood around each
    opening boundary.  Only nodes within n_rings hops of the opening ring are
    moved; nodes at the outer boundary of that region stay fixed as anchors.

    Parameters
    ----------
    mesh : str or pyvista.PolyData
        Merged mesh (output of forward_mesh_fusion_v2).
    post_dir : str
        Case directory containing the forward_fusion_info file.
    r_forward_fusion_info_filename : str
        Info filename relative to post_dir.
    n_rings : int
        Number of edge-hops from the opening ring to include in the smooth region.
    n_iter : int
        Laplacian smoothing iterations.
    lam : float
        Smoothing weight per iteration (0 = no change, 1 = full neighbour mean).

    Returns
    -------
    mesh_out : pyvista.PolyData
        Copy of the mesh with the near-opening regions smoothed.
    """
    mesh   = pv.read(mesh) if isinstance(mesh, str) else mesh
    points = np.array(mesh.points, dtype=float)

    # --- Full mesh edge graph ---
    faces        = mesh.faces.reshape(-1, 4)[:, 1:]
    all_edges    = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges_sorted = np.sort(all_edges, axis=1)
    unique_edges = np.unique(edges_sorted, axis=0)
    G = nx.Graph()
    G.add_edges_from(unique_edges.tolist())

    # --- Load opening vertex IDs (still valid in merged mesh) ---
    info         = np.load(os.path.join(post_dir, r_forward_fusion_info_filename),
                           allow_pickle=True)
    opening_vids = info['opening_vertex_ids']   # (B,) object array of index arrays

    for i in range(len(opening_vids)):
        seed_nodes = set(opening_vids[i].tolist())

        # BFS to collect nodes within n_rings hops
        region  = set(seed_nodes)
        frontier = set(seed_nodes)
        for _ in range(n_rings):
            next_frontier = set()
            for node in frontier:
                for nbr in G.neighbors(node):
                    if nbr not in region:
                        next_frontier.add(nbr)
            region   |= next_frontier
            frontier  = next_frontier

        # Nodes at the outermost ring are anchors (not moved)
        anchors    = frontier
        free_nodes = list(region - anchors)

        # Build local neighbour lists (only within region, for speed)
        region_list = list(region)
        subgraph    = G.subgraph(region_list)
        neighbors   = {n: list(subgraph.neighbors(n)) for n in free_nodes}

        # Laplacian smoothing — anchors provide fixed boundary conditions
        for _ in range(n_iter):
            new_pts = points.copy()
            for n in free_nodes:
                nbrs = neighbors[n]
                if nbrs:
                    new_pts[n] = (1.0 - lam) * points[n] + lam * points[nbrs].mean(axis=0)
            points = new_pts

    return pv.PolyData(points, mesh.faces.copy())
