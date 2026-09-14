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

from pathlib import Path

import numpy as np
import torch


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
            # NOT IMPLEMENTED, kept here because this is where it would go:
            # TAPERING THE SWEEP. radius_transit below holds the cross-section
            # radius at every station along the tube, and its SCALE is constant,
            # so the tube's walls run parallel for its whole length. The dome
            # arrives at the cut still narrowing -- measured over 144 branches,
            # radius falls 1.449 -> 1.239 over the last unit, a gradient of
            # -0.211 per unit, and 90% of branches narrow rather than widen. So
            # width is continuous across the join but its SLOPE is not, which
            # reads as a corner in the silhouette.
            #
            # The fix is one multiply: scale radius_transit per station by
            # roughly (1 - rate * arc_distance), with `rate` fitted from the
            # dome's own last few rings so each branch tapers at its own rate,
            # and a floor so a long branch cannot shrink to nothing.
            #
            # Deliberately deferred: there is no ground truth past the cut (the
            # centerlines were clipped there), so this extrapolates rather than
            # measures. Cosmetic for the current metrics, but it would change
            # flow resistance if these meshes go to CFD.
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

    # Baseline is the ORIGINAL 3-D ring radius, not the projected one.
    #
    # `coords` is measured after projecting onto the plane, so using it here
    # undid only the Laplacian contraction and silently kept the loss from the
    # projection itself: a ring tilted relative to the plane normal foreshortens
    # when projected along it. Measured over 74 rims that cost a mean radius
    # ratio of 0.961, with 20% of rims losing more than 5% and the worst 0.675 --
    # which is what produced the sudden narrowing where the swept tube meets the
    # dome. Referencing the true 3-D radius makes this one rescale undo both.
    mean_r_before = np.mean(np.linalg.norm(points - centroid, axis=1))
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


# ── merged-mesh builder ─────────────────────────────────────────────────────

class MergedMeshBuilder:
    """Build a merged dome + branch mesh, with the sweep's known defects handled.

    The free-function path above (and MultiCanonicalGHDReconstruct.reconstruct_
    fused_mesh, which orchestrates it) works, but it hands the caller one big
    call with twenty keyword arguments and no way to see what went wrong. This
    class keeps the same geometry and adds three things.

    ORDER. Planarize each opening BEFORE sweeping. The rim left by deleting a cap
    is not flat, and lofting a ring onto a non-planar rim puts a kink in the
    first tube segment. Planarizing first means the first ring and the rim agree
    by construction.

    RING COLLISION. A tube of radius r swept along a curve of curvature k folds
    into itself on the inside of the turn once k*r approaches 1: consecutive
    rings cross, and the merged surface self-intersects. The centerline comes
    from a generative model with no such constraint, so this has to be enforced
    here. `max_kr` caps the product; violating stations are smoothed locally
    until they comply, which bends the centerline rather than thinning the tube
    -- the radius is measured from the real opening and should not be invented.
    The branch's first point and initial tangent are held fixed so the tube still
    leaves the opening along the direction the sensor measured.

    SEAM. Smoothing over a fixed ring count leaves its own edge, a visible circle
    where smoothing stopped. Here the strength decays with ring distance from the
    join, so the smoothed region blends into the untouched surface.

    Every build returns diagnostics alongside the mesh: how many stations were
    curvature-limited, whether any rings still fold, and the open-boundary
    centroids. Judge the output by those rather than by the call not raising.
    """

    def __init__(self, multi_recon,
                 max_kr=0.45, warn_kr=0.85, curvature_iters=200, curvature_lam=0.35,
                 anchor_span=6,
                 seam_rings=6, seam_iters=12, seam_lam=0.5,
                 planarize_iters=10, planarize_lam=0.5, check_self_intersection=True,
                 inlet_max_length=7.5, outlet_max_length=3.0,
                 inlet_extrusion=5.0, outlet_extrusion=2.5,
                 inlet_min_length=5.0, outlet_min_length=None,
                 extrude_length=3.0, min_branch_arc=3.0, init_step=1, ds_r=3,
                 min_torsion=True, flaw_opening_min_size=10, max_cl_length=None):
        self.multi_recon = multi_recon
        # max_kr is the TARGET the limiter bends toward, a safety margin.
        # warn_kr is where the mesh is actually in danger: rings only cross as
        # the ratio approaches 1. Warning at the target flagged cases that were
        # visibly fine (0.57 and 0.63, both inspected, both clean, neither with
        # a single folded ring step), so the two numbers are kept apart.
        self.max_kr = max_kr
        self.warn_kr = warn_kr
        self.curvature_iters = curvature_iters
        self.curvature_lam = curvature_lam
        self.anchor_span = anchor_span
        self.seam_rings = seam_rings
        self.seam_iters = seam_iters
        self.seam_lam = seam_lam
        self.planarize_iters = planarize_iters
        self.planarize_lam = planarize_lam
        self.check_self_intersection = check_self_intersection
        # Branch 0 is the INLET, every other branch is an outlet: a bifurcated
        # shape has two outlets and a sidewall one. Lengths are in mesh units,
        # which are millimetres here, and are deliberately short: the meshes go
        # to CFD, where length costs solver time.
        self.inlet_max_length = inlet_max_length
        self.outlet_max_length = outlet_max_length
        self.inlet_extrusion = inlet_extrusion
        self.outlet_extrusion = outlet_extrusion
        # A generated centerline shorter than this is topped up by extrusion, so the
        # branch is never shorter than min_length + extrusion. None disables it.
        self.inlet_min_length = inlet_min_length
        self.outlet_min_length = outlet_min_length
        self.extrude_length = extrude_length
        self.min_branch_arc = min_branch_arc
        self.init_step = init_step
        self.ds_r = ds_r
        self.min_torsion = min_torsion
        self.flaw_opening_min_size = flaw_opening_min_size
        self.max_cl_length = max_cl_length

    # ---- centerline conditioning -------------------------------------------

    @staticmethod
    def discrete_curvature(cl):
        """Menger curvature per interior station: 1 / radius of the circle
        through each consecutive triple.

        Turn-angle-over-arc-step was tried first and is unusable here. Smoothing
        moves points ALONG the curve as well as across it, so the step shrinks
        where the curve flattens, and dividing by it makes the estimate grow as
        the curve gets straighter -- measured, the reported maximum rose from
        1.67 to 8.65 as smoothing proceeded. Menger curvature depends only on
        the triangle the three points make, so it is immune to that.
        """
        a = np.linalg.norm(cl[1:-1] - cl[:-2], axis=1)
        b = np.linalg.norm(cl[2:] - cl[1:-1], axis=1)
        c = np.linalg.norm(cl[2:] - cl[:-2], axis=1)
        cross = np.cross(cl[1:-1] - cl[:-2], cl[2:] - cl[1:-1])
        area = 0.5 * np.linalg.norm(cross, axis=1)
        k = np.zeros(len(cl))
        k[1:-1] = 4.0 * area / np.maximum(a * b * c, 1e-12)
        return k

    def preprocess_centerline(self, cl, branch_index, step):
        """Trim a generated centerline to length, then extend it straight.

        Two separate operations, in this order.

        TRIM. Only the first stretch of each branch is wanted: 10 mm for the
        inlet (branch 0), 3 mm for each outlet -- kept short on purpose, since
        every extra millimetre is solver time downstream. The centerlines run
        much further, and the far end is the least trustworthy part of them --
        it is the end of an autoregressive rollout, with nothing downstream to
        constrain it.

        EXTEND. A straight run is then appended along the terminal direction, 5
        mm for the inlet and 2.5 mm for an outlet, so the mesh ends in a straight
        section. A solver wants its inlet and outlet faces perpendicular to the
        flow with developed profiles, which a boundary cut across a bend does not
        give. The direction is averaged over the last few points rather than
        taken from the final pair, which would inherit that pair's noise.

        Returns (centerline, trimmed_from, extended_by).
        """
        cl = np.asarray(cl, dtype=float)
        is_inlet = (branch_index == 0)
        max_len = self.inlet_max_length if is_inlet else self.outlet_max_length
        ext_len = self.inlet_extrusion if is_inlet else self.outlet_extrusion
        min_len = self.inlet_min_length if is_inlet else self.outlet_min_length

        seg = np.linalg.norm(np.diff(cl, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(seg)])
        full = float(arc[-1])
        if max_len is not None and full > max_len:
            cut = int(np.searchsorted(arc, max_len, side="right"))
            cl = cl[:max(cut, 3)]
        trimmed_from = full

        # TOP UP A SHORT BRANCH. The straight run is meant to follow a stretch of
        # real centerline, so if the model generated less than min_len, the missing
        # part is added to the extrusion: a 2 mm inlet gets 5 + 3 = 8 mm of straight
        # run rather than 5, and ends at the same 10 mm a 5 mm inlet would.
        if min_len:
            kept = float(np.linalg.norm(np.diff(cl, axis=0), axis=1).sum()) if len(cl) > 1 else 0.0
            ext_len = (ext_len or 0.0) + max(0.0, min_len - kept)

        extended = 0.0
        if ext_len and ext_len > 0 and len(cl) >= 2:
            k = min(5, len(cl) - 1)
            d = cl[-1] - cl[-1 - k]
            n = np.linalg.norm(d)
            if n > 1e-9:
                d = d / n
                n_steps = max(2, int(np.ceil(ext_len / max(step, 1e-6))))
                tail = cl[-1] + d * np.linspace(step, ext_len, n_steps)[:, None]
                cl = np.vstack([cl, tail])
                extended = float(ext_len)
        return cl, trimmed_from, extended

    def limit_curvature(self, cl, radius):
        """Bend the centerline until curvature * radius stays under `max_kr`.

        Local Laplacian smoothing weighted by how far each station exceeds the
        cap, so straight stretches are untouched. The first two points are
        anchored: the first is the opening centroid and the second sets the
        direction the tube leaves by, which is the quantity the sensor measured
        and the branch model was penalised to respect.
        """
        cl = np.array(cl, dtype=float)
        if len(cl) < 5 or radius <= 0:
            return cl, 0
        n_violating = int((self.discrete_curvature(cl) * radius > self.max_kr).sum())
        if n_violating == 0:
            return cl, 0
        for _ in range(self.curvature_iters):
            k = self.discrete_curvature(cl)
            excess = np.clip(k * radius / self.max_kr - 1.0, 0.0, None)
            if not np.any(excess > 0):
                break
            # spread the weight to each violating station's neighbours, so a
            # single sharp corner is relieved by a short arc rather than by
            # pulling one point and leaving two new corners beside it
            excess = np.maximum.reduce([excess,
                                        np.roll(excess, 1), np.roll(excess, -1)])
            w = self.curvature_lam * np.clip(excess, 0.0, 1.0)
            # Anchor the opening end with a RAMP, not a hard stop. Freezing the
            # first two points while smoothing their neighbours builds a corner
            # against the frozen pair: measured, the peak curvature after
            # smoothing sat at station 1 and was the single reason the loop
            # diverged. The ramp lets the anchor region relax with the rest
            # while still pinning the very first point and its direction.
            ramp = np.clip(np.arange(len(cl)) / max(self.anchor_span, 1), 0.0, 1.0)
            w *= ramp
            w[0] = 0.0
            w[-1] = 0.0
            mid = np.zeros_like(cl)
            mid[1:-1] = 0.5 * (cl[2:] + cl[:-2])
            cl[1:-1] = (1 - w[1:-1, None]) * cl[1:-1] + w[1:-1, None] * mid[1:-1]
            # Re-space after every pass. Laplacian smoothing slides points along
            # the curve as well as across it, and a bunched triple reads as high
            # curvature however curvature is estimated, so without this the
            # measured maximum RISES as the curve is straightened (1.7 -> 8.2
            # over 600 passes, measured). Re-spacing keeps the estimate honest
            # and is what makes the loop converge at all.
            cl = self.respace(cl)
        return cl, n_violating

    @staticmethod
    def respace(cl):
        """Re-sample a polyline to uniform arc-length spacing, keeping its ends."""
        seg = np.linalg.norm(np.diff(cl, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(seg)])
        if arc[-1] <= 1e-12:
            return cl
        target = np.linspace(0.0, arc[-1], len(cl))
        return np.stack([np.interp(target, arc, cl[:, i]) for i in range(3)], axis=1)

    @staticmethod
    def ring_fold_fraction(tube_verts, tangents):
        """Fraction of ring steps that move backwards along the tube.

        A swept tube is sound while every vertex advances: (P[i+1] - P[i]) . t[i]
        must stay positive. Where it does not, that part of the ring has crossed
        the one before it and the surface self-intersects. This is the direct
        test, rather than inferring it from curvature.
        """
        if tube_verts.shape[0] < 2:
            return 0.0
        step = tube_verts[1:] - tube_verts[:-1]                    # [dpi-1, N, 3]
        adv = np.einsum('dnc,dc->dn', step, tangents[:-1])
        return float((adv <= 0).mean())

    # ---- seam smoothing ----------------------------------------------------

    def smooth_seam(self, mesh, seam_vertex_ids):
        """Laplacian smoothing whose strength decays with distance from the seam.

        A uniform k-ring smooth replaces one visible edge (the join) with
        another (the boundary of the smoothed patch). Grading the strength by
        ring index removes the second one.
        """
        import pyvista as pv
        pts = np.array(mesh.points, dtype=float)
        faces = mesh.faces.reshape(-1, 4)[:, 1:]
        e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        e = np.unique(np.sort(e, axis=1), axis=0)
        n = len(pts)
        adj = [[] for _ in range(n)]
        for a, b in e:
            adj[a].append(b); adj[b].append(a)

        # ring index by breadth-first search from the seam
        ring = np.full(n, -1, dtype=int)
        frontier = [int(v) for v in seam_vertex_ids if 0 <= int(v) < n]
        for v in frontier:
            ring[v] = 0
        for r in range(1, self.seam_rings + 1):
            nxt = []
            for v in frontier:
                for w in adj[v]:
                    if ring[w] < 0:
                        ring[w] = r; nxt.append(w)
            frontier = nxt
            if not frontier:
                break
        touched = np.where(ring >= 0)[0]
        if len(touched) == 0:
            return mesh
        # full strength at the join, zero at the outer ring: a cosine ramp, so
        # the weight and its slope both reach zero at the boundary
        w = np.zeros(n)
        w[touched] = self.seam_lam * 0.5 * (
            1.0 + np.cos(np.pi * np.clip(ring[touched] / max(self.seam_rings, 1), 0, 1)))
        for _ in range(self.seam_iters):
            new = pts.copy()
            for v in touched:
                nb = adj[v]
                if nb:
                    new[v] = (1 - w[v]) * pts[v] + w[v] * pts[nb].mean(axis=0)
            pts = new
        return pv.PolyData(pts, mesh.faces.copy())

    # ---- hole repair -------------------------------------------------------

    @staticmethod
    def ordered_boundary_loops(faces, n_points):
        """Every open boundary as an ordered cycle, by walking EDGES not vertices.

        A vertex-to-vertex walk breaks wherever a vertex carries more than two
        boundary edges, which happens at the pinches left by cap deletion: it
        returned fragments whose consecutive pairs were not even edges, and a fan
        built on one of those seals nothing. Consuming each boundary edge exactly
        once cannot produce that.
        """
        e = np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
        key = e[:, 0].astype(np.int64) * n_points + e[:, 1]
        uniq, counts = np.unique(key, return_counts=True)
        border = uniq[counts == 1]
        if len(border) == 0:
            return []
        be = np.stack([border // n_points, border % n_points], 1).astype(int)
        inc = {}
        for ei, (a, b) in enumerate(be):
            inc.setdefault(int(a), []).append(ei)
            inc.setdefault(int(b), []).append(ei)
        used = np.zeros(len(be), bool)
        loops = []
        for e0 in range(len(be)):
            if used[e0]:
                continue
            used[e0] = True
            start, cur = int(be[e0][0]), int(be[e0][1])
            loop = [start, cur]
            while cur != start:
                nxt = [k for k in inc[cur] if not used[k]]
                if not nxt:
                    break
                used[nxt[0]] = True
                a, b = int(be[nxt[0]][0]), int(be[nxt[0]][1])
                cur = b if a == cur else a
                loop.append(cur)
            if loop[-1] == start:
                loop.pop()
            loops.append(np.array(loop, dtype=int))
        return loops

    def close_small_holes(self, mesh, keep_n):
        """Keep the `keep_n` largest open boundaries; seal everything else.

        A bifurcated shape has exactly three openings and a sidewall two, so any
        further boundary is a defect: small tears left where the cap was deleted
        or where smoothing pulled the surface apart. Measured on the first
        cohort, one sidewall case carried holes of radius 0.07 to 0.13 beside
        real tube ends of 1.44 and 1.83, more than ten times smaller, so the
        split is never ambiguous.

        The triangulation is VTK's hole filler rather than a hand-rolled fan.
        The threshold is placed between the smallest boundary we keep and the
        largest we close, which is why the loops are measured first.
        """
        pts = np.asarray(mesh.points, dtype=float)
        faces = mesh.faces.reshape(-1, 4)[:, 1:]
        loops = self.ordered_boundary_loops(faces, len(pts))
        if len(loops) <= keep_n:
            return mesh, 0, []
        size = np.array([2.0 * np.linalg.norm(pts[l] - pts[l].mean(0), axis=1).mean()
                         if len(l) else 0.0 for l in loops])
        order = np.argsort(-size)
        keep_min, close_max = size[order[keep_n - 1]], size[order[keep_n]]
        if not (close_max < keep_min):
            return mesh, 0, []          # not separable; leave it alone and report

        # Fan to each hole's own centroid, rather than VTK's vtkFillHolesFilter:
        # on this geometry that filter declines the job, adding one triangle and
        # leaving all five boundaries intact, even though the mesh is clean
        # (no non-manifold edges, no degenerate faces, no duplicate vertices).
        import pyvista as pv
        directed = set()
        for f in faces:
            directed.update({(int(f[0]), int(f[1])), (int(f[1]), int(f[2])), (int(f[2]), int(f[0]))})
        new_pts, new_faces = list(pts), list(faces)
        for li in order[keep_n:]:
            loop = loops[li]
            if len(loop) < 3:
                continue                 # a two-vertex boundary is a slit, not a hole
            ci = len(new_pts)
            new_pts.append(pts[loop].mean(0))
            for k in range(len(loop)):
                a, b = int(loop[k]), int(loop[(k + 1) % len(loop)])
                # the one face owning this edge fixes the winding; the patch
                # triangle must run the other way or its normal points inward
                new_faces.append([b, a, ci] if (a, b) in directed else [a, b, ci])
        nf = np.asarray(new_faces, dtype=int)
        filled = pv.PolyData(np.asarray(new_pts),
                             np.hstack([np.full((len(nf), 1), 3), nf]).ravel())
        return filled, int(len(loops) - keep_n), [round(float(x), 4) for x in size[order[keep_n:]]]

    # ---- self-intersection --------------------------------------------------

    @staticmethod
    def self_intersections(mesh, max_pairs=4_000_000):
        """Face pairs that actually cross. Returns (n_pairs, checked_all).

        Neither VTK, libigl's Python bindings nor trimesh exposes a
        self-intersection test in this environment, so this is the separating
        axis theorem on triangle pairs, with the pairs pruned by a KD-tree over
        face centroids. Thirteen axes decide it: the two face normals and the
        nine edge-edge cross products, plus a guard for the degenerate ones.

        Faces sharing a vertex are excluded, since they are supposed to touch.
        `checked_all` is False when the candidate set was too large and had to be
        truncated, so a zero result can be read as "none found" rather than
        misread as "none exist".
        """
        from scipy.spatial import cKDTree
        pts = np.asarray(mesh.points, dtype=float)
        faces = mesh.faces.reshape(-1, 4)[:, 1:]
        tri = pts[faces]                                     # [F, 3, 3]
        cen = tri.mean(1)
        rad = np.linalg.norm(tri - cen[:, None], axis=2).max(1)
        pairs = np.array(sorted(cKDTree(cen).query_pairs(r=2.0 * rad.max())), dtype=int)
        if len(pairs) == 0:
            return 0, True
        checked_all = len(pairs) <= max_pairs
        pairs = pairs[:max_pairs]
        # drop pairs that share a vertex: those meet by construction
        fa, fb = faces[pairs[:, 0]], faces[pairs[:, 1]]
        shares = (fa[:, :, None] == fb[:, None, :]).any(axis=(1, 2))
        pairs = pairs[~shares]
        if len(pairs) == 0:
            return 0, checked_all
        # cheap reject: centroid distance beyond the two circumradii
        d = np.linalg.norm(cen[pairs[:, 0]] - cen[pairs[:, 1]], axis=1)
        pairs = pairs[d <= rad[pairs[:, 0]] + rad[pairs[:, 1]]]
        if len(pairs) == 0:
            return 0, checked_all

        P, Q = tri[pairs[:, 0]], tri[pairs[:, 1]]            # [M, 3, 3]
        ep = np.stack([P[:, 1] - P[:, 0], P[:, 2] - P[:, 1], P[:, 0] - P[:, 2]], 1)
        eq = np.stack([Q[:, 1] - Q[:, 0], Q[:, 2] - Q[:, 1], Q[:, 0] - Q[:, 2]], 1)
        axes = [np.cross(ep[:, 0], ep[:, 1]), np.cross(eq[:, 0], eq[:, 1])]
        for i in range(3):
            for j in range(3):
                axes.append(np.cross(ep[:, i], eq[:, j]))
        alive = np.ones(len(pairs), bool)
        for ax in axes:
            n = np.linalg.norm(ax, axis=1)
            ok = n > 1e-12                                   # a degenerate axis separates nothing
            a = np.where(ok[:, None], ax / np.maximum(n, 1e-12)[:, None], 0.0)
            pp = np.einsum('mvc,mc->mv', P, a)
            qq = np.einsum('mvc,mc->mv', Q, a)
            sep = ok & ((pp.min(1) > qq.max(1) + 1e-12) | (qq.min(1) > pp.max(1) + 1e-12))
            alive &= ~sep
            if not alive.any():
                break
        return int(alive.sum()), checked_all

    # ---- open boundaries ---------------------------------------------------

    @staticmethod
    def boundary_loops(mesh):
        """Centroid, normal and size of every open boundary of the final mesh.

        These are the tube ends -- the inlet and outlets a solver would need --
        found on the finished mesh rather than assumed from the inputs, so they
        reflect whatever smoothing and cleanup actually did.
        """
        faces = mesh.faces.reshape(-1, 4)[:, 1:]
        e = np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
        key = e[:, 0].astype(np.int64) * (faces.max() + 1) + e[:, 1]
        uniq, counts = np.unique(key, return_counts=True)
        border = uniq[counts == 1]
        if len(border) == 0:
            return []
        be = np.stack([border // (faces.max() + 1), border % (faces.max() + 1)], 1)
        adj = {}
        for a, b in be:
            adj.setdefault(int(a), []).append(int(b))
            adj.setdefault(int(b), []).append(int(a))
        seen, loops = set(), []
        pts = np.asarray(mesh.points)
        for s in adj:
            if s in seen:
                continue
            comp, stack = [], [s]
            seen.add(s)
            while stack:
                v = stack.pop(); comp.append(v)
                for w in adj[v]:
                    if w not in seen:
                        seen.add(w); stack.append(w)
            ring = pts[comp]
            c = ring.mean(0)
            _, _, Vt = np.linalg.svd(ring - c, full_matrices=False)
            loops.append({"centroid": c, "normal": Vt[-1], "n_vertices": len(comp),
                          "radius": float(np.linalg.norm(ring - c, axis=1).mean())})
        return loops

    # ---- the build ---------------------------------------------------------

    def build(self, phi, aneurysm_type, branch_points, branch_mask=None,
              opening_indices=None, opening_normals=None, trimmed_faces=None,
              save_path=None):
        """Dome + branches, merged. Returns a dict: mesh, centroids, diagnostics.

        `opening_indices` must be ORDERED rim loops and `opening_normals` the
        sensor's tangents. Both come from SensorUncapper or from a pre-computed
        pool; the old canonical rings are not a fallback worth having, their
        plane normals sit a mean 52 degrees off the true tangent once GHD has
        warped the template.
        """
        import pyvista as pv

        atype = int(aneurysm_type)
        verts = self.multi_recon._reconstruct_verts_np(np.asarray(phi).reshape(-1, 3), atype)
        mesh_ctr = verts.mean(0)
        opening_indices = [np.asarray(i, dtype=int) for i in opening_indices]
        n_open = len(opening_indices)
        tri = (self.multi_recon._trimmed_faces(atype) if trimmed_faces is None
               else np.asarray(trimmed_faces))
        uncapped = pv.PolyData(verts, np.hstack([np.full((len(tri), 1), 3), tri]).ravel())
        step = avg_edge_length(uncapped)
        if branch_mask is not None:
            branch_mask = np.asarray(branch_mask).astype(bool).ravel()

        def outward(ring, centroid):
            _, _, Vt = np.linalg.svd(ring - centroid, full_matrices=False)
            nrm = Vt[-1]
            if np.dot(nrm, centroid - mesh_ctr) < 0:
                nrm = -nrm
            return nrm / np.linalg.norm(nrm)

        # 1. per-opening centerline and outward direction, before any reshaping
        specs = []
        for o in range(n_open):
            ring0 = verts[opening_indices[o]]
            c0 = ring0.mean(0)
            bp = None
            if branch_points is not None and o < len(branch_points) and branch_points[o] is not None:
                b = branch_points[o]
                bp = b.detach().cpu().numpy() if torch.is_tensor(b) else np.asarray(b, float)
            arc = (float(np.linalg.norm(np.diff(bp, axis=0), axis=1).sum())
                   if bp is not None and len(bp) >= 2 else 0.0)
            valid = (bp is not None and len(bp) >= 2
                     and (branch_mask is None or (o < len(branch_mask) and branch_mask[o]))
                     and arc >= self.min_branch_arc)
            nrm = (np.asarray(opening_normals[o], float) if opening_normals is not None
                   else outward(ring0, c0))
            specs.append({"valid": valid, "bp": bp, "normal0": nrm, "centroid0": c0,
                          "radius": float(np.linalg.norm(ring0 - c0, axis=1).mean())})

        # 2. PLANARIZE FIRST, so the first ring is lofted onto a flat rim
        import os, tempfile
        with tempfile.TemporaryDirectory() as td:
            info = os.path.join(td, "pre.npz")
            np.savez(info,
                     cpcd_glo_tangent=np.array([s["normal0"][None] for s in specs], dtype=object),
                     opening_centroids=np.array([s["centroid0"] for s in specs]),
                     allow_pickle=True)
            uncapped = planarize_openings(
                uncapped, r_forward_fusion_info_filename=info,
                flaw_opening_min_size=self.flaw_opening_min_size,
                n_iter=self.planarize_iters, lam=self.planarize_lam, smooth=True)
        verts = np.asarray(uncapped.points)

        # 3. centerlines: resample, trim, then cap curvature against the radius
        diag = {"curvature_limited": [], "ring_fold_fraction": [], "branch_valid": [],
                "arc_length": [], "max_kr": [], "exit_drift_deg": [],
                "generated_arc": [], "extended_by": []}
        opening_pcd, cpcd_glo, cpcd_tan, necks = [], [], [], []
        for o, s in enumerate(specs):
            ring = verts[opening_indices[o]]
            centroid = ring.mean(0)
            radius = float(np.linalg.norm(ring - centroid, axis=1).mean())
            if s["valid"]:
                cl = resample_branch(s["bp"], step)[self.init_step:]
                if self.max_cl_length is not None:      # legacy override, off by default
                    a = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(cl, axis=0), axis=1))])
                    cl = cl[:max(int(np.searchsorted(a, self.max_cl_length, "right")), 2)]
                # trim to length and add the straight run BEFORE limiting
                # curvature, so the limiter sees the curve that will be swept and
                # the straight tail (curvature zero) cannot be bent by it
                cl, full_arc, ext = self.preprocess_centerline(cl, o, step)
                diag["generated_arc"].append(full_arc)
                diag["extended_by"].append(ext)
                cl_raw = cl
                cl, n_lim = self.limit_curvature(cl, radius)
                tan = get_cpcd_tangent(cl)
                # Straightening a branch bends the direction it leaves by, which
                # is the quantity the sensor measured and the branch model was
                # penalised to respect. Recorded, not silently accepted.
                if n_lim and len(cl_raw) > 1:
                    e0 = cl_raw[1] - cl_raw[0]; e1 = cl[1] - cl[0]
                    drift = float(np.degrees(np.arccos(np.clip(
                        np.dot(e0 / (np.linalg.norm(e0) + 1e-12),
                               e1 / (np.linalg.norm(e1) + 1e-12)), -1, 1))))
                else:
                    drift = 0.0
                diag["max_kr"].append(float(self.discrete_curvature(cl).max() * radius))
                diag["exit_drift_deg"].append(drift)
            else:
                # No usable centerline (missing, or shorter than min_branch_arc), so
                # a straight stub along the sensor tangent stands in for it. For a
                # branch with a minimum length the stub must meet it too: the
                # generated part is zero, so the whole min_len is made up here, and
                # without this a short inlet came out at extrude_length (3 mm)
                # instead of the 10 mm every other inlet gets.
                min_len = self.inlet_min_length if o == 0 else self.outlet_min_length
                ext = self.inlet_extrusion if o == 0 else self.outlet_extrusion
                stub_len = (max(self.extrude_length, min_len + (ext or 0.0))
                            if min_len else self.extrude_length)
                k = max(5, int(np.ceil(stub_len / step)) + 1)
                cl = centroid + s["normal0"] * np.linspace(0.0, stub_len, k)[:, None]
                tan = np.tile(s["normal0"], (k, 1))
                n_lim = 0
                diag["max_kr"].append(0.0)
                diag["exit_drift_deg"].append(0.0)
                diag["generated_arc"].append(0.0)
                diag["extended_by"].append(0.0)
            opening_pcd.append(ring); cpcd_glo.append(cl); cpcd_tan.append(tan)
            necks.append(centroid)
            diag["curvature_limited"].append(int(n_lim))
            diag["branch_valid"].append(bool(s["valid"]))
            diag["arc_length"].append(float(np.linalg.norm(np.diff(cl, axis=0), axis=1).sum()))

        # 4. sweep and merge
        l2w = [get_tubular_l2w_trans(t, min_torsion=self.min_torsion) for t in cpcd_tan]
        tube_verts, sort_idx = get_tubular_mesh_verts(opening_pcd, cpcd_glo, cpcd_tan, l2w)
        for tv, tn in zip(tube_verts, cpcd_tan):
            diag["ring_fold_fraction"].append(self.ring_fold_fraction(tv, tn))
        idx_sorted = [vid[si] for vid, si in zip(opening_indices, sort_idx)]
        tube_faces, tube_verts = get_tubular_mesh_faces(tube_verts, ds_r=self.ds_r)
        merged = merge_meshes(uncapped, tube_faces, tube_verts, idx_sorted)

        # 5. blend the join, clean up, flatten the tube ends
        seam = np.concatenate([np.asarray(v) for v in opening_indices])
        merged = self.smooth_seam(merged, seam)
        merged = remove_orphan_vertices(merged)
        merged = planarize_openings(merged, r_forward_fusion_info_filename=None,
                                    flaw_opening_min_size=self.flaw_opening_min_size,
                                    smooth=False)

        merged, n_closed, closed_radii = self.close_small_holes(merged, keep_n=n_open)
        diag["holes_closed"] = int(n_closed)
        diag["closed_hole_radii"] = [round(r, 4) for r in closed_radii]

        loops = self.boundary_loops(merged)
        diag["n_open_boundaries"] = len(loops)
        diag["expected_boundaries"] = int(n_open)
        diag["n_points"] = int(merged.n_points)
        diag["n_faces"] = int(merged.n_cells)
        diag["max_ring_fold_fraction"] = float(max(diag["ring_fold_fraction"], default=0.0))
        diag["worst_kr"] = float(max(diag["max_kr"], default=0.0))
        diag["worst_exit_drift_deg"] = float(max(diag["exit_drift_deg"], default=0.0))

        # ---- checks. A mesh that passes these is not guaranteed good, but one
        # that fails them is definitely not usable, and the failure is named.
        warnings = []
        folded = [o for o, fr in enumerate(diag["ring_fold_fraction"]) if fr > 0]
        if folded:
            warnings.append(
                "RING FOLDING on branch(es) " + ",".join(map(str, folded)) +
                " (worst %.4f of ring steps move backwards): the centerline turns "
                "tighter than the tube is wide, so the sweep crosses itself"
                % diag["max_ring_fold_fraction"])
        if diag["n_open_boundaries"] != n_open:
            warnings.append(
                "UNFIXED HOLES: %d open boundaries, expected %d. close_small_holes "
                "leaves them alone when the small ones are not clearly separable "
                "from the real openings by size" % (diag["n_open_boundaries"], n_open))
        if self.check_self_intersection:
            n_si, complete = self.self_intersections(merged)
            diag["self_intersecting_pairs"] = int(n_si)
            diag["self_intersection_check_complete"] = bool(complete)
            diag["self_intersecting"] = bool(n_si)
            if n_si:
                warnings.append(
                    "SELF-INTERSECTION: %d triangle pair(s) cross, most likely a tube "
                    "passing through the dome or through another tube" % n_si)
            elif not complete:
                warnings.append("SELF-INTERSECTION CHECK TRUNCATED: too many candidate "
                                "pairs, so a clean result here means 'none found'")
        notes = []
        if diag["worst_kr"] > self.warn_kr:
            warnings.append(
                "CURVATURE NEAR THE FOLDING LIMIT: worst curvature*radius %.2f. Rings "
                "cross as this reaches 1.0" % diag["worst_kr"])
        elif diag["worst_kr"] > self.max_kr + 1e-3:   # tolerance: the limiter stops AT
                                                      # the target, and equality is not news
            notes.append(
                "limiter stopped at curvature*radius %.2f, short of its %.2f target; "
                "not a defect, the margin is just smaller than asked"
                % (diag["worst_kr"], self.max_kr))
        # Self-intersection is FATAL and everything else is advisory. Downstream
        # repair tools handle holes and rough seams; a surface that passes
        # through itself has no inside, so meshing and CFD both fail on it.
        diag["warnings"] = warnings
        diag["notes"] = notes
        diag["fatal"] = bool(diag.get("self_intersecting", False))
        diag["ok"] = not warnings

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            if save_path.suffix.lower() == ".obj":
                pv.save_meshio(str(save_path), merged)
            else:
                merged.save(str(save_path))

        return {"mesh": merged, "boundaries": loops,
                "neck_centroids": np.asarray(necks),
                "neck_normals": np.asarray([t[0] for t in cpcd_tan]),
                "centerlines": cpcd_glo, "diagnostics": diag}
