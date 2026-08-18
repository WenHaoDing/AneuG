"""
record_topology.py — interactive tool to record a canonical shape's full
topology (openings + centerline branches + neck) into ONE consolidated file,
replacing the openings-only dataset/record_openings.ipynb.

Run this on a machine WITH A DISPLAY (like dataset/label_endcaps.py, this
workstation is headless) for the PICKING steps. VMTK extraction and the
sanity renders don't need a display, but are kept in the same script/session
since they follow on immediately from picking.

conda activate vmtk_autogen
python dataset/canonical/record_topology.py --canonical-dir dataset/canonical/Sidewall --aneurysm-type 1
python dataset/canonical/record_topology.py --canonical-dir dataset/canonical/Bifurcated --aneurysm-type 0

What it does, in order
-----------------------
1. Auto-detects the boundary loops of mesh_trimmed.obj (open rings left by
   trimming the branch stubs off mesh.obj — one per opening/branch). This
   reuses record_openings.ipynb's proven edge-walk approach; it's reliable
   here because mesh_trimmed.obj has no duplicate vertices or degenerate
   faces (verified separately) — a real hand-picking UI isn't needed for
   fixing which VERTICES form a loop, only for fixing which loop is which.

2. Interactive: for each loop, in turn, click near the one you want as the
   next opening in sequence (0, 1, 2, ...) — one fresh PyVista window per
   slot, same pattern as label_endcaps.py's brush_one_branch. Order matters:
   this fixes which physical branch is "opening 0" vs "opening 1" etc.
   everywhere downstream.

3. Each opening's loop is walked in ring order (already an ordered walk from
   step 1) and checked for outward orientation: the "average cross vector"
   -- sum_i (p_i - centroid) x (p_{i+1} - centroid), the standard
   sum-of-triangle-fan formula for a polygon's normal -- depends on which
   way the ring is walked. If it points INTO the mesh instead of away from
   it, the saved index order for that opening is reversed (not just the
   vector) so future code walking that ring the same way gets a consistent
   outward sense.

4. Interactive: brush the aneurysm-neck vertices (one shared brush, single
   confirm) -- same brush mechanic as label_endcaps.py's brush_one_branch,
   just one region instead of per-branch. The centroid of these points is
   the reference point AneuSeg's centerline machinery needs in place of a
   real case's aneurysm_centroid:
     - Bifurcated: which detected bifurcation zone is "the" one (relevant if
       VMTK's branch extractor ever finds more than one bifurcation-like
       region -- merge_centerline_v3 already picks the nearest to this point).
     - Sidewall: WHERE to split the single through-vessel branch into two --
       the point on it nearest this centroid, exactly as merge_centerline_v3
       already does for real sidewall cases.

5. Runs AneuSeg's own VMTK centerline extraction (vmtk_extract_centerline)
   and branch decomposition (centerline_post_v3 / merge_centerline_v3) on
   mesh.obj, using each opening's centroid as a VMTK source/target point and
   the neck centroid from step 4 as the split/bifurcation reference --
   reused unchanged, not reimplemented, since this logic already handles
   both topologies correctly for real cases.

6. Matches the resulting branches back to opening 0..n-1 by nearest far-
   endpoint, and reorients each branch's points so pts[0] is the
   neck/bifurcation end and pts[-1] is the opening end (matching this
   codebase's established _ensure_outward convention elsewhere).

7. Saves everything into ONE file: <canonical-dir>/canonical_topology.npy --
   a single pickled dict (matching this codebase's own merged_centerline.npy
   convention, not a flat openings.npz-style key explosion): opening indices
   (outward-oriented, into mesh.obj), opening cross vectors, neck points +
   centroid, per-branch centerline points (matched to opening order), and
   the bifurcation/split point. MultiCanonicalGHDReconstruct._load_openings
   reads this directly; the old openings.npz format is no longer written.

8. Saves sanity images (off-screen, multi-angle) to
   <canonical-dir>/topology_sanity/ showing: opening nodes labelled by
   index (so you can confirm the walk-through is right, not random), each
   opening's cross vector as an arrow labelled by index (confirm outward),
   and centerline branches in per-opening-matched colors plus the neck
   points and bifurcation/split point.
"""

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import trimesh
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
ANEUSEG_DIR = ROOT / "AneuSeg"
if str(ANEUSEG_DIR) not in sys.path:
    sys.path.insert(0, str(ANEUSEG_DIR))

from IAgents.tools.centerline_extraction import vmtk_extract_centerline, centerline_post_v3

TYPE_N_OPEN = {0: 3, 1: 2, 2: 2}
BRANCH_COLORS = ["red", "blue", "green", "orange", "purple", "cyan"]

CONSOLIDATED_NAME = "canonical_topology.npy"


# ── boundary loop detection (reused from record_openings.ipynb) ────────────

def find_boundary_loops(faces):
    """Return list of vertex-index arrays (mesh_trimmed.obj indexing), one
    ordered walk per boundary loop."""
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


def map_loop_to_full(loop_idx_trimmed, trim_verts, full_verts, tol=1e-6):
    """mesh_trimmed.obj shares exact vertex positions with mesh.obj -- map a
    loop's trimmed-mesh indices to mesh.obj indices by exact position match."""
    tree = cKDTree(full_verts)
    dists, full_idx = tree.query(trim_verts[loop_idx_trimmed], k=1)
    assert dists.max() < tol, f"Non-matching vertex (max dist {dists.max():.2e})"
    return full_idx


# ── outward orientation ─────────────────────────────────────────────────────

def cross_vector(pts_ordered):
    """Sum-of-triangle-fan formula: sum_i (p_i - c) x (p_{i+1} - c). Standard
    way to get a polygon's normal from an ORDERED ring -- unlike SVD, its
    sign depends on walk direction, which is exactly what lets us detect and
    fix an inward-facing ring."""
    c = pts_ordered.mean(axis=0)
    rel = pts_ordered - c
    rel_next = np.roll(rel, -1, axis=0)
    v = np.cross(rel, rel_next).sum(axis=0)
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def orient_opening_outward(idx_ordered, full_verts, mesh_centroid):
    """Reverses idx_ordered if its cross_vector points toward the mesh
    interior instead of away from it. Returns (idx_ordered, cross_vec,
    opening_centroid), all consistent with the (possibly reversed) order."""
    pts = full_verts[idx_ordered]
    centroid = pts.mean(axis=0)
    cv = cross_vector(pts)
    if np.dot(cv, centroid - mesh_centroid) < 0:
        idx_ordered = idx_ordered[::-1]
        pts = full_verts[idx_ordered]
        cv = cross_vector(pts)
    return idx_ordered, cv, centroid


# ── interactive picking (needs a real display) ──────────────────────────────

def pick_opening_sequence(mesh_pv, loops_full, full_verts):
    """One fresh window per opening slot: click near the loop you want next.
    Already-assigned loops are shown in their final color + index label;
    unassigned loops are shown faint grey. Returns a list of loop indices
    (into loops_full) in the picked sequence, or raises if the user closes a
    window without picking (so the caller doesn't silently write a partial
    canonical_topology.npy).

    NOT independently tested (no display on this machine) -- same caveat as
    dataset/label_endcaps.py: if _on_pick's warning about 'vtkOriginalCellIds'
    fires, adjust it there for your PyVista version.
    """
    import pyvista as pv

    n = len(loops_full)
    remaining = list(range(n))
    sequence = []

    for slot in range(n):
        picked_loop = {"idx": None}
        plotter = pv.Plotter()
        plotter.add_mesh(mesh_pv, color="whitesmoke", opacity=0.5, show_edges=False)

        for i, li in enumerate(sequence):
            pts = full_verts[loops_full[li]]
            plotter.add_points(pts, color=BRANCH_COLORS[i % len(BRANCH_COLORS)], point_size=10)
            cx, cy, cz = pts.mean(axis=0)
            plotter.add_point_labels([[cx, cy, cz]], [f"opening {i}"], font_size=16, point_size=0)
        for li in remaining:
            pts = full_verts[loops_full[li]]
            plotter.add_points(pts, color="grey", point_size=8, opacity=0.6)

        def _on_pick(picked):
            if picked is None or picked.n_points == 0:
                return
            click_pt = np.asarray(picked.points).mean(axis=0)
            best_li, best_d = None, np.inf
            for li in remaining:
                d = np.linalg.norm(full_verts[loops_full[li]] - click_pt, axis=1).min()
                if d < best_d:
                    best_d, best_li = d, li
            picked_loop["idx"] = best_li
            print(f"  Picked loop {best_li} for opening {slot} (nearest click, dist={best_d:.2f})")

        plotter.add_text(
            f"Click near the boundary loop that is opening {slot} "
            f"({len(remaining)} unassigned loop(s) left, in grey)",
            font_size=11, position="upper_left",
        )
        plotter.enable_point_picking(callback=_on_pick, show_message=True, use_picker=True)
        plotter.show()

        if picked_loop["idx"] is None:
            raise RuntimeError(
                f"No loop picked for opening {slot} — window was closed without a pick. "
                "Rerun the script (nothing has been saved yet)."
            )
        sequence.append(picked_loop["idx"])
        remaining.remove(picked_loop["idx"])

    return sequence


def _mesh_edge_graph(verts, faces):
    """Sparse symmetric weighted (edge-length) adjacency matrix over all mesh
    vertices -- same construction preprocess_endcaps.py uses for graph-distance
    patches."""
    from scipy.sparse import coo_matrix

    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    lengths = np.linalg.norm(verts[edges[:, 0]] - verts[edges[:, 1]], axis=1)
    n = len(verts)
    graph = coo_matrix((lengths, (edges[:, 0], edges[:, 1])), shape=(n, n))
    return (graph + graph.T).tocsr()


def connect_neck_components(verts, faces, picked_idx):
    """A brush pass over a curved, rotating 3D view can easily miss a few
    vertices between two clicked regions, leaving the neck pick as several
    disconnected islands rather than one patch. If that happens, bridge the
    closest pair of islands with the shortest mesh-SURFACE path between them
    (via the same edge-length graph preprocess_endcaps.py uses), repeating
    until the whole neck selection is one connected region, and returns the
    expanded (still index-into-mesh.obj) vertex set."""
    from scipy.sparse.csgraph import connected_components, dijkstra

    graph = _mesh_edge_graph(verts, faces)
    picked_idx = np.array(sorted(set(int(i) for i in picked_idx)))

    sub = graph[picked_idx][:, picked_idx]
    n_comp, labels = connected_components(sub, directed=False)
    if n_comp <= 1:
        return picked_idx

    print(f"  Neck pick has {n_comp} disconnected cluster(s) — "
          f"bridging with shortest mesh-surface paths.")
    components = [picked_idx[labels == c].tolist() for c in range(n_comp)]
    bridge = set()

    while len(components) > 1:
        best = None  # (dist, path, ci, cj)
        for ci, comp in enumerate(components):
            dist, pred, _sources = dijkstra(graph, indices=comp, min_only=True, return_predecessors=True)
            for cj, other in enumerate(components):
                if ci == cj:
                    continue
                k = int(np.argmin(dist[other]))
                d = float(dist[other][k])
                if best is None or d < best[0]:
                    target, cur = other[k], other[k]
                    path = [target]
                    while pred[cur] != -9999:
                        cur = pred[cur]
                        path.append(cur)
                    best = (d, path, ci, cj)
        d, path, ci, cj = best
        print(f"    bridging cluster {ci} <-> cluster {cj}: {len(path)} extra vertex(es), "
              f"{d:.2f} mm apart")
        bridge.update(path)
        merged = components[ci] + components[cj] + path
        components = [c for k, c in enumerate(components) if k not in (ci, cj)] + [merged]

    return np.array(sorted(set(picked_idx.tolist()) | bridge))


def pick_neck_points(mesh_pv):
    """Brush the aneurysm-neck vertices -- same brush mechanic as
    label_endcaps.py's brush_one_branch (drag boxes to accumulate, 'z'
    clears, 'c' confirms), just one region instead of per-branch. The
    confirmed selection is passed through connect_neck_components before
    being returned, so a brush that missed a few connecting vertices between
    two clicked regions doesn't silently leave the neck as several islands."""
    import pyvista as pv

    picked_points = set()
    state = {"highlight": None, "confirmed": False}

    def _refresh_highlight():
        if state["highlight"] is not None:
            plotter.remove_actor(state["highlight"])
            state["highlight"] = None
        if picked_points:
            pts = np.asarray(mesh_pv.points)[sorted(picked_points)]
            state["highlight"] = plotter.add_points(pts, color="orange", point_size=14)

    def _on_pick(picked):
        if picked is None or picked.n_cells == 0:
            return
        ids = picked.cell_data.get("vtkOriginalCellIds")
        if ids is None:
            print(f"[record_topology] 'vtkOriginalCellIds' not in picked.cell_data; "
                  f"available arrays: {picked.array_names}. Edit _on_pick in "
                  f"dataset/canonical/record_topology.py to use the right key for your PyVista version.")
            return
        faces = mesh_pv.faces.reshape(-1, 4)[:, 1:]
        for fid in np.asarray(ids):
            picked_points.update(faces[int(fid)].tolist())
        _refresh_highlight()

    def _reset():
        picked_points.clear()
        _refresh_highlight()

    def _confirm():
        state["confirmed"] = True
        plotter.close()

    plotter = pv.Plotter()
    plotter.add_mesh(mesh_pv, color="whitesmoke", opacity=0.55, show_edges=False)
    plotter.add_text(
        "Drag boxes over the aneurysm NECK region (repeat to add more), "
        "'z' clears, 'c' confirms",
        font_size=11, position="upper_left",
    )
    plotter.enable_cell_picking(callback=_on_pick, through=False, show=False, show_message=True)
    plotter.add_key_event("z", _reset)
    plotter.add_key_event("c", _confirm)
    plotter.show()

    if not state["confirmed"] or not picked_points:
        raise RuntimeError("No neck points confirmed — window was closed without a pick. "
                           "Rerun the script (nothing has been saved yet).")
    verts = np.asarray(mesh_pv.points)
    faces = mesh_pv.faces.reshape(-1, 4)[:, 1:]
    idx = connect_neck_components(verts, faces, np.array(sorted(picked_points)))
    return idx, verts[idx]


# ── VMTK centerline + branch matching ───────────────────────────────────────

def _ensure_split_side_first(pts, ref_point):
    """Orient so pts[0] is the end nearest ref_point (neck/bifurcation side)
    and pts[-1] is the far/opening end -- same convention as
    _ensure_outward elsewhere in this codebase (vessel_clipping.py etc.)."""
    if np.linalg.norm(pts[-1] - ref_point) < np.linalg.norm(pts[0] - ref_point):
        return pts[::-1]
    return pts


def extract_and_match_branches(canonical_dir, mesh_path, opening_centroids, ref_point,
                                aneurysm_type, source_id=0):
    """Runs vmtk_extract_centerline + centerline_post_v3 (reused unchanged
    from AneuSeg) and matches the resulting branches back to opening order
    0..n-1 by nearest far-endpoint (Hungarian assignment, robust to more
    detected branches than openings). Returns (branch_points_by_opening,
    split_point)."""
    endpoints = [np.asarray(c) for c in opening_centroids]
    cl_smooth, _ = vmtk_extract_centerline(str(canonical_dir), source_id, endpoints,
                                           mesh_filename=Path(mesh_path).name)
    merged = centerline_post_v3(cl_smooth, ref_point, aneurysm_type, str(canonical_dir),
                                centerline_filename="vmtk_centerline_raw.npy")

    branches = merged["branches"]
    connected_branch_ids = merged["connected_branch_ids"]
    split_point = np.asarray(merged["nearest_bif"]["centroid"])

    candidates = []
    for bid in connected_branch_ids:
        pts = np.asarray(branches[bid]["pts"])
        pts = _ensure_split_side_first(pts, split_point)
        far_end = pts[-1]
        candidates.append((bid, pts, far_end))

    n_open = len(opening_centroids)
    cost = np.full((n_open, len(candidates)), np.inf)
    for i, oc in enumerate(opening_centroids):
        for j, (_, _, far_end) in enumerate(candidates):
            cost[i, j] = np.linalg.norm(np.asarray(oc) - far_end)
    row_ind, col_ind = linear_sum_assignment(cost)
    if len(row_ind) < n_open:
        raise RuntimeError(
            f"Only matched {len(row_ind)}/{n_open} openings to centerline branches "
            f"({len(candidates)} branch(es) found). Check the neck pick / VMTK log."
        )

    branch_points_by_opening = [None] * n_open
    for r, c in zip(row_ind, col_ind):
        d = cost[r, c]
        print(f"  Opening {r} <- branch {candidates[c][0]}  (endpoint match dist={d:.3f})")
        if d > 5.0:
            print(f"  Warning: opening {r}'s matched branch endpoint is {d:.2f} away "
                  f"from the opening centroid -- double check this match.")
        branch_points_by_opening[r] = candidates[c][1]

    return branch_points_by_opening, split_point


# ── sanity images ────────────────────────────────────────────────────────────

def render_sanity_images(canonical_dir, mesh_pv, opening_indices, opening_cross_vecs,
                         neck_points, branch_points_by_opening, split_point,
                         n_angles=6):
    import pyvista as pv

    render_dir = Path(canonical_dir) / "topology_sanity"
    render_dir.mkdir(parents=True, exist_ok=True)
    full_verts = np.asarray(mesh_pv.points)

    focal = np.array(mesh_pv.center)
    bounds = mesh_pv.bounds
    xy_diag = np.sqrt((bounds[1] - bounds[0]) ** 2 + (bounds[3] - bounds[2]) ** 2)
    cam_dist = xy_diag * 1.4
    arrow_scale = xy_diag * 0.15

    for i in range(n_angles):
        angle_deg = round(360 * i / n_angles)
        angle_rad = np.deg2rad(angle_deg)
        cam_pos = focal + cam_dist * np.array([np.cos(angle_rad), np.sin(angle_rad), 0.3])

        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(mesh_pv, color="whitesmoke", opacity=0.25, smooth_shading=True)

        for o, idx in enumerate(opening_indices):
            c = BRANCH_COLORS[o % len(BRANCH_COLORS)]
            pts = full_verts[idx]
            plotter.add_points(pts, color=c, point_size=10, label=f"opening {o}")
            centroid = pts.mean(axis=0)
            plotter.add_point_labels([centroid], [str(o)], font_size=20, point_size=0,
                                     text_color=c, shape=None)
            arrow = pv.Arrow(start=centroid, direction=opening_cross_vecs[o], scale=arrow_scale)
            plotter.add_mesh(arrow, color=c)

        if neck_points is not None and len(neck_points) > 0:
            plotter.add_points(neck_points, color="black", point_size=6, opacity=0.5, label="neck")

        if split_point is not None:
            plotter.add_mesh(pv.Sphere(radius=arrow_scale * 0.15, center=split_point),
                             color="yellow", label="bifurcation/split")

        for o, pts in enumerate(branch_points_by_opening):
            if pts is None:
                continue
            c = BRANCH_COLORS[o % len(BRANCH_COLORS)]
            plotter.add_mesh(pv.Spline(np.asarray(pts)), color=c, line_width=5)

        plotter.add_legend(size=(0.25, 0.2))
        plotter.set_background("white")
        plotter.camera.position = cam_pos
        plotter.camera.focal_point = focal
        plotter.camera.up = (0.0, 0.0, 1.0)

        out_path = render_dir / f"topology_{angle_deg:03d}deg.png"
        plotter.screenshot(str(out_path))
        plotter.close()
        print(f"  Sanity render saved: {out_path}")

    return render_dir


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-dir", required=True)
    parser.add_argument("--aneurysm-type", type=int, required=True, choices=[0, 1, 2],
                        help="0=bifurcated, 1/2=sidewall (1 and 2 share the same canonical shape).")
    parser.add_argument("--source-id", type=int, default=0,
                        help="Which opening (in picked sequence order) VMTK treats as the source "
                             "point; the rest become targets. Doesn't affect the resulting topology.")
    args = parser.parse_args()

    canonical_dir = Path(args.canonical_dir)
    mesh_path = canonical_dir / "mesh.obj"
    mesh_trimmed_path = canonical_dir / "mesh_trimmed.obj"

    mesh = trimesh.load(mesh_path, process=False)
    mesh_trimmed = trimesh.load(mesh_trimmed_path, process=False)
    full_verts = np.asarray(mesh.vertices)
    mesh_centroid = full_verts.mean(axis=0)

    loops_trimmed = find_boundary_loops(mesh_trimmed.faces)
    loops_full = [map_loop_to_full(loop, np.asarray(mesh_trimmed.vertices), full_verts)
                 for loop in loops_trimmed]
    print(f"Found {len(loops_full)} boundary loop(s): sizes {[len(l) for l in loops_full]}")
    expected = TYPE_N_OPEN[args.aneurysm_type]
    if len(loops_full) != expected:
        print(f"Warning: expected {expected} opening(s) for aneurysm_type={args.aneurysm_type}, "
              f"found {len(loops_full)}.")

    import pyvista as pv
    mesh_pv = pv.PolyData(full_verts, np.hstack([np.full((len(mesh.faces), 1), 3), mesh.faces]).ravel())

    print("\n--- Step 1/3: pick the opening sequence ---")
    sequence = pick_opening_sequence(mesh_pv, loops_full, full_verts)

    opening_indices, opening_cross_vecs, opening_centroids = [], [], []
    for li in sequence:
        idx, cv, centroid = orient_opening_outward(loops_full[li], full_verts, mesh_centroid)
        opening_indices.append(idx)
        opening_cross_vecs.append(cv)
        opening_centroids.append(centroid)

    print("\n--- Step 2/3: brush the aneurysm neck ---")
    neck_idx, neck_points = pick_neck_points(mesh_pv)
    neck_centroid = neck_points.mean(axis=0)

    print("\n--- Step 3/3: VMTK centerline extraction + branch matching ---")
    branch_points_by_opening, split_point = extract_and_match_branches(
        canonical_dir, mesh_path, opening_centroids, neck_centroid,
        args.aneurysm_type, source_id=args.source_id,
    )

    out_path = canonical_dir / CONSOLIDATED_NAME
    topology = {
        "aneurysm_type": args.aneurysm_type,
        "num_openings": len(opening_indices),
        "openings": [
            {
                "indices": opening_indices[i].astype(np.int64),
                "cross_vector": opening_cross_vecs[i].astype(np.float32),
                "centroid": opening_centroids[i].astype(np.float32),
                "branch_points": branch_points_by_opening[i].astype(np.float32),
            }
            for i in range(len(opening_indices))
        ],
        "neck_vertex_indices": neck_idx.astype(np.int64),
        "neck_points": neck_points.astype(np.float32),
        "neck_centroid": neck_centroid.astype(np.float32),
        "split_point": split_point.astype(np.float32),
    }
    np.save(out_path, topology, allow_pickle=True)
    print(f"\nSaved consolidated topology -> {out_path}")

    render_dir = render_sanity_images(
        canonical_dir, mesh_pv, opening_indices, opening_cross_vecs,
        neck_points, branch_points_by_opening, split_point,
    )
    print(f"Sanity images saved to {render_dir}")


if __name__ == "__main__":
    main()
