"""
record_topology.py — interactive tool to record a canonical shape's full
topology (openings + centerline branches + neck) into ONE consolidated file,
replacing the openings-only dataset/record_openings.ipynb.

Picking (steps 1-4) needs a real display (like dataset/label_endcaps.py, this
workstation is headless). VMTK extraction (steps 5-8) needs the vmtk env but
NOT a display. These don't have to be the same machine -- --mode pick and
--mode process split the script exactly at that boundary, handing off via
canonical_picks.npy. --mode all (default) does both in one go, for a machine
that has both a display and vmtk.

# one machine with both:
conda activate vmtk_autogen
python dataset/canonical/record_topology.py --canonical-dir dataset/canonical/Sidewall --aneurysm-type 1

# split across two machines (e.g. local workstation has a display but no vmtk,
# a headless machine has vmtk but no display):
#   on the machine WITH A DISPLAY (any env with pyvista/trimesh, no vmtk needed):
python dataset/canonical/record_topology.py --canonical-dir dataset/canonical/Sidewall --aneurysm-type 1 --mode pick
python dataset/canonical/record_topology.py --canonical-dir dataset/canonical/Bifurcated --aneurysm-type 0 --mode pick
#   copy canonical_picks.npy to the machine WITH VMTK, then:
conda activate vmtk_autogen
python dataset/canonical/record_topology.py --canonical-dir dataset/canonical/Sidewall --mode process

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

4. Interactive: brush the aneurysm DOME -- the whole bulging sac, not the
   thin neck ring directly (much more forgiving to cover completely; one
   shared brush, single confirm, same mechanic as label_endcaps.py's
   brush_one_branch). The neck is then DERIVED, not picked: the dome patch
   is treated as its own trimmed mesh and its boundary loop is found with
   the same edge-walk technique used for mesh.obj's own openings
   (detect_neck_from_dome / find_boundary_loops) -- the neck is simply where
   the dome patch meets the rest of the vessel. Both the dome's vertex
   indices (into mesh.obj) and the derived neck ring are recorded. The
   neck centroid is the reference point AneuSeg's centerline machinery needs
   in place of a real case's aneurysm_centroid:
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
   (outward-oriented, into mesh.obj), opening cross vectors, dome vertex
   indices (into mesh.obj), neck points + centroid (the dome patch's own
   boundary), per-branch centerline points (matched to opening order), and
   the bifurcation/split point. MultiCanonicalGHDReconstruct._load_openings
   reads this directly; the old openings.npz format is no longer written.

8. Saves sanity images (off-screen, multi-angle) to
   <canonical-dir>/topology_sanity/ showing: opening nodes labelled by
   index (so you can confirm the walk-through is right, not random), each
   opening's cross vector as an arrow labelled by index (confirm outward),
   the dome patch (faint gold) and derived neck ring (black), and centerline
   branches in per-opening-matched colors plus the bifurcation/split point.
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

# NOT imported at module level: IAgents.tools.centerline_extraction pulls in
# vmtk (and several other heavy deps) at ITS top level, which would make
# --mode pick fail to even start on a machine that has a display but no vmtk
# env. Imported lazily inside extract_and_match_branches instead, so only
# --mode process/all (which actually need vmtk) require it.

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
    """One fresh window per opening slot: click near the loop you want next,
    then press 'c' to confirm it and close the window (click again first if
    you want to change your pick -- clicking doesn't close the window by
    itself, only 'c' does). Already-assigned loops are shown in their final
    color + index label; unassigned loops are shown faint grey; your current
    tentative pick is highlighted orange. Returns a list of loop indices
    (into loops_full) in the picked sequence, or raises if the user closes a
    window without confirming a pick (so the caller doesn't silently write a
    partial canonical_topology.npy).

    NOT independently tested (no display on this machine) -- same caveat as
    dataset/label_endcaps.py: if _on_pick's warning about 'vtkOriginalCellIds'
    fires, adjust it there for your PyVista version. Also flagging enable_point_picking
    itself as unverified here: it's documented as a plain left-click picker (unlike
    enable_cell_picking's rubber-band-drag style used elsewhere in this project, which
    turned out to need a specific gesture we only found by testing), but if left-click
    alone doesn't trigger _on_pick, check the terminal for PyVista's own on-screen
    hint (show_message=True) and adjust the enable_point_picking call below.
    """
    import pyvista as pv

    n = len(loops_full)
    remaining = list(range(n))
    sequence = []

    for slot in range(n):
        state = {"idx": None, "highlight": None, "confirmed": False}
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

        def _refresh_highlight():
            if state["highlight"] is not None:
                plotter.remove_actor(state["highlight"])
                state["highlight"] = None
            if state["idx"] is not None:
                pts = full_verts[loops_full[state["idx"]]]
                state["highlight"] = plotter.add_points(pts, color="orange", point_size=14)

        def _on_pick(point):
            # enable_point_picking's callback receives the picked 3D point
            # itself (a plain [3] coordinate) -- NOT a picked sub-mesh with
            # .points/.n_points the way enable_cell_picking's callback does
            # elsewhere in this script. use_picker=True (removed below) would
            # instead call this with (point, picker), a 2-arg signature.
            if point is None:
                return
            click_pt = np.asarray(point)
            best_li, best_d = None, np.inf
            for li in remaining:
                d = np.linalg.norm(full_verts[loops_full[li]] - click_pt, axis=1).min()
                if d < best_d:
                    best_d, best_li = d, li
            state["idx"] = best_li
            print(f"  Tentatively picked loop {best_li} for opening {slot} (nearest click, "
                  f"dist={best_d:.2f}) -- click again to change, 'c' to confirm.")
            _refresh_highlight()

        def _confirm():
            if state["idx"] is None:
                print("  Nothing picked yet -- click near a (grey) loop first, then press 'c'.")
                return
            state["confirmed"] = True
            plotter.close()

        plotter.add_text(
            f"Opening {slot}: LEFT-CLICK near the loop you want next (orange = current pick, "
            f"grey = unassigned, {len(remaining)} left), then 'c' to confirm and continue",
            font_size=11, position="upper_left",
        )
        # left_clicking=True: enable_point_picking defaults to RIGHT-click, which would be
        # inconsistent with pick_neck_points' left-click-drag brush elsewhere in this script.
        plotter.enable_point_picking(callback=_on_pick, show_message=True, left_clicking=True)
        plotter.add_key_event("c", _confirm)
        plotter.show()

        if not state["confirmed"] or state["idx"] is None:
            raise RuntimeError(
                f"No loop confirmed for opening {slot} — window was closed without pressing 'c'. "
                "Rerun the script (nothing has been saved yet)."
            )
        sequence.append(state["idx"])
        remaining.remove(state["idx"])

    return sequence


def _picked_faces_largest_component(faces_all, picked_face_ids):
    """Face-adjacency (share-an-edge) connected components of the picked face
    subset. A dome brush is a big, forgiving 2D blob -- much easier to cover
    without gaps in one pass than a thin neck ring was -- so rather than
    trying to bridge disconnected pieces (which isn't well-defined for a 2D
    patch the way it was for a 1D ring), a stray disconnected island just
    means a misclick: keep the largest piece and tell the user what got
    dropped, so they can redo the brush if it wasn't a misclick."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    picked_face_ids = np.array(sorted(set(int(i) for i in picked_face_ids)))
    sub_faces = faces_all[picked_face_ids]  # [P, 3], global vertex indices

    edge_to_local_faces = defaultdict(list)
    for local_i, f in enumerate(sub_faces):
        for e in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])]:
            edge_to_local_faces[tuple(sorted(e))].append(local_i)

    rows, cols = [], []
    for local_faces in edge_to_local_faces.values():
        for a in range(len(local_faces)):
            for b in range(a + 1, len(local_faces)):
                rows.append(local_faces[a])
                cols.append(local_faces[b])

    p = len(sub_faces)
    adj = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(p, p))
    n_comp, labels = connected_components(adj, directed=False)
    if n_comp <= 1:
        return picked_face_ids

    sizes = np.bincount(labels)
    keep = int(np.argmax(sizes))
    print(f"  Dome pick has {n_comp} disconnected piece(s) (face counts {sizes.tolist()}) -- "
          f"keeping the largest ({sizes[keep]} faces), discarding the rest as likely stray clicks. "
          f"Redo the brush if that's wrong.")
    return picked_face_ids[labels == keep]


def detect_neck_from_dome(faces_all, dome_face_ids):
    """Treats the picked dome patch as its own trimmed mesh and finds ITS
    boundary loop -- the neck is where the dome patch meets the rest of the
    vessel, exactly the same edge-walk technique used to find mesh_trimmed.obj's
    openings (find_boundary_loops), just applied to a hand-picked patch instead
    of a pre-trimmed file. dome_face_ids indexes faces_all (mesh.obj's own face
    array), so the returned loop is already in mesh.obj vertex indexing -- no
    mapping step needed, unlike the openings (which come from the SEPARATE
    mesh_trimmed.obj file and do need one).

    Returns (neck_loop_indices, dome_vertex_indices)."""
    dome_faces = faces_all[dome_face_ids]
    loops = find_boundary_loops(dome_faces)
    if len(loops) == 0:
        raise RuntimeError(
            "Dome patch has no boundary -- did the brush accidentally cover the WHOLE "
            "closed mesh? Redo the brush covering just the dome bulge."
        )
    if len(loops) > 1:
        sizes = [len(l) for l in loops]
        print(f"  Warning: dome patch has {len(loops)} boundary loops (sizes {sizes}), expected "
              f"exactly 1 for a clean disk-shaped patch -- using the largest. If the dome brush "
              f"has a hole in it or wraps around the vessel, redo it.")
    neck_loop = max(loops, key=len)
    dome_vertex_indices = np.unique(dome_faces.ravel())
    return neck_loop, dome_vertex_indices


def pick_dome_points(mesh_pv):
    """Brush the aneurysm DOME (the whole bulging sac, not the thin neck
    ring -- much easier to cover completely) -- same brush mechanic as
    label_endcaps.py's brush_one_branch (drag boxes to accumulate faces, 'z'
    clears, 'c' confirms). The neck is derived afterward from this patch's
    own boundary (detect_neck_from_dome), not picked directly. Returns the
    confirmed dome face ids (into mesh.obj's face array, post
    largest-connected-component cleanup)."""
    import pyvista as pv

    picked_faces = set()
    state = {"highlight": None, "confirmed": False}
    faces = mesh_pv.faces.reshape(-1, 4)[:, 1:]

    def _refresh_highlight():
        if state["highlight"] is not None:
            plotter.remove_actor(state["highlight"])
            state["highlight"] = None
        if picked_faces:
            sub = mesh_pv.extract_cells(sorted(picked_faces))
            state["highlight"] = plotter.add_mesh(sub, color="orange", opacity=0.9, show_edges=True)

    def _on_pick(picked):
        if picked is None or picked.n_cells == 0:
            return
        ids = picked.cell_data.get("vtkOriginalCellIds")
        if ids is None:
            print(f"[record_topology] 'vtkOriginalCellIds' not in picked.cell_data; "
                  f"available arrays: {picked.array_names}. Edit _on_pick in "
                  f"dataset/canonical/record_topology.py to use the right key for your PyVista version.")
            return
        picked_faces.update(int(i) for i in np.asarray(ids))
        _refresh_highlight()

    def _reset():
        picked_faces.clear()
        _refresh_highlight()

    def _confirm():
        state["confirmed"] = True
        plotter.close()

    plotter = pv.Plotter()
    plotter.add_mesh(mesh_pv, color="whitesmoke", opacity=0.55, show_edges=False)
    plotter.add_text(
        "Drag boxes over the WHOLE aneurysm DOME (repeat/rotate to cover it fully -- "
        "don't worry about precision at the edge), 'z' clears, 'c' confirms",
        font_size=11, position="upper_left",
    )
    plotter.enable_cell_picking(callback=_on_pick, through=False, show=False, show_message=True)
    plotter.add_key_event("z", _reset)
    plotter.add_key_event("c", _confirm)
    plotter.show()

    if not state["confirmed"] or not picked_faces:
        raise RuntimeError("No dome patch confirmed — window was closed without a pick. "
                           "Rerun the script (nothing has been saved yet).")
    return _picked_faces_largest_component(faces, np.array(sorted(picked_faces)))


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
    from IAgents.tools.centerline_extraction import vmtk_extract_centerline, centerline_post_v3

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
                         n_angles=6, dome_vertex_indices=None):
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

        if dome_vertex_indices is not None and len(dome_vertex_indices) > 0:
            plotter.add_points(full_verts[dome_vertex_indices], color="gold", point_size=4,
                               opacity=0.3, label="dome")

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


# ── pick / process split ─────────────────────────────────────────────────────
# vmtk (needed for centerline extraction) and a real display (needed for
# interactive picking) don't have to live on the same machine -- e.g. a local
# workstation with a display but no vmtk env, and a headless machine with
# vmtk but no display. --mode pick does ONLY steps 1-4 (no vmtk import at
# module load time even) and saves an intermediate canonical_picks.npy;
# --mode process picks that up and does steps 5-8 (no display needed, off-
# screen rendering only). --mode all (default) does everything in one go,
# for a machine that has both.

PICKS_NAME = "canonical_picks.npy"


def load_mesh_and_loops(canonical_dir, aneurysm_type):
    mesh_path = canonical_dir / "mesh.obj"
    mesh_trimmed_path = canonical_dir / "mesh_trimmed.obj"

    mesh = trimesh.load(mesh_path, process=False)
    mesh_trimmed = trimesh.load(mesh_trimmed_path, process=False)
    full_verts = np.asarray(mesh.vertices)

    loops_trimmed = find_boundary_loops(mesh_trimmed.faces)
    loops_full = [map_loop_to_full(loop, np.asarray(mesh_trimmed.vertices), full_verts)
                 for loop in loops_trimmed]
    print(f"Found {len(loops_full)} boundary loop(s): sizes {[len(l) for l in loops_full]}")
    expected = TYPE_N_OPEN[aneurysm_type]
    if len(loops_full) != expected:
        print(f"Warning: expected {expected} opening(s) for aneurysm_type={aneurysm_type}, "
              f"found {len(loops_full)}.")
    return mesh, full_verts, loops_full


def run_pick(canonical_dir, aneurysm_type):
    """Steps 1-4 only (no vmtk import needed at all): pick the opening
    sequence, orient each ring outward, brush the neck. Saves
    canonical_picks.npy for run_process to pick up later, possibly on a
    different machine."""
    import pyvista as pv

    mesh, full_verts, loops_full = load_mesh_and_loops(canonical_dir, aneurysm_type)
    mesh_centroid = full_verts.mean(axis=0)
    mesh_pv = pv.PolyData(full_verts, np.hstack([np.full((len(mesh.faces), 1), 3), mesh.faces]).ravel())

    print("\n--- Step 1/2: pick the opening sequence ---")
    sequence = pick_opening_sequence(mesh_pv, loops_full, full_verts)

    opening_indices, opening_cross_vecs, opening_centroids = [], [], []
    for li in sequence:
        idx, cv, centroid = orient_opening_outward(loops_full[li], full_verts, mesh_centroid)
        opening_indices.append(idx)
        opening_cross_vecs.append(cv)
        opening_centroids.append(centroid)

    print("\n--- Step 2/2: brush the aneurysm dome (neck is derived from its boundary) ---")
    dome_face_ids = pick_dome_points(mesh_pv)
    faces = mesh_pv.faces.reshape(-1, 4)[:, 1:]
    neck_idx, dome_vertex_idx = detect_neck_from_dome(faces, dome_face_ids)
    neck_points = full_verts[neck_idx]
    neck_centroid = neck_points.mean(axis=0)
    print(f"  Dome patch: {len(dome_vertex_idx)} vertices, {len(dome_face_ids)} faces. "
          f"Detected neck: {len(neck_idx)} boundary vertices.")

    picks = {
        "aneurysm_type": aneurysm_type,
        "opening_indices": [idx.astype(np.int64) for idx in opening_indices],
        "opening_cross_vectors": [cv.astype(np.float32) for cv in opening_cross_vecs],
        "opening_centroids": [c.astype(np.float32) for c in opening_centroids],
        "dome_face_indices": dome_face_ids.astype(np.int64),
        "dome_vertex_indices": dome_vertex_idx.astype(np.int64),
        "neck_vertex_indices": neck_idx.astype(np.int64),
        "neck_points": neck_points.astype(np.float32),
        "neck_centroid": neck_centroid.astype(np.float32),
    }
    out_path = canonical_dir / PICKS_NAME
    np.save(out_path, picks, allow_pickle=True)
    print(f"\nSaved picks -> {out_path}")
    print(f"Run with --mode process on a machine with vmtk to finish (no display needed there).")


def run_process(canonical_dir, source_id=0):
    """Steps 5-8: load canonical_picks.npy (from run_pick, possibly on a
    different machine), run vmtk centerline extraction + branch matching,
    save the final canonical_topology.npy, and render sanity images. No
    display needed -- off-screen rendering only."""
    import pyvista as pv

    picks_path = canonical_dir / PICKS_NAME
    if not picks_path.exists():
        raise FileNotFoundError(
            f"{picks_path} not found. Run with --mode pick first (on a machine with a display)."
        )
    picks = np.load(picks_path, allow_pickle=True).item()
    aneurysm_type = picks["aneurysm_type"]
    opening_indices = picks["opening_indices"]
    opening_cross_vecs = picks["opening_cross_vectors"]
    opening_centroids = picks["opening_centroids"]
    neck_points = picks["neck_points"]
    neck_centroid = picks["neck_centroid"]
    dome_vertex_indices = picks.get("dome_vertex_indices")  # None for a picks file from before dome brushing

    mesh_path = canonical_dir / "mesh.obj"
    mesh = trimesh.load(mesh_path, process=False)
    full_verts = np.asarray(mesh.vertices)
    mesh_pv = pv.PolyData(full_verts, np.hstack([np.full((len(mesh.faces), 1), 3), mesh.faces]).ravel())

    print("\n--- Step 3/3: VMTK centerline extraction + branch matching ---")
    branch_points_by_opening, split_point = extract_and_match_branches(
        canonical_dir, mesh_path, opening_centroids, neck_centroid,
        aneurysm_type, source_id=source_id,
    )

    out_path = canonical_dir / CONSOLIDATED_NAME
    topology = {
        "aneurysm_type": aneurysm_type,
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
        "dome_vertex_indices": (dome_vertex_indices.astype(np.int64)
                                if dome_vertex_indices is not None else None),
        "neck_vertex_indices": picks["neck_vertex_indices"].astype(np.int64),
        "neck_points": neck_points.astype(np.float32),
        "neck_centroid": neck_centroid.astype(np.float32),
        "split_point": split_point.astype(np.float32),
    }
    np.save(out_path, topology, allow_pickle=True)
    print(f"\nSaved consolidated topology -> {out_path}")

    render_dir = render_sanity_images(
        canonical_dir, mesh_pv, opening_indices, opening_cross_vecs,
        neck_points, branch_points_by_opening, split_point,
        dome_vertex_indices=dome_vertex_indices,
    )
    print(f"Sanity images saved to {render_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-dir", required=True)
    parser.add_argument("--aneurysm-type", type=int, choices=[0, 1, 2],
                        help="0=bifurcated, 1/2=sidewall (1 and 2 share the same canonical shape). "
                             "Required for --mode pick/all.")
    parser.add_argument("--mode", choices=["pick", "process", "all"], default="all",
                        help="pick: interactive steps only (needs a display, NOT vmtk), saves "
                             "canonical_picks.npy. process: vmtk + save + render (needs vmtk, "
                             "NOT a display), reads canonical_picks.npy from a prior --mode pick "
                             "run -- possibly on a different machine. all: everything in one go, "
                             "for a machine with both.")
    parser.add_argument("--source-id", type=int, default=0,
                        help="Which opening (in picked sequence order) VMTK treats as the source "
                             "point; the rest become targets. Doesn't affect the resulting topology.")
    args = parser.parse_args()

    canonical_dir = Path(args.canonical_dir)

    if args.mode in ("pick", "all") and args.aneurysm_type is None:
        parser.error("--aneurysm-type is required for --mode pick/all")

    if args.mode == "pick":
        run_pick(canonical_dir, args.aneurysm_type)
    elif args.mode == "process":
        run_process(canonical_dir, source_id=args.source_id)
    else:
        run_pick(canonical_dir, args.aneurysm_type)
        run_process(canonical_dir, source_id=args.source_id)


if __name__ == "__main__":
    main()
