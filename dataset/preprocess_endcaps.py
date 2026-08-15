"""
Endcap dataset (endpoint + tangent + local patch) — derived from the
already-processed runtime/dataset/processed checkpoints, not raw scans.

Context (Plan C): rather than predicting each branch's start point as a raw,
ungrounded xyz regression (models/branch_mlp_vae_claydoll.py's
GCNMeshEncoderWithStartPoints) or relying on a FIXED canonical vertex-index
lookup that can drift for generated shapes
(MultiCanonicalGHDReconstruct.compute_branch_conditions, openings.npz), the new
plan is: a GCN+FPS encoder cross-attends per branch slot over real mesh-vertex
tokens, predicting the endpoint as a soft-argmax (attention-weighted average)
over those vertices' own positions — grounded on the surface by construction —
plus a tangent direction, and (this script's newest addition) an auxiliary
per-vertex classification target so the attention distribution itself has a
direct supervision signal, not just the downstream regression loss. This
script produces all three targets.

Endpoint target: why not just reuse runtime/dataset/processed's own clipped_centerline
branch_points[0] directly — empirically, point[0] sits ~0.4-1.6mm OUTSIDE the
canonical/GHD-reconstructed mesh surface (verified: 15-case spot check, see
conversation), almost exactly along the branch's own forward tangent — because
clipped_centerline's cut location comes from a different (real, per-case CFD)
processing pipeline than the canonical GHD template's own (fixed) vessel-stub
extent. A target that isn't ON the mesh surface would teach the model the
wrong thing (the whole point is "a point at the end of the vessel, on the
surface"). So for each branch we instead find where the branch's own polyline,
walked forward from point[0], first crosses the reconstructed mesh's surface
(a real ray/triangle intersection test). Two things can go wrong, each with
its own fallback (both flagged in the output for diagnostics, not silently
accepted):
  - No crossing within the search window, but the branch has more length than
    the window covers (rare) -> nearest point on the mesh surface to pts[0].
  - The branch itself is shorter than needed to reach the mesh (common — many
    real branches are only a fraction of a mm long) -> extrapolate a ray from
    the branch's own trailing tangent direction, continuing past its last
    recorded point until it actually crosses the mesh. This is what lets
    short branches contribute an endpoint at all, instead of being dropped
    (previously: any branch under MIN_ARC_LENGTH_MM was skipped outright,
    which is why some cases were missing a branch in the first pass).

Tangent target: the average unit tangent direction along the branch's own
path from pts[0] up to the endpoint (including the extrapolated segment, if
elongation was used) — i.e. only the portion of the branch BEFORE the mesh
crossing contributes, nothing past it (the vessel may curve further out in
ways that shouldn't influence "which way does it enter the mesh").

Patch target (in_patch / patch_distances): a naive auxiliary classification
loss against a one-hot "nearest vertex" label is blind to how wrong a wrong
prediction is — cross-entropy only looks at probability mass on the true
class, so putting weight on an immediately-adjacent vertex costs exactly the
same as putting it on the far side of the mesh. The fix is a soft target with
real spatial extent around the endpoint. Naively that extent would be defined
by Euclidean distance, but the mesh isn't uniformly dense and can fold back on
itself (e.g. across a thin vessel wall, two points close in 3D can be far
apart along the actual surface) — so extent is measured as GRAPH distance:
total edge length of the shortest path across the mesh surface, via Dijkstra.
Every vertex reachable within PATCH_THRESHOLD_MM of edge-length from the
endpoint is in the patch. in_patch is the resulting boolean per-vertex mask;
patch_distances is the underlying continuous distance field (kept too, so the
threshold can be revisited later without re-running Dijkstra over every case).

Kept deliberately minimal (per instruction: no need to record so much stuff):
each output checkpoint has case / aneurysm_type / phi (needed to reconstruct
the mesh later) plus endpoints / tangents / branch_mask / used_fallback /
used_elongation / in_patch / patch_distances. Nothing else from
runtime/dataset/processed's much richer schema is carried over.

conda activate new
python dataset/preprocess_endcaps.py --save-sanity
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.preprocess_ImperialNHS import _reconstruct_ghd_numpy, _set_axes_equal

DEFAULT_INPUT_DIR = ROOT / "runtime" / "dataset" / "processed"
DEFAULT_OUTPUT_DIR = ROOT / "runtime" / "dataset" / "processed_endcaps"
MAX_BRANCHES = 3
SEARCH_WINDOW_MM = 8.0        # how far forward along the branch's OWN points to look for a crossing
TANGENT_TAIL_POINTS = 5       # points used to estimate direction when extrapolating past a short branch
TANGENT_ARROW_MM = 3.0        # sanity-plot arrow length only
PATCH_THRESHOLD_MM = 1.0      # graph-distance radius defining the "cap" patch around each endpoint


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return (v / n) if n > 1e-8 else v


def _arc_length_cumulative(pts):
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def _average_tangent(path_pts):
    """Mean of consecutive unit segment tangents along path_pts [K,3], K>=2 —
    the branch's OWN direction from its start up to (only up to) the endpoint."""
    seg = np.diff(path_pts, axis=0)
    lengths = np.linalg.norm(seg, axis=1)
    valid = lengths > 1e-8
    if not valid.any():
        return np.zeros(3, dtype=np.float32)
    units = seg[valid] / lengths[valid, None]
    return _unit(units.mean(axis=0)).astype(np.float32)


def _forward_crossing(mesh, seg_pts, seg_arc):
    """First mesh crossing walking forward along consecutive seg_pts.
    Returns (hit_point, path_pts_up_to_and_including_hit) or None."""
    origins = seg_pts[:-1]
    dirs = seg_pts[1:] - seg_pts[:-1]
    lengths = np.linalg.norm(dirs, axis=1)
    valid = lengths > 1e-8
    if not valid.any():
        return None
    v_origins = origins[valid]
    v_dirs = dirs[valid] / lengths[valid, None]
    v_arc = seg_arc[:-1][valid]
    v_orig_idx = np.where(valid)[0]

    locs, ray_idx, _ = mesh.ray.intersects_location(ray_origins=v_origins, ray_directions=v_dirs)
    if len(locs) == 0:
        return None
    hit_dist = np.linalg.norm(locs - v_origins[ray_idx], axis=1)
    in_segment = hit_dist <= lengths[valid][ray_idx] + 1e-6
    if not in_segment.any():
        return None
    locs, ray_idx = locs[in_segment], ray_idx[in_segment]
    order = np.argsort(v_arc[ray_idx])   # first crossing = smallest arc length from pts[0]
    best = order[0]
    hit_point = locs[best].astype(np.float32)
    seg_start_idx = v_orig_idx[ray_idx[best]]
    path_pts = np.concatenate([seg_pts[:seg_start_idx + 1], hit_point[None]], axis=0)
    return hit_point, path_pts


def find_branch_endpoint(mesh, pts, search_window_mm=SEARCH_WINDOW_MM,
                         tangent_tail_points=TANGENT_TAIL_POINTS):
    """Where branch `pts` (pts[0] nearest the aneurysm) crosses `mesh`'s surface.

    1. Walk forward along the branch's own recorded points, looking for a crossing.
    2. If the branch runs out of points before crossing (short branch), extrapolate
       a ray from its own trailing tangent direction until it hits the mesh.
    3. Last resort: nearest point on the mesh surface to pts[0].

    Returns (endpoint [3] float32, tangent [3] float32 unit vector,
             used_fallback bool, used_elongation bool).
    """
    cum = _arc_length_cumulative(pts)
    n = int(np.searchsorted(cum, search_window_mm)) + 1
    n = max(2, min(n, len(pts)))

    hit = _forward_crossing(mesh, pts[:n], cum[:n])
    if hit is not None:
        endpoint, path_pts = hit
        return endpoint, _average_tangent(path_pts), False, False

    tail = pts[-min(tangent_tail_points, len(pts)):]
    tail_dir = _unit(tail[-1] - tail[0])
    if np.linalg.norm(tail_dir) > 1e-8:
        locs, ray_idx, _ = mesh.ray.intersects_location(
            ray_origins=pts[-1:], ray_directions=tail_dir[None, :])
        if len(locs) > 0:
            dists = np.linalg.norm(locs - pts[-1], axis=1)
            endpoint = locs[np.argmin(dists)].astype(np.float32)
            path_pts = np.concatenate([pts, endpoint[None]], axis=0)
            return endpoint, _average_tangent(path_pts), False, True

    closest, _, _ = trimesh.proximity.closest_point(mesh, pts[:1])
    endpoint = closest[0].astype(np.float32)
    # Fallback tangent: the direction to an arbitrary nearest-surface point can
    # point anywhere (perpendicular, even backward) relative to the branch, so
    # use the branch's own observed initial direction instead of the
    # direction-to-endpoint here.
    tangent = _unit(pts[1] - pts[0]).astype(np.float32)
    return endpoint, tangent, True, False


def _mesh_edge_graph(mesh):
    """Undirected, edge-length-weighted sparse adjacency matrix over mesh
    vertices — built once per mesh, reused for every branch's Dijkstra call."""
    edges = mesh.edges_unique
    lengths = mesh.edges_unique_length
    n = len(mesh.vertices)
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.concatenate([lengths, lengths]).astype(np.float64)
    return coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()


def graph_distance_from_point(mesh, graph, point):
    """Graph (geodesic, edge-length-weighted) distance from `point` (assumed
    on/near the surface) to every mesh vertex — snaps to the nearest vertex,
    runs Dijkstra from there over mesh edges weighted by edge length, then
    adds back the small snap offset so distances are measured from `point`
    itself, not the snapped vertex. Returns per-vertex distances, float64 [N].
    """
    verts = mesh.vertices
    seed = int(np.argmin(np.linalg.norm(verts - point, axis=1)))
    offset = float(np.linalg.norm(verts[seed] - point))
    return dijkstra(graph, directed=False, indices=seed) + offset


def process_checkpoint(chk, max_branches=MAX_BRANCHES, search_window_mm=SEARCH_WINDOW_MM,
                       patch_threshold_mm=PATCH_THRESHOLD_MM):
    verts, faces = _reconstruct_ghd_numpy(chk, denormalize_shape=True)
    mesh = trimesh.Trimesh(verts, faces, process=False)
    graph = _mesh_edge_graph(mesh)

    branch_points = chk["clipped_centerline"]["branch_points"][:max_branches]
    endpoints = np.zeros((max_branches, 3), dtype=np.float32)
    tangents = np.zeros((max_branches, 3), dtype=np.float32)
    branch_mask = np.zeros(max_branches, dtype=bool)
    used_fallback = np.zeros(max_branches, dtype=bool)
    used_elongation = np.zeros(max_branches, dtype=bool)
    patch_distances = np.full((max_branches, len(verts)), np.inf, dtype=np.float32)
    in_patch = np.zeros((max_branches, len(verts)), dtype=bool)

    for b, pts in enumerate(branch_points):
        pts = np.asarray(pts, dtype=np.float32)
        if len(pts) < 2:
            continue
        endpoint, tangent, fallback, elongated = find_branch_endpoint(mesh, pts, search_window_mm)
        endpoints[b] = endpoint
        tangents[b] = tangent
        branch_mask[b] = True
        used_fallback[b] = fallback
        used_elongation[b] = elongated

        dist = graph_distance_from_point(mesh, graph, endpoint)
        patch_distances[b] = dist.astype(np.float32)
        in_patch[b] = dist <= patch_threshold_mm

    return {
        "case": chk["case"],
        "aneurysm_type": int(chk["aneurysm_type"]),
        "phi": np.asarray(chk["ghd"]["phi"], dtype=np.float32),
        "endpoints": endpoints,
        "tangents": tangents,
        "branch_mask": branch_mask,
        "patch_distances": patch_distances,
        "in_patch": in_patch,
        "used_fallback": used_fallback,
        "used_elongation": used_elongation,
    }, verts, faces, branch_points


def visualize_sample(record, verts, faces, branch_points, save_path):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces,
                    color="lightgray", edgecolor="none", alpha=0.25)

    colors = ["black", "blue", "red"]
    all_pts = [verts]
    for b, pts in enumerate(branch_points):
        pts = np.asarray(pts, dtype=np.float32)
        if not record["branch_mask"][b]:
            continue
        c = colors[b % len(colors)]
        near = pts[:60]
        ax.plot(near[:, 0], near[:, 1], near[:, 2], color=c, lw=2.0, ls=(0, (4, 2)))
        ax.scatter(*pts[0], s=30, marker="o", color=c, label=f"b{b} pts[0]")

        if record["used_fallback"][b]:
            marker, tag = "^", " (fallback)"
        elif record["used_elongation"][b]:
            marker, tag = "s", " (elongated)"
        else:
            marker, tag = "x", ""
        endpoint = record["endpoints"][b]
        ax.scatter(*endpoint, s=80, marker=marker, color=c, edgecolor="black",
                   label=f"b{b} endpoint{tag}")

        patch_verts = verts[record["in_patch"][b]]
        if len(patch_verts) > 0:
            ax.scatter(patch_verts[:, 0], patch_verts[:, 1], patch_verts[:, 2],
                      s=10, color=c, alpha=0.5, label=f"b{b} patch ({len(patch_verts)}v)")
            all_pts.append(patch_verts)

        tangent = record["tangents"][b]
        arrow_end = endpoint + tangent * TANGENT_ARROW_MM
        ax.quiver(*endpoint, *(tangent * TANGENT_ARROW_MM), color="yellow", linewidth=2.0,
                  arrow_length_ratio=0.3)
        all_pts += [near, endpoint[None], arrow_end[None]]

    ax.set_title(f"{record['case']} (type {record['aneurysm_type']})")
    ax.set_axis_off()
    _set_axes_equal(ax, *all_pts)
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--max-branches", type=int, default=MAX_BRANCHES)
    parser.add_argument("--search-window-mm", type=float, default=SEARCH_WINDOW_MM)
    parser.add_argument("--patch-threshold-mm", type=float, default=PATCH_THRESHOLD_MM,
                        help="Graph-distance radius (mm, edge-length-weighted) defining each "
                             "branch's cap patch around its endpoint.")
    parser.add_argument("--save-sanity", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = ([input_dir / f"{c}.npy" for c in args.cases] if args.cases
             else sorted(input_dir.glob("*.npy")))

    n_branches = n_fallback = n_elongated = n_saved = n_failed = 0
    patch_sizes = []
    for path in paths:
        if not path.exists():
            print(f"[SKIP] {path} not found")
            continue
        chk = np.load(path, allow_pickle=True).item()
        try:
            record, verts, faces, branch_points = process_checkpoint(
                chk, args.max_branches, args.search_window_mm, args.patch_threshold_mm)
        except Exception as exc:
            print(f"[FAIL] {chk.get('case', path.stem)}: {exc}")
            n_failed += 1
            continue

        np.save(output_dir / f"{record['case']}.npy", record, allow_pickle=True)
        n_saved += 1
        n_branches += int(record["branch_mask"].sum())
        n_fallback += int(record["used_fallback"].sum())
        n_elongated += int(record["used_elongation"].sum())
        for b in range(args.max_branches):
            if record["branch_mask"][b]:
                patch_sizes.append(int(record["in_patch"][b].sum()))

        if args.save_sanity:
            visualize_sample(record, verts, faces, branch_points,
                             output_dir / "sanity" / f"{record['case']}_endpoints.png")

    print(f"Saved {n_saved} checkpoints to {output_dir} ({n_failed} failed).")
    if n_branches:
        print(f"Endpoints: {n_branches} total, {n_elongated} elongated "
              f"({100 * n_elongated / n_branches:.1f}%), {n_fallback} used the nearest-surface "
              f"fallback ({100 * n_fallback / n_branches:.1f}%).")
    if patch_sizes:
        patch_sizes = np.array(patch_sizes)
        print(f"Patch size (vertices) at {args.patch_threshold_mm}mm: "
              f"mean={patch_sizes.mean():.1f} median={np.median(patch_sizes):.0f} "
              f"min={patch_sizes.min()} max={patch_sizes.max()}")


if __name__ == "__main__":
    main()
