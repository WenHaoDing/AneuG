"""
Automatic endcap labelling by centerline-tangent plane clipping.

The idea: we are clipping the WARPED CANONICAL mesh, so unlike a raw
segmentation we already know its topology -- which vertices are dome, which
are neck, and which ring belongs to which opening (canonical_topology.npy,
whose indices stay valid under GHD deformation since the topology is fixed).
That makes a purely geometric cut safe:

  1. Each opening gets a cutting plane. Its NORMAL is the tangent at the start
     of that opening's clipped centerline branch, and its POINT is the branch's
     start vertex -- so the plane sits across the vessel exactly where the real
     centerline begins, which is the opening itself.

  2. The mesh is pre-divided into n regions (n = number of openings) by graph
     distance from each opening ring. A vertex belongs to the opening it is
     closest to along the surface.

  3. Clipping happens ONLY inside the matching region, and the dome vertex set
     is excluded outright. A plane is an infinite object: without this, a
     branch whose tangent happens to point back across the sac would slice the
     dome off. Region + dome protection makes that impossible rather than
     unlikely.

  4. Whatever is left beyond the plane, taken as the connected component
     touching that opening's ring, is the cap.

Why not just use the canonical ring indices directly: they drift under
deformation, which is what motivated the learned endcap predictor in the first
place (see dataset/label_morpho.py). The ring is used here only as a SEED --
for region assignment and component selection -- never as the cut itself, so
drift costs nothing.

Debug run:
    python dataset/auto_uncap.py --limit 6
writes per-case checkpoints and renders to runtime/debug_auto_uncap/.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

CANONICAL_ROOT = _REPO_ROOT / "dataset" / "canonical"
CANONICAL_BY_TYPE = {0: CANONICAL_ROOT / "Bifurcated",
                     1: CANONICAL_ROOT / "Sidewall",
                     2: CANONICAL_ROOT / "Sidewall"}
DEFAULT_PROCESSED = _REPO_ROOT / "runtime_dataset" / "AneuG_processed"
DEFAULT_ASSEMBLED = _REPO_ROOT / "runtime_dataset" / "assembled"
DEFAULT_OUT = _REPO_ROOT / "runtime" / "debug_auto_uncap"

# How far along the branch to measure the tangent. The first centerline points
# are the noisiest (they sit right at the clip), so the tangent is fitted over
# a short span rather than taken from the first segment alone.
TANGENT_SPAN = 2.0


def load_topology(aneurysm_type):
    t = np.load(CANONICAL_BY_TYPE[int(aneurysm_type)] / "canonical_topology.npy",
                allow_pickle=True).item()
    rings = [np.asarray(o["indices"], dtype=np.int64) for o in t["openings"]]
    return rings, np.asarray(t["dome_vertex_indices"], dtype=np.int64)


def branch_tangent(points, span=TANGENT_SPAN):
    """Unit tangent at the start of a branch, fitted over the first `span` of
    arc length. Least squares over a span rather than a single difference: the
    first segment alone is short and noisy."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        raise ValueError("branch has fewer than 2 points")
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    head = pts[cum <= span] if cum[-1] > span else pts
    if len(head) < 2:
        head = pts[:2]
    centred = head - head.mean(0)
    d = np.linalg.svd(centred, full_matrices=False)[2][0]
    if d @ (head[-1] - head[0]) < 0:
        d = -d
    return d / (np.linalg.norm(d) + 1e-12)


def surface_crossing(mesh, points, max_span=6.0, step=0.02, extend=4.0):
    """Where a branch leaves the mesh, walking the centerline outward.

    Normally the branch starts INSIDE the capped mesh and exits through the
    cap, so the first inside->outside flip along the polyline is that exit.
    Two cases have no such flip on the polyline as given, and both are fixed
    by EXTENDING the centerline rather than giving up:

      * the branch never leaves the mesh within its own length (a short stub
        entirely inside) -- extend FORWARD past the last point along the
        tangent at the end until it breaks the surface;

      * the branch already starts outside (its first point lies beyond the
        cap) -- extend BACKWARD before the first point along the reversed
        start tangent, back into the vessel, so the scan begins inside.

    Both are handled by building one extended polyline and scanning it
    outward for the first inside->outside flip, which is the cap either way.
    The crossing is then bisected onto the straddling segment.

    Returns (crossing_point, signed distance from points[0], mode). The
    distance is NEGATIVE when the crossing lies behind the start. mode is one
    of "direct", "extended_forward", "extended_backward", or None when no
    crossing exists even after extending.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return None, None, None
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    span = min(float(cum[-1]), max_span)
    if span <= 0:
        return None, None, None

    n = max(int(span / step) + 1, 3)
    t = np.linspace(0.0, span, n)
    dense = np.stack([np.interp(t, cum, pts[:, d]) for d in range(3)], axis=1)

    head_tan = branch_tangent(pts)
    end_tan = branch_tangent(pts[::-1][:max(len(pts) // 4, 2)][::-1])
    n_ext = max(int(extend / step), 1)
    back = pts[0][None] - head_tan[None] * np.linspace(extend, step, n_ext)[:, None]
    fwd = dense[-1][None] + end_tan[None] * np.linspace(step, extend, n_ext)[:, None]

    full = np.vstack([back, dense, fwd])
    origin = len(back)                       # index of points[0] within `full`
    inside = mesh.contains(full)
    flips = np.flatnonzero(inside[:-1] & ~inside[1:])
    if not len(flips):
        return None, None, None
    i = int(flips[0])

    lo, hi = full[i], full[i + 1]
    for _ in range(24):                      # bisect onto the surface
        mid = 0.5 * (lo + hi)
        if mesh.contains(mid[None])[0]:
            lo = mid
        else:
            hi = mid
    cross = 0.5 * (lo + hi)

    mode = "direct" if origin <= i < origin + len(dense) - 1 else (
        "extended_backward" if i < origin else "extended_forward")
    d = float(np.linalg.norm(cross - pts[0])) * (1.0 if i >= origin else -1.0)
    return cross, d, mode


def vertex_graph(verts, faces):
    """Sparse edge-length graph over the mesh, for surface distances."""
    from scipy.sparse import coo_matrix

    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    w = np.linalg.norm(verts[e[:, 0]] - verts[e[:, 1]], axis=1)
    n = len(verts)
    return coo_matrix((np.concatenate([w, w]),
                       (np.concatenate([e[:, 0], e[:, 1]]),
                        np.concatenate([e[:, 1], e[:, 0]]))), shape=(n, n)).tocsr()


def dijkstra_from(graph, sources):
    from scipy.sparse.csgraph import dijkstra
    return dijkstra(graph, indices=np.asarray(sources), min_only=True)


def partition_by_opening(graph, rings):
    """Assign every vertex to the opening whose ring is closest along the
    surface. Returns (labels, dists[n_openings, n_verts])."""
    dists = np.stack([dijkstra_from(graph, r) for r in rings])
    return np.argmin(dists, axis=0), dists


def cap_for_opening(verts, graph, ring, plane_pt, normal, allowed):
    """Cap vertices beyond `plane_pt` along `normal`, restricted to `allowed`
    and grown from `ring` so only the piece attached to this opening is taken."""
    from scipy.sparse.csgraph import connected_components

    signed = (verts - plane_pt[None]) @ normal
    cand = (signed > 0) & allowed
    mask = np.zeros(len(verts), dtype=bool)
    if not cand.any():
        return mask, signed

    idx = np.flatnonzero(cand)
    sub = graph[idx][:, idx]
    _, lab = connected_components(sub, directed=False)
    ring_in = ring[cand[ring]]
    if len(ring_in):
        keep = np.unique(lab[np.searchsorted(idx, ring_in)])
    else:
        # ring entirely inboard of the plane -- take the component nearest the
        # ring rather than silently accepting every candidate
        d = dijkstra_from(graph, ring)
        keep = [lab[int(np.argmin(d[idx]))]]
    mask[idx[np.isin(lab, keep)]] = True
    return mask, signed


def compute_caps(record, verts, faces, plane_frac=0.5):
    """Cap labels for an already-loaded warped canonical mesh.

    Split out from uncap_case so callers that already hold the geometry --
    dataset/label_morpho.py reconstructs it from phi -- can get the automatic
    labels without a mesh file or an output directory.

    Returns (caps [n_verts] bool, cap_id [n_verts] int, info list).
    """
    import trimesh

    V = np.asarray(verts, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
    rings, dome = load_topology(record["aneurysm_type"])
    branches = [np.asarray(b, dtype=np.float64)
                for b in record["clipped_centerline"]["branch_points"]]
    if len(branches) != len(rings):
        raise ValueError(f"{len(branches)} centerline branches vs {len(rings)} "
                         f"canonical openings -- cannot pair them")

    graph = vertex_graph(V, F)
    labels, _ = partition_by_opening(graph, rings)
    dome_mask = np.zeros(len(V), dtype=bool)
    dome_mask[dome] = True

    caps = np.zeros(len(V), dtype=bool)
    cap_id = np.full(len(V), -1, dtype=np.int64)
    info = []
    for i, (ring, br) in enumerate(zip(rings, branches)):
        normal = branch_tangent(br)
        cross, cross_d, cross_mode = surface_crossing(mesh, br)
        if cross is not None:
            plane_pt = (1.0 - plane_frac) * br[0] + plane_frac * cross
        else:
            plane_pt = br[0]
        allowed = (labels == i) & ~dome_mask       # region i, dome never clipped
        mask, signed = cap_for_opening(V, graph, ring, plane_pt, normal, allowed)
        caps |= mask
        cap_id[mask] = i
        info.append({"opening": i, "n_cap_verts": int(mask.sum()),
                     "normal": normal.tolist(), "plane_point": np.asarray(plane_pt).tolist(),
                     "ring_frac_beyond_plane": float((signed[ring] > 0).mean()),
                     "region_verts": int((labels == i).sum()),
                     "dome_hits": int((mask & dome_mask).sum()),
                     "surface_crossing_dist": cross_d,
                     "crossing_mode": cross_mode,
                     "used_crossing": cross is not None})
    return caps, cap_id, info, labels, dome_mask


def uncap_case(record, mesh_path, out_dir, save_render=True, plane_frac=0.5):
    """Label the cap regions of one warped canonical mesh, from a mesh file."""
    import trimesh

    mesh = trimesh.load(mesh_path, process=False)
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int64)
    rings, dome = load_topology(record["aneurysm_type"])
    branches = [np.asarray(b, dtype=np.float64)
                for b in record["clipped_centerline"]["branch_points"]]
    if len(branches) != len(rings):
        raise ValueError(f"{len(branches)} centerline branches vs {len(rings)} "
                         f"canonical openings -- cannot pair them")

    caps, cap_id, info, labels, dome_mask = compute_caps(record, V, F, plane_frac)

    out = {"case": record["case"], "dataset": record.get("dataset", ""),
           "aneurysm_type": int(record["aneurysm_type"]),
           "cap_vertex_mask": caps, "cap_vertex_opening": cap_id,
           "cap_face_mask": caps[F].all(axis=1), "region_labels": labels,
           "openings": info, "mesh": str(mesh_path),
           "plane_frac": float(plane_frac)}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{record['case']}_caps.npy", out, allow_pickle=True)
    if save_render:
        render(V, F, caps, cap_id, branches, rings, dome_mask, record,
               out_dir / "sanity" / f"{record['case']}.png")
    return out


def render(V, F, caps, cap_id, branches, rings, dome_mask, record, save_path):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    colors = ["#e6194B", "#3cb44b", "#4363d8"]
    # Three views rather than two: a cap that is under- or over-cut on one
    # side can hide behind the body from any single pair of angles. 120 deg
    # apart in azimuth plus a raised third view covers the mesh.
    views = ((20, 30), (20, 150), (55, 270))
    fig = plt.figure(figsize=(19, 6.5))
    for panel, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, len(views), panel + 1, projection="3d")
        body = ~caps[F].any(axis=1)
        ax.plot_trisurf(V[:, 0], V[:, 1], V[:, 2], triangles=F[body],
                        color="lightgray", edgecolor="none", alpha=0.35)
        ax.scatter(*V[dome_mask].T, s=1.5, color="#888888", alpha=0.30,
                   label="dome (protected)")
        for i in range(len(rings)):
            m = caps & (cap_id == i)
            if m.any():
                ax.scatter(*V[m].T, s=9, color=colors[i % 3], label=f"cap {i} ({m.sum()}v)")
            b = branches[i]
            ax.plot(*b[:60].T, color=colors[i % 3], lw=1.6, alpha=0.9)
            ax.scatter(*b[0], s=40, color=colors[i % 3], edgecolor="black", zorder=6)
        lo, hi = V.min(0), V.max(0)
        c, r = (lo + hi) / 2, float((hi - lo).max() / 2)
        ax.set_xlim(c[0]-r, c[0]+r); ax.set_ylim(c[1]-r, c[1]+r); ax.set_zlim(c[2]-r, c[2]+r)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_title(f"elev {elev}  azim {azim}", fontsize=7)
        if panel == 0:
            ax.legend(loc="upper left", fontsize=7)
    fig.suptitle(f"{record['case']}  [{record.get('dataset','')}]  "
                 f"type {record['aneurysm_type']}  |  caps: {int(caps.sum())} verts, "
                 f"{int(caps[F].all(axis=1).sum())} faces", fontsize=10)
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Automatic endcap labelling by "
                                             "centerline-tangent plane clipping.")
    ap.add_argument("--processed-root", default=str(DEFAULT_PROCESSED))
    ap.add_argument("--assembled-root", default=str(DEFAULT_ASSEMBLED))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--case", action="append", dest="cases")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--plane-frac", type=float, default=0.5,
                    help="Where the cutting plane sits between the centerline start (0) "
                         "and the point where the centerline leaves the mesh (1). "
                         "Higher clips less. Default 0.5 (the midpoint).")
    ap.add_argument("--no-render", dest="render", action="store_false")
    ap.set_defaults(render=True)
    args = ap.parse_args()

    proc = Path(args.processed_root)
    paths = ([proc / f"{c}.npy" for c in args.cases] if args.cases
             else sorted(proc.glob("*.npy"))[:args.limit])
    ok = fail = 0
    for p in paths:
        rec = np.load(p, allow_pickle=True).item()
        mesh = Path(args.assembled_root) / rec["dataset"] / rec["case"] / "ghd_fitted.obj"
        try:
            out = uncap_case(rec, mesh, args.out_dir, save_render=args.render,
                             plane_frac=args.plane_frac)
            print(f"[OK] {rec['case']:<36} caps={int(out['cap_vertex_mask'].sum()):5d}v "
                  f"{int(out['cap_face_mask'].sum()):5d}f  per-opening="
                  f"{[o['n_cap_verts'] for o in out['openings']]}  "
                  f"ring_beyond={[round(o['ring_frac_beyond_plane'],2) for o in out['openings']]}")
            ok += 1
        except Exception as exc:
            print(f"[FAIL] {rec['case']}: {exc}")
            fail += 1
    print(f"\n{ok} ok, {fail} failed -> {args.out_dir}")


if __name__ == "__main__":
    main()
