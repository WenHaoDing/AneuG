"""Dense 1D vessel graph (centerline + radius) from a surface mesh and its endpoints.

Local copy of the AneuSeg centerline steps (IAgents/tools/centerline_extraction.py:
vmtk_extract_centerline, vmtkBranchExtractor), so this repo does not import AneuSeg. Unlike
merge_centerline_v3, the bifurcation groups (Blanking = 1) are kept as edges, so the graph is
connected. Run in an env with vmtk (vmtk310 or vmtk_autogen).

    from geometry import extract_skeleton
    graph = extract_skeleton("patched.ply", endpoints)                  # endpoints (n, 3)
    graph = extract_skeleton("patched.ply", endpoints, zone_pts=zone)   # + radius interpolated
                                                                        #   across the aneurysm

    python geometry/extract_skeleton.py --mesh m.ply --endpoints endpoints_manual.npy \
        [--zone zone_pts.npy] --out skeleton_graph.npy

graph = dict(
  nodes  (N, 3): branch ends, bifurcation points and vessel ends
  edges  list of dict:
           u, v          node ids
           kind          "branch" | "bifurcation" (the short piece through a bifurcation)
           group         vmtkBranchExtractor GroupId
           pts (n, 3)    dense points, 0.1 mm apart, corners at the nodes rounded
                         (smooth_pos_mm, default 0.5 mm; vessel ends fixed)
           pts_extracted the points before that smoothing
           radius (n,)   radius used as the condition: MaximumInscribedSphereRadius, interpolated
                         across the aneurysm (when zone_pts is given), then smoothed over the
                         whole graph (smooth_mm, default 1 mm)
           radius_interp before smoothing
           radius_raw    before interpolation
           aneurysm      bool per point: inside the interpolated region
           extracted     bool per point: False only for an added node point (a straight
                         connector where two VMTK lines meet a node apart from each other)
  meta   dict: source_id, endpoints, continuity check)
"""
import argparse
import json

import numpy as np

RADIUS = "MaximumInscribedSphereRadius"


# ------------------------------------------------------------------ centerlines (AneuSeg copy)

def vmtk_centerlines(surface, endpoints, source_id=0, resample_mm=0.1, smooth_iters=100,
                     smooth_factor=0.1):
    """Same settings as AneuSeg vmtk_extract_centerline: source = endpoints[source_id]."""
    from vmtk import vmtkscripts
    cl = vmtkscripts.vmtkCenterlines()
    cl.Surface = surface
    cl.SeedSelectorName = "pointlist"
    cl.SourcePoints = list(map(float, endpoints[source_id]))
    cl.TargetPoints = [float(x) for j, pt in enumerate(endpoints) if j != source_id for x in pt]
    cl.AppendEndPoints = 1
    cl.Execute()
    rs = vmtkscripts.vmtkCenterlineResampling()
    rs.Centerlines = cl.Centerlines
    rs.Length = resample_mm
    rs.Execute()
    sm = vmtkscripts.vmtkCenterlineSmoothing()
    sm.Centerlines = rs.Centerlines
    sm.NumberOfSmoothingIterations = smooth_iters
    sm.SmoothFactor = smooth_factor
    sm.Execute()
    return sm.Centerlines


def to_surface(mesh):
    """Path, trimesh-like (vertices/faces) or vtkPolyData -> vtkPolyData."""
    import pyvista as pv
    if isinstance(mesh, str) or hasattr(mesh, "__fspath__"):
        from vmtk import vmtkscripts
        r = vmtkscripts.vmtkSurfaceReader()
        r.InputFileName = str(mesh)
        r.Execute()
        return r.Surface
    if hasattr(mesh, "vertices") and hasattr(mesh, "faces"):
        f = np.asarray(mesh.faces)
        return pv.PolyData(np.asarray(mesh.vertices, float),
                           np.hstack([np.full((len(f), 1), 3), f]).ravel())
    return mesh


# ------------------------------------------------------------------ aneurysm radius

def aneurysm_region(pts, r, zone_tree, zone_pts, tol=0.35, ext_factor=0.75):
    """Points whose inscribed sphere comes within tol of the zone, plus the points the zone
    projects onto (within 2.5 r); gaps shorter than r filled; widened by ext_factor * r."""
    from scipy.ndimage import label as cc_label
    from scipy.spatial import cKDTree
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    d, _ = zone_tree.query(pts)
    mark = d < r + tol
    dz, j = cKDTree(pts).query(zone_pts)
    mark[j[dz < 2.5 * r[j] + 0.5]] = True
    idx = np.flatnonzero(mark)
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a > 1 and s[b] - s[a] < np.median(r[a:b + 1]):
            mark[a:b] = True
    lab, n = cc_label(mark)
    region = np.zeros(len(pts), bool)
    for k in range(1, n + 1):
        seg = np.flatnonzero(lab == k)
        ext = max(0.5, ext_factor * np.median(r[max(seg[0] - 20, 0):seg[-1] + 20]))
        region[np.searchsorted(s, s[seg[0]] - ext):np.searchsorted(s, s[seg[-1]] + ext, side="right")] = True
    return region, s


def interpolate_radius(r, s, region, fit_mm=1.0):
    """C1 Hermite across each region between the healthy anchors (slopes from a 1 mm line fit);
    a region reaching the end of a line is held at the anchor radius."""
    from scipy.ndimage import label as cc_label
    out = r.copy()
    lab, n = cc_label(region)
    for k in range(1, n + 1):
        seg = np.flatnonzero(lab == k)
        a, b = seg[0] - 1, seg[-1] + 1
        if a < 0 and b >= len(r):
            continue
        if a < 0 or b >= len(r):
            out[seg] = r[b if a < 0 else a]
            continue

        def slope(i, direction):
            w = ((s >= s[i] - fit_mm) & (s <= s[i])) if direction < 0 else ((s >= s[i]) & (s <= s[i] + fit_mm))
            w &= ~region
            return float(np.polyfit(s[w], r[w], 1)[0]) if w.sum() >= 3 else 0.0

        L = s[b] - s[a]
        t = (s[a:b + 1] - s[a]) / L
        m0, m1 = slope(a, -1) * L, slope(b, +1) * L
        out[a:b + 1] = ((2 * t ** 3 - 3 * t ** 2 + 1) * r[a] + (t ** 3 - 2 * t ** 2 + t) * m0 +
                        (-2 * t ** 3 + 3 * t ** 2) * r[b] + (t ** 3 - t ** 2) * m1)
        out[a:b + 1] = np.clip(out[a:b + 1], 0.3 * min(r[a], r[b]), None)
    return out


def correct_radius(cl, zone_pts):
    """Interpolate the radius across the aneurysm on every source->target line, in place.
    Adds point arrays RadiusRaw and AneurysmRegion."""
    import pyvista as pv
    from scipy.spatial import cKDTree
    poly = pv.wrap(cl)
    r_all = np.array(poly.point_data[RADIUS], float)
    r_new, flag = r_all.copy(), np.zeros(len(r_all), np.uint8)
    if zone_pts is not None:
        tree = cKDTree(zone_pts)
        for ci in range(poly.n_cells):
            ids = np.array(poly.get_cell(ci).point_ids)
            region, s = aneurysm_region(poly.points[ids], r_all[ids], tree, zone_pts)
            r_new[ids] = interpolate_radius(r_all[ids], s, region)
            flag[ids] |= region.astype(np.uint8)
    poly.point_data["RadiusRaw"] = r_all
    poly.point_data["AneurysmRegion"] = flag
    poly.point_data[RADIUS] = r_new
    return poly


# ------------------------------------------------------------------ graph

def branch_split(cl):
    import pyvista as pv
    from vmtk import vmtkscripts
    be = vmtkscripts.vmtkBranchExtractor()
    be.Centerlines = cl
    be.RadiusArrayName = RADIUS
    be.Execute()
    return pv.wrap(be.Centerlines)


def build_graph(b):
    """Graph from vmtkBranchExtractor's own topology (no distance tolerance).

    Consecutive tracts on one VMTK line are connected. Every branch group becomes an edge (its
    longest cell, as merge_centerline_v3). Every bifurcation group (Blanking = 1) becomes a
    node joining its upstream branch, plus one edge per downstream group: the bifurcation piece,
    taken from the same VMTK line as that downstream branch so the two meet without a gap.
    Where two different lines meet at a node, the node is joined by a straight connector."""
    from collections import defaultdict
    g = np.array(b.cell_data["GroupIds"])
    blank = np.array(b.cell_data["Blanking"]).astype(bool)
    line = np.array(b.cell_data["CenterlineIds"])
    tract = np.array(b.cell_data["TractIds"])
    ids_of = lambda c: np.array(b.get_cell(c).point_ids)
    length = lambda p: float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(p) > 1 else 0.0
    longest = lambda cells: max(cells, key=lambda c: (length(b.points[ids_of(c)]), -c))

    nxt = {}                                   # cell -> next cell on the same line
    by_line = defaultdict(list)
    for c in range(b.n_cells):
        by_line[line[c]].append((tract[c], c))
    for lst in by_line.values():
        lst.sort()
        for (_, c0), (_, c1) in zip(lst[:-1], lst[1:]):
            nxt[c0] = c1
    prv = {v: k for k, v in nxt.items()}
    succ, pred = defaultdict(set), defaultdict(set)
    for c0, c1 in nxt.items():
        succ[g[c0]].add(g[c1])
        pred[g[c1]].add(g[c0])

    groups = np.unique(g)
    gblank = {gid: bool(blank[np.flatnonzero(g == gid)[0]]) for gid in groups}
    rep = {gid: longest(np.flatnonzero(g == gid)) for gid in groups if not gblank[gid]}

    nodes, node_id = [], {}

    def node(key, pos):
        if key not in node_id:
            node_id[key] = len(nodes)
            nodes.append(np.asarray(pos, float))
        return node_id[key]

    def start_node(gid):
        """Node where branch group gid starts."""
        first = b.points[ids_of(rep[gid])[0]]
        ups = sorted(pred[gid])
        if not ups:
            return node(("leaf_start", gid), first)
        u = ups[0]
        if gblank[u]:
            return node(("start", gid), first)            # a bifurcation piece ends here
        return node(("chain", u, gid), b.points[ids_of(rep[u])[-1]])

    def end_node(gid):
        last = b.points[ids_of(rep[gid])[-1]]
        downs = sorted(succ[gid])
        if not downs:
            return node(("leaf_end", gid), last)
        d = downs[0]
        if gblank[d]:
            return node(("bif", d), last)                 # the bifurcation's junction
        return node(("chain", gid, d), last)

    raw = []                                   # (cell, kind, group, u, v)
    for gid in groups:
        if not gblank[gid]:
            raw.append((rep[gid], "branch", gid, start_node(gid), end_node(gid)))
    for gid in groups:
        if not gblank[gid]:
            continue
        cells = np.flatnonzero(g == gid)
        ups = [u for u in sorted(pred[gid]) if not gblank[u]]
        if ups:
            j = end_node(ups[0])
        else:
            j = node(("bif", gid), b.points[ids_of(longest(cells))[0]])
        for d in sorted(succ[gid]):
            if gblank[d]:
                continue                                   # bifurcation followed by bifurcation: rare
            into_d = [c for c in cells if c in nxt and g[nxt[c]] == d]
            same_line = [c for c in into_d if line[c] == line[rep[d]]]
            c = longest(same_line or into_d)
            raw.append((c, "bifurcation", gid, j, start_node(d)))

    nodes = np.asarray(nodes)
    keys = ("radius", "radius_raw", "aneurysm")
    edges = []
    for c, kind, gid, u, v in raw:
        ids = ids_of(c)
        edges.append(dict(kind=kind, group=int(gid), u=int(u), v=int(v),
                          pts=np.asarray(b.points[ids], float),
                          radius=np.asarray(b.point_data[RADIUS])[ids],
                          radius_raw=np.asarray(b.point_data["RadiusRaw"])[ids],
                          aneurysm=np.asarray(b.point_data["AneurysmRegion"])[ids].astype(bool),
                          extracted=np.ones(len(ids), bool)))
    # values at a node: those of the extracted point that sits exactly on it
    at_node = {}
    for e in edges:
        for n, i in ((e["u"], 0), (e["v"], -1)):
            if n not in at_node and np.linalg.norm(nodes[n] - e["pts"][i]) <= 1e-6:
                at_node[n] = {k: e[k][i] for k in keys}
    for e in edges:
        # straight connector where this VMTK line ends apart from the node; the added node point
        # carries the node's own radius, so radius changes along the connector, not at the node
        for n, front in ((e["u"], True), (e["v"], False)):
            end = e["pts"][0] if front else e["pts"][-1]
            if np.linalg.norm(nodes[n] - end) <= 1e-6:
                continue
            val = at_node.get(n, {k: e[k][0 if front else -1] for k in keys})
            # a line that ends apart from the node often runs alongside the other branch first;
            # cut it back to its point closest to the node so the connector cannot fold back
            half = max(len(e["pts"]) // 2, 1)
            if front:
                k0 = int(np.argmin(np.linalg.norm(e["pts"][:half] - nodes[n], axis=1)))
                cut = slice(k0, None)
            else:
                k0 = int(np.argmin(np.linalg.norm(e["pts"][-half:][::-1] - nodes[n], axis=1)))
                cut = slice(None, len(e["pts"]) - k0)
            for k in keys + ("pts", "extracted"):
                e[k] = e[k][cut]
            if np.linalg.norm(nodes[n] - (e["pts"][0] if front else e["pts"][-1])) <= 1e-6:
                continue
            # the connector is filled at the same 0.1 mm spacing as the rest of the graph; radius
            # (and the region flag) go linearly from the node's value to the line's
            end = e["pts"][0] if front else e["pts"][-1]
            m = max(int(np.ceil(np.linalg.norm(end - nodes[n]) / 0.1)), 1)
            t = np.arange(m) / m                                  # node ... just before the line
            seg = nodes[n][None] + t[:, None] * (end - nodes[n])[None]
            e["pts"] = np.vstack([seg, e["pts"]] if front else [e["pts"], seg[::-1]])
            for k in keys:
                a0, a1 = val[k], (e[k][0] if front else e[k][-1])
                fill = (a0 + t * (a1 - a0)) if k != "aneurysm" else np.full(m, bool(a0) and bool(a1))
                e[k] = np.r_[fill, e[k]] if front else np.r_[e[k], fill[::-1]]
            e["extracted"] = np.r_[np.zeros(m, bool), e["extracted"]] if front else np.r_[e["extracted"], np.zeros(m, bool)]
        e["length_mm"] = length(e["pts"])
    # cutting a line back to its node can leave an edge of a single point (a zero-length
    # self-loop at that node, seen in AneuX C0084_cut2); it carries no geometry and breaks
    # the smoothing operator, so drop it
    edges = [e for e in edges if len(e["pts"]) >= 2 and not (e["u"] == e["v"] and e["length_mm"] < 1e-6)]
    return drop_specks(dict(nodes=nodes, edges=edges))


def drop_specks(graph, max_mm=0.5):
    """Remove connected pieces shorter than max_mm in total (degenerate VMTK groups of a couple
    of points with no neighbours), and renumber the nodes."""
    n = len(graph["nodes"])
    par = list(range(n))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for e in graph["edges"]:
        par[find(e["u"])] = find(e["v"])
    total = {}
    for e in graph["edges"]:
        total[find(e["u"])] = total.get(find(e["u"]), 0.0) + e["length_mm"]
    keep_root = {r for r, L in total.items() if L >= max_mm}
    edges = [e for e in graph["edges"] if find(e["u"]) in keep_root]
    used = sorted({x for e in edges for x in (e["u"], e["v"])})
    remap = {o: i for i, o in enumerate(used)}
    for e in edges:
        e["u"], e["v"] = remap[e["u"]], remap[e["v"]]
    return dict(nodes=graph["nodes"][used], edges=edges,
                dropped_specks=len(graph["edges"]) - len(edges))


def diffusion_operator(graph):
    """Dense point graph: every edge point is a vertex, edges joined at their shared nodes
    (vertex id = node id). Returns per-edge vertex ids, vertex count, weights W and degree."""
    import scipy.sparse as sp
    n_nodes = len(graph["nodes"])
    index, nxt_id = [], n_nodes
    for e in graph["edges"]:
        k = len(e["pts"])
        index.append(np.r_[e["u"], np.arange(nxt_id, nxt_id + k - 2), e["v"]])
        nxt_id += k - 2
    rows, cols, w = [], [], []
    for e, ids in zip(graph["edges"], index):
        d = np.maximum(np.linalg.norm(np.diff(e["pts"], axis=0), axis=1), 0.05)  # near-duplicates
        rows += [ids[:-1], ids[1:]]
        cols += [ids[1:], ids[:-1]]
        w += [1 / d ** 2, 1 / d ** 2]
    W = sp.csr_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(nxt_id, nxt_id))
    return index, nxt_id, W, np.asarray(W.sum(1)).ravel()


def diffuse(graph, key, sigma_mm, pinned=()):
    """Gaussian-like smoothing of a per-point field over the graph (explicit heat diffusion for
    time sigma^2 / 2); values at `pinned` node ids are held fixed. Returns per-edge arrays."""
    index, n, W, deg = diffusion_operator(graph)
    x = np.zeros((n,) + np.shape(graph["edges"][0][key])[1:])
    for e, ids in zip(graph["edges"], index):
        x[ids] = e[key]
    pinned = np.asarray(sorted(pinned), int)
    held = x[pinned].copy()
    dt = 0.4 / deg.max()
    deg_ = deg.reshape((-1,) + (1,) * (x.ndim - 1))
    for _ in range(int(np.ceil(sigma_mm ** 2 / 2 / dt))):
        x = x + dt * (W @ x - deg_ * x)
        x[pinned] = held
    return [x[ids] for ids in index], x[:len(graph["nodes"])]


def smooth_positions(graph, sigma_mm=0.5):
    """Round the corners left where VMTK lines meet at a node (connectors, bifurcation
    pieces). Vessel ends are pinned so branches keep their length. Keeps pts_extracted."""
    if sigma_mm <= 0:
        return graph
    deg = np.bincount([x for e in graph["edges"] for x in (e["u"], e["v"])], minlength=len(graph["nodes"]))
    per_edge, nodes = diffuse(graph, "pts", sigma_mm, pinned=np.flatnonzero(deg == 1))
    for e, p in zip(graph["edges"], per_edge):
        e["pts_extracted"], e["pts"] = e["pts"], p
        e["length_mm"] = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
    graph["nodes"] = nodes
    graph["smooth_positions_mm"] = sigma_mm
    return graph


def smooth_radius(graph, sigma_mm=1.0):
    """Smooth the radius over the whole graph, so small segmentation-level fluctuations are
    removed everywhere alike and the interpolated aneurysm stretch does not stand out.
    Keeps the pre-smoothing radius as radius_interp."""
    if sigma_mm <= 0:
        return graph
    per_edge, _ = diffuse(graph, "radius", sigma_mm)
    for e, r in zip(graph["edges"], per_edge):
        e["radius_interp"], e["radius"] = e["radius"], r
    graph["smooth_sigma_mm"] = sigma_mm
    return graph


def continuity(graph):
    """Radius jumps: between consecutive points on an edge, and between edge ends at a node."""
    step = max(float(np.abs(np.diff(e["radius"])).max()) for e in graph["edges"] if len(e["radius"]) > 1)
    at_node = {}
    for e in graph["edges"]:
        at_node.setdefault(e["u"], []).append(e["radius"][0])
        at_node.setdefault(e["v"], []).append(e["radius"][-1])
    node_jump = {n: float(np.ptp(r)) for n, r in at_node.items() if len(r) > 1}
    conn = []
    for e in graph["edges"]:
        x = ~e["extracted"]
        for side in (slice(None), slice(None, None, -1)):
            run = np.argmax(~x[side]) if x[side][0] else 0
            if run:
                p = e["pts"][side][:run + 1]
                conn.append(float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()))
    deg = np.bincount([x for e in graph["edges"] for x in (e["u"], e["v"])], minlength=len(graph["nodes"]))
    par = list(range(len(graph["nodes"])))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for e in graph["edges"]:
        par[find(e["u"])] = find(e["v"])
    n_comp = len({find(i) for i in range(len(par))})
    return dict(n_nodes=len(graph["nodes"]), n_edges=len(graph["edges"]), components=n_comp,
                cycles=len(graph["edges"]) - len(graph["nodes"]) + n_comp,
                n_bifurcation_edges=sum(e["kind"] == "bifurcation" for e in graph["edges"]),
                degree_counts={int(k): int(v) for k, v in zip(*np.unique(deg, return_counts=True))},
                max_radius_step_on_edge_mm=round(step, 4),
                max_radius_jump_at_node_mm=round(max(node_jump.values()), 4) if node_jump else 0.0,
                n_connectors=len(conn), max_connector_mm=round(max(conn), 3) if conn else 0.0,
                dropped_specks=graph.get("dropped_specks", 0))


# ------------------------------------------------------------------ point graph (model input)

def to_point_graph(graph):
    """Flatten the graph into one point graph, the format saved for the models.

    Every dense point (0.1 mm apart) is a node. A branch is a maximal chain between
    bifurcations / vessel ends (each bifurcation piece joins the branch it leads into).
    Branch ids are given breadth-first from the VMTK source endpoint, so the root branch is 1.

    Saved by the CLI as <out>_points.npz (save_point_graph / load_point_graph).

      pos           float32 (N, 3)  mm, mesh frame (origin + ijk * spacing)
      radius        float32 (N,)    the condition radius (interpolated + smoothed)
      edge_index    int64   (2, E)  undirected point adjacency, both directions
      branch        int64   (N,)    branch id 1..B for every point
      branch_id     int64   (N,)    same, but 0 inside the aneurysm region
      branch_onehot float32 (N, B+1) one-hot of branch_id (column 0 = aneurysm)
      aneurysm      bool    (N,)    radius was interpolated here (training target side,
                                    not a model input)
      is_junction   bool    (N,)    bifurcation points (degree >= 3)
      is_end        bool    (N,)    vessel ends
      extracted     bool    (N,)    False for points filling a connector at a node
    """
    from collections import deque
    nodes, edges = graph["nodes"], graph["edges"]
    n_nodes = len(nodes)
    index, n = [], n_nodes
    for e in edges:
        k = len(e["pts"])
        index.append(np.r_[e["u"], np.arange(n, n + k - 2), e["v"]])
        n += k - 2
    pos, radius = np.zeros((n, 3)), np.zeros(n)
    aneurysm, extracted = np.zeros(n, bool), np.ones(n, bool)
    for e, ids in zip(edges, index):
        pos[ids], radius[ids] = e["pts"], e["radius"]
        aneurysm[ids] |= e["aneurysm"]
        extracted[ids[1:-1]] = e["extracted"][1:-1]
    pairs = np.concatenate([np.stack([ids[:-1], ids[1:]]) for ids in index], axis=1)
    edge_index = np.concatenate([pairs, pairs[::-1]], axis=1)

    deg = np.bincount([x for e in edges for x in (e["u"], e["v"])], minlength=n_nodes)
    incident = [[] for _ in range(n_nodes)]
    for k, e in enumerate(edges):
        incident[e["u"]].append(k)
        incident[e["v"]].append(k)
    # branches: union the two edges meeting at every degree-2 node
    par = list(range(len(edges)))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for nd in np.flatnonzero(deg == 2):
        a, b_ = incident[nd]
        par[find(a)] = find(b_)
    # breadth-first from the source end (the vessel end nearest the VMTK source endpoint)
    ends = np.flatnonzero(deg == 1)
    src = graph.get("meta", {}).get("endpoints")
    sid = graph.get("meta", {}).get("source_id", 0)
    root = ends[np.argmin(np.linalg.norm(nodes[ends] - src[sid], axis=1))] if src is not None and len(ends) else 0
    bid, parent_edge, seen, q = {}, {root: None}, {root}, deque([root])
    while q:
        nd = q.popleft()
        for k in incident[nd]:
            r = find(k)
            if r not in bid:
                bid[r] = len(bid) + 1
            other = edges[k]["v"] if edges[k]["u"] == nd else edges[k]["u"]
            if other not in seen:
                seen.add(other)
                parent_edge[other] = k
                q.append(other)
    branch = np.zeros(n, np.int64)
    for k, ids in enumerate(index):
        branch[ids[1:-1]] = bid.get(find(k), 0)
    for nd in range(n_nodes):                     # a node takes the branch it was reached from
        k = parent_edge.get(nd)
        k = incident[nd][0] if k is None and incident[nd] else k
        branch[nd] = bid.get(find(k), 0) if k is not None else 0
    branch_id = np.where(aneurysm, 0, branch)
    n_branch = int(branch.max())
    onehot = np.zeros((n, n_branch + 1), np.float32)
    onehot[np.arange(n), branch_id] = 1.0
    is_junction = np.zeros(n, bool)
    is_junction[np.flatnonzero(deg >= 3)] = True
    is_end = np.zeros(n, bool)
    is_end[ends] = True
    return dict(pos=pos.astype(np.float32), radius=radius.astype(np.float32),
                edge_index=edge_index.astype(np.int64), branch=branch, branch_id=branch_id,
                branch_onehot=onehot, aneurysm=aneurysm, is_junction=is_junction,
                is_end=is_end, extracted=extracted,
                meta=dict(n_points=int(n), n_branches=n_branch, root_point=int(root),
                          smooth_radius_mm=graph.get("smooth_sigma_mm"),
                          smooth_positions_mm=graph.get("smooth_positions_mm"),
                          units="mm", frame="mesh frame: origin + ijk * spacing (no direction)"))


def save_point_graph(pg, path):
    """Save as .npz (meta as a JSON string)."""
    np.savez(path, **{k: v for k, v in pg.items() if k != "meta"}, meta=np.array(json.dumps(pg["meta"])))


def load_point_graph(path):
    """Load a saved point graph (.npz) as a dict of numpy arrays plus meta."""
    z = np.load(path)
    pg = {k: z[k] for k in z.files if k != "meta"}
    pg["meta"] = json.loads(str(z["meta"]))
    return pg


# ------------------------------------------------------------------ entry point

def extract_skeleton(mesh, endpoints, source_id=0, zone_pts=None, smooth_mm=1.0, smooth_pos_mm=0.5):
    """Mesh + endpoints -> dense 1D graph (see module docstring)."""
    endpoints = np.asarray(endpoints, float)
    cl = vmtk_centerlines(to_surface(mesh), endpoints, source_id)
    # VMTK can return a degenerate line (source and target only, "Target not reached") without
    # raising; on a broken surface (e.g. thin sheets left by a failed patch) it does
    import pyvista as pv
    poly = pv.wrap(cl)
    lines = [np.asarray(poly.points[np.array(poly.get_cell(i).point_ids)]) for i in range(poly.n_cells)]
    length = [float(np.linalg.norm(np.diff(l, axis=0), axis=1).sum()) for l in lines]
    # a real line is resampled every 0.1 mm; a straight source->target jump is not
    bad = [i for i, (l, L) in enumerate(zip(lines, length)) if len(l) < 2 or L / max(len(l) - 1, 1) > 0.5]
    if bad or not any(L > 1.0 for L in length):
        raise ValueError(f"degenerate VMTK centerline: lines {bad} of {len(lines)} "
                         f"(points {[len(l) for l in lines]})")
    cl = correct_radius(cl, zone_pts)
    graph = smooth_radius(smooth_positions(build_graph(branch_split(cl)), smooth_pos_mm), smooth_mm)
    graph["meta"] = dict(source_id=source_id, endpoints=endpoints, continuity=continuity(graph))
    return graph


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mesh", required=True)
    p.add_argument("--endpoints", required=True, help=".npy: (n, 3) array or AneuSeg endpoints dict")
    p.add_argument("--zone", default=None, help="zone_pts.npy (sac + collar samples) to interpolate radius")
    p.add_argument("--out", required=True)
    p.add_argument("--source_id", type=int, default=0)
    p.add_argument("--smooth_mm", type=float, default=1.0, help="radius smoothing scale (0 = off)")
    p.add_argument("--smooth_pos_mm", type=float, default=0.5, help="position smoothing scale (0 = off)")
    a = p.parse_args()
    ep = np.load(a.endpoints, allow_pickle=True)
    ep = ep.item()["endpoints"] if ep.dtype == object else ep
    zone = np.load(a.zone) if a.zone else None
    graph = extract_skeleton(a.mesh, ep, a.source_id, zone, a.smooth_mm, a.smooth_pos_mm)
    np.save(a.out, graph, allow_pickle=True)
    # the flattened point graph for the models, next to it
    save_point_graph(to_point_graph(graph), str(a.out).replace(".npy", "") + "_points.npz")
    print("\n" + json.dumps(graph["meta"]["continuity"]))  # VMTK leaves its last line open


if __name__ == "__main__":
    main()
