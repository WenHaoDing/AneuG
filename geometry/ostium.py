"""Ground-truth ostium of a fitted case, in image space (the frame of the case's big meshes).

The GHD fit of every case in runtime_dataset/assembled/<dataset>/<case>/ is placed back in
the original geometry frame by inverting its Stage-1 alignment (alignment_result.npy:
V_img = (V_fit - t) @ R / s; see cfd_data_factory/HANDOFF_stage1_alignment.md). That frame
is the frame of reconstruction4clipping.ply / reconstruction.ply and of the 1D graph
(origin + ijk * spacing, no direction matrix). On 9 ImperialNHS cases the placed fit lies
0.11-0.16 mm (median) from reconstruction4clipping.ply.

The dome is the reviewed per-vertex label used to train the morphology sensor
(runtime_dataset/AneuG_morpho/<case>.npy, `manual_dome`: 523 cases, all reviewed, 23 brushed
by hand). The ostium is the rim of the dome patch: the faces whose three vertices are all
dome, largest patch, holes filled; its boundary walked as one closed loop.

Run in the `new` env (torch, CPU is enough):

    from geometry.ostium import fitted_ostium
    ost = fitted_ostium("0bjVkRg2pM_aneurysm1", "ImperialNHS")

returns dict(
  ring       (K, 3) ostium rim resampled to K points by arc length (image space, mm)
  centroid   (3,)   mean of the ring
  normal     (3,)   unit normal of the best-fit ring plane, pointing from the vessel into the sac
  spans      (2,)   in-plane extents along the ring's two principal axes (mm, larger first)
  area       float  area of the ring (fan around the centroid, mm^2)
  verts, faces      the placed fitted surface (image space) and its faces
  dome       (V,)   bool dome label per vertex
  aneurysm_type, n_rim_loops (dome boundary loops found; the longest is the ostium)
)
"""
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ASSEMBLED = REPO / "runtime_dataset" / "assembled"
MORPHO = REPO / "runtime_dataset" / "AneuG_morpho"
CANONICAL = REPO / "dataset" / "canonical"

_RECON = {}


def _recon(aneurysm_type):
    from models.multi_canonical_ghd_reconstruct import MultiCanonicalGHDReconstruct
    if "mc" not in _RECON:
        _RECON["mc"] = MultiCanonicalGHDReconstruct(str(CANONICAL))
    return _RECON["mc"].get(aneurysm_type)


def placed_fit(case, dataset):
    """Fitted GHD surface in image space: reconstruct, place, denormalize, invert Stage 1."""
    import torch
    d = ASSEMBLED / dataset / case
    t = int(np.load(d / "landmarks.npz", allow_pickle=True)["aneurysm_type"])
    recon = _recon(t)
    G = np.load(d / "ghd_coefficients.npz", allow_pickle=True)
    dev = recon.GHD_eigvec.device
    phi = torch.tensor(np.asarray(G["phi"], np.float32), device=dev).reshape(1, -1)
    mesh = recon.ghd_forward_as_Meshes(phi, denormalize_shape=False)
    V = mesh.verts_packed().detach().cpu().numpy().astype(float)
    F = mesh.faces_packed().detach().cpu().numpy()
    w = np.asarray(G["w_rot"], float)
    th = np.linalg.norm(w)
    if th < 1e-12:
        Rw = np.eye(3)
    else:
        k = w / th
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        Rw = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
    V = (float(np.exp(G["log_scale"])) * (V @ Rw.T) + np.asarray(G["t_vec"], float)) * float(recon.norm_canonical)
    al = np.load(d / "alignment_result.npy", allow_pickle=True).item()
    V = (V - np.asarray(al["t"], float)) @ np.asarray(al["R"], float) / float(al["s"])
    return V, F, t


def clean_dome_faces(F, dome):
    """Dome faces (all three vertices dome), largest connected patch, holes filled.

    Returns a bool per face. Holes = non-dome face islands not connected to the largest
    non-dome region (the vessel)."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    nf = len(F)
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    fid = np.tile(np.arange(nf), 3)
    key = np.sort(e, axis=1)
    order = np.lexsort(key.T[::-1])
    k, f = key[order], fid[order]
    same = (k[1:] == k[:-1]).all(1)
    pairs = np.stack([f[:-1][same], f[1:][same]], 1)        # faces sharing an edge

    def components(mask):
        sel = pairs[mask[pairs[:, 0]] & mask[pairs[:, 1]]]
        A = sp.coo_matrix((np.ones(len(sel)), (sel[:, 0], sel[:, 1])), shape=(nf, nf))
        _, lab = connected_components(A + A.T, directed=False)
        return lab

    fd = dome[F].all(1)
    lab = components(fd)
    big = np.bincount(lab[fd]).argmax()
    fd = fd & (lab == big)
    lab = components(~fd)
    vessel = np.bincount(lab[~fd]).argmax()
    return fd | (~fd & (lab != vessel))


def rim_loops(F, dome):
    """Boundary of the cleaned dome patch, one closed walk per boundary component.

    An Euler circuit follows every boundary edge, so a rim that pinches at a vertex (the
    dome one vertex wide there) still comes out as one loop."""
    import networkx as nx
    fd = F[clean_dome_faces(F, dome)]
    e = np.sort(np.concatenate([fd[:, [0, 1]], fd[:, [1, 2]], fd[:, [2, 0]]]), axis=1)
    u, c = np.unique(e, axis=0, return_counts=True)
    G = nx.Graph()
    G.add_edges_from(u[c == 1].tolist())
    loops = []
    for comp in nx.connected_components(G):
        sub = G.subgraph(comp)
        if not nx.is_eulerian(sub):
            continue
        loops.append(np.array([a for a, _ in nx.eulerian_circuit(sub)]))
    return loops


def resample_loop(P, K):
    Q = np.vstack([P, P[:1]])
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))]
    t = np.linspace(0, s[-1], K, endpoint=False)
    return np.stack([np.interp(t, s, Q[:, i]) for i in range(3)], 1)


def fitted_ostium(case, dataset="ImperialNHS", K=32):
    V, F, t = placed_fit(case, dataset)
    dome = np.asarray(np.load(MORPHO / f"{case}.npy", allow_pickle=True).item()["manual_dome"], bool)
    loops = rim_loops(F, dome)
    if not loops:
        raise ValueError(f"{case}: dome patch has no boundary loop")
    P = [V[l] for l in loops]
    perim = [np.linalg.norm(np.diff(np.vstack([p, p[:1]]), axis=0), axis=1).sum() for p in P]
    ring = resample_loop(P[int(np.argmax(perim))], K)
    c = ring.mean(0)
    _, sv, vt = np.linalg.svd(ring - c, full_matrices=False)
    n = vt[2]
    if (V[dome].mean(0) - c) @ n < 0:          # point into the sac
        n = -n
    proj = (ring - c) @ vt[:2].T
    spans = np.ptp(proj, axis=0)
    area = 0.5 * np.abs(np.cross(ring - c, np.roll(ring, -1, 0) - c) @ n).sum()
    return dict(ring=ring, centroid=c, normal=n, spans=np.sort(spans)[::-1], area=float(area),
                verts=V, faces=F, dome=dome, aneurysm_type=t, n_rim_loops=len(loops))


# =========================================================================================
# Canonical Fourier code of an ostium, anchored to the 1D graph (design: TASK §5.2)
#
#   C        ring centre (mean of the arc-length-uniform ring = Fourier DC term)
#   P        graph point nearest to C (index into the point graph)
#   n        (C - P) / |C - P|                     axis 1, from the vessel towards the ostium
#   e1       graph tangent at P, minus its n part  axis 2; sign = away from the graph root,
#            times `sign` (+1 / -1: random flip at training time, given to the model)
#   e2       n x e1                                axis 3
#   s        scale: median graph radius within `scale_mm` of P
#   ring     local coords X = (ring - C) [e1 e2 n] / s, counter-clockwise about n,
#            starting where the ring crosses the +e1 half-plane, resampled to M points
#            uniformly in arc length
#   coeffs   rfft(X) harmonics 1..H  -> (H, 3) complex = 6H real numbers (truncation = the
#            smoothing of the target)
# =========================================================================================

def rim_points(verts, faces, dome):
    """The raw ostium rim (full resolution): longest boundary loop of the cleaned dome patch."""
    loops = rim_loops(faces, dome)
    P = [verts[l] for l in loops]
    perim = [np.linalg.norm(np.diff(np.vstack([p, p[:1]]), axis=0), axis=1).sum() for p in P]
    return P[int(np.argmax(perim))]


def _arc_resample(P, M):
    return resample_loop(P, M)


def graph_frame_axes(pg, p, sign=1, hops=3):
    """Unit tangent of the point graph at point p, oriented away from the graph root, times sign.

    Central difference over `hops` points back (towards the root) and forward (only when the
    path continues uniquely; at a junction or a vessel end the backward direction alone)."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import breadth_first_order
    pos = np.asarray(pg["pos"], float)
    ei = np.asarray(pg["edge_index"])
    n = len(pos)
    A = csr_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n))
    _, pred = breadth_first_order(A, int(pg["meta"]["root_point"]), directed=False)
    children = [[] for _ in range(n)]
    for j in range(n):
        if pred[j] >= 0:
            children[pred[j]].append(j)
    back = p
    for _ in range(hops):
        back = pred[back] if pred[back] >= 0 else back
    fwd = p
    for _ in range(hops):
        if len(children[fwd]) != 1:
            break
        fwd = children[fwd][0]
    if fwd == p and back == p:
        raise ValueError("isolated graph point")
    t = pos[fwd] - pos[back]
    return sign * t / np.linalg.norm(t)


def graph_tangents(pg, sign=1, hops=3):
    """graph_frame_axes for every point at once: (N, 3) unit tangents, oriented away from the root,
    times sign."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import breadth_first_order
    pos = np.asarray(pg["pos"], float)
    ei = np.asarray(pg["edge_index"])
    n = len(pos)
    A = csr_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n))
    _, pred = breadth_first_order(A, int(pg["meta"]["root_point"]), directed=False)
    idx = np.arange(n)
    up = np.where(pred >= 0, pred, idx)
    n_child = np.bincount(pred[pred >= 0], minlength=n)
    only_child = idx.copy()
    has = pred >= 0
    only_child[pred[has]] = idx[has]                    # valid where n_child == 1
    step = np.where(n_child == 1, only_child, idx)
    back, fwd = idx, idx
    for _ in range(hops):
        back, fwd = up[back], step[fwd]
    t = pos[fwd] - pos[back]
    return sign * t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-12)


def local_scale(pg, p, scale_mm=3.0):
    pos = np.asarray(pg["pos"], float)
    near = np.linalg.norm(pos - pos[p], axis=1) < scale_mm
    return float(np.median(np.asarray(pg["radius"])[near]))


def local_scales(pg, scale_mm=3.0):
    """local_scale for every point: (N,)."""
    from scipy.spatial import cKDTree
    pos = np.asarray(pg["pos"], float)
    r = np.asarray(pg["radius"], float)
    nb = cKDTree(pos).query_ball_point(pos, scale_mm - 1e-9)
    return np.array([np.median(r[j]) for j in nb])


def _frame(pg, p, C, sign):
    n = C - np.asarray(pg["pos"][p], float)
    n = n / np.linalg.norm(n)
    t = graph_frame_axes(pg, p, sign)
    e1 = t - (t @ n) * n
    e1 /= np.linalg.norm(e1)
    return np.stack([e1, np.cross(n, e1), n])          # rows: e1, e2, n


def _canonical_local(X, M):
    """Counter-clockwise about +z, start at the +x crossing, uniform arc length (M points)."""
    area = 0.5 * np.sum(X[:, 0] * np.roll(X[:, 1], -1) - np.roll(X[:, 0], -1) * X[:, 1])
    if area < 0:
        X = X[::-1]
    th = np.arctan2(X[:, 1], X[:, 0])
    nxt = np.roll(np.arange(len(X)), -1)
    # segment that crosses theta = 0 going counter-clockwise (y from - to +, on the +x side)
    cross = np.flatnonzero((X[:, 1] < 0) & (X[nxt, 1] >= 0) & ((X[:, 0] + X[nxt, 0]) > 0))
    if len(cross) == 0:                                   # ring does not surround C: nearest angle
        i0 = int(np.argmin(np.abs(th)))
        start, X = X[i0], np.roll(X, -i0, axis=0)
    else:
        i = cross[np.argmax((X[cross, 0] + X[nxt[cross], 0]))]
        a = -X[i, 1] / (X[nxt[i], 1] - X[i, 1])
        start = X[i] + a * (X[nxt[i]] - X[i])
        X = np.vstack([start[None], np.roll(X, -(i + 1), axis=0)])
    return _arc_resample(X, M)


def encode_ostium(rim, pg, H=5, M=128, sign=1, scale_mm=3.0, site=None):
    """Raw rim (ordered loop, image space) + point graph -> canonical Fourier code.

    site: graph point to anchor at (default: the point nearest the ring centre). Pass it to
    encode relative to a jittered site (teacher forcing with a slightly-off P)."""
    rim = np.asarray(rim, float)
    dense = _arc_resample(rim, M)
    C = dense.mean(0)
    p = int(np.argmin(np.linalg.norm(np.asarray(pg["pos"], float) - C, axis=1))) if site is None else int(site)
    s = local_scale(pg, p, scale_mm)
    Rf = _frame(pg, p, C, sign)
    X = _canonical_local((rim - C) @ Rf.T / s, M)
    F = np.fft.rfft(X, axis=0) / M
    coeffs = F[1:H + 1]                                   # (H, 3) complex; DC ~ 0 by construction
    return dict(p=p, offset=(C - np.asarray(pg["pos"][p], float)) / s, scale=s, sign=int(sign),
                frame=Rf, coeffs=coeffs, H=H, M=M)


def decode_ostium(code, pg, M=None, coeffs=None, drop_tangential=False):
    """Canonical code (+ the point graph for P and the frame) -> ring (M, 3) in image space.

    drop_tangential: remove the offset's component along the graph tangent first (a generated
    offset may tilt along the vessel; in the data it is perpendicular to within 1-2 deg)."""
    M = M or code["M"]
    c = code["coeffs"] if coeffs is None else coeffs
    F = np.zeros((M // 2 + 1, 3), complex)
    F[1:len(c) + 1] = c
    X = np.fft.irfft(F * M, n=M, axis=0)
    p, s = code["p"], code["scale"]
    off = np.asarray(code["offset"], float)
    if drop_tangential:
        t = graph_frame_axes(pg, p, 1)
        off = off - (off @ t) * t
    C = np.asarray(pg["pos"][p], float) + off * s
    Rf = _frame(pg, p, C, code["sign"])
    return C + s * X @ Rf


def code_to_vector(code):
    """33 numbers: offset (3) then a_1..a_H, b_1..b_H interleaved per harmonic as
    [a_k (3), b_k (3)] -- the model's target y."""
    c = code["coeffs"]
    a, b = 2 * c.real, -2 * c.imag
    return np.concatenate([code["offset"], np.concatenate([a, b], axis=1).ravel()])


def vector_to_coeffs(y, H=5):
    """Inverse of code_to_vector's Fourier part: (33,) -> offset (3), complex coeffs (H, 3)."""
    y = np.asarray(y, float)
    ab = y[3:3 + 6 * H].reshape(H, 6)
    return y[:3], (ab[:, :3] - 1j * ab[:, 3:]) / 2


def coeffs_to_vector(coeffs):
    """(H, 3) complex -> (6H,) real: [Re x, Re y, Re z, Im x, Im y, Im z] per harmonic."""
    return np.concatenate([coeffs.real, coeffs.imag], axis=1).ravel()


def flip_code(code, pg, M=None):
    """The same ring expressed with the opposite tangent sign (training-time augmentation):
    decode densely, re-encode with -sign. Exact up to the H-band truncation."""
    ring = decode_ostium(code, pg, M=4 * code["M"])
    return encode_ostium(ring, pg, H=code["H"], M=code["M"], sign=-code["sign"])


# =========================================================================================
# Endpoints inside the dome
# =========================================================================================

def closed_dome(verts, faces, dome):
    """The dome patch (cleaned, as for the ostium) closed by a fan cap over its rim."""
    import trimesh
    fd = faces[clean_dome_faces(faces, dome)]
    loops = rim_loops(faces, dome)
    V = np.asarray(verts, float)
    # each fan triangle runs its rim edge against the patch's direction for that edge (the loop
    # walk's own direction is arbitrary), so the closed mesh is consistently oriented
    patch_dir = set(map(tuple, np.concatenate([fd[:, [0, 1]], fd[:, [1, 2]], fd[:, [2, 0]]]).tolist()))
    caps, extra = [], []
    for l in loops:
        c_id = len(V) + len(extra)
        extra.append(V[l].mean(0))
        a, b = np.asarray(l), np.roll(l, -1)
        same = np.array([(int(i), int(j)) in patch_dir for i, j in zip(a, b)])
        caps.append(np.stack([np.where(same, b, a), np.where(same, a, b), np.full(len(l), c_id)], 1))
    V2 = np.vstack([V, np.asarray(extra)]) if extra else V
    F2 = np.vstack([fd] + caps) if caps else fd
    m = trimesh.Trimesh(V2, F2, process=False)
    m.remove_unreferenced_vertices()
    return m


def points_in_dome(points, verts, faces, dome, near_mm=0.5):
    """Bool per point: inside the closed dome (winding number >= 0.5) or within near_mm of
    the dome surface. Returns (mask, winding numbers, distance to the dome patch)."""
    import igl
    import trimesh
    m = closed_dome(verts, faces, dome)
    P = np.ascontiguousarray(np.atleast_2d(points), dtype=np.float64)
    w = np.abs(igl.winding_number(np.ascontiguousarray(m.vertices, dtype=np.float64),
                                  np.ascontiguousarray(m.faces, dtype=np.int64), P))
    patch = trimesh.Trimesh(np.asarray(verts, float), faces[clean_dome_faces(faces, dome)], process=False)
    _, d, _ = trimesh.proximity.closest_point(patch, P)
    return (w >= 0.5) | (d < near_mm), w, d
