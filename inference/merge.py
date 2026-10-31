"""Merge a generated aneurysm dome into a patch mask, as label 2 attached to the artery (label 1).

Mirrors how the masks were drawn (artery + aneurysm as one structure, then the aneurysm split off),
so the sac touches the artery everywhere at the neck and never overlaps it:

  1. dome patch      faces of the generated complex the sensor calls dome (p > 0.5), small holes
                     filled; its open boundary is the rim
  2. rim check       the share of rim points inside the artery lumen is reported (warning flag
                     rim_inside_lumen); with skip_inside=True such samples are not built
  3. rim projection  every rim point is moved onto the artery wall along the ray from its nearest
                     centerline point (near the site) through it: the first point where the ray
                     leaves the artery. The ray starts inside the lumen and points at the rim, so
                     the projection stays on the rim's side of the centerline, and rims floating
                     off the wall move in, rims inside the lumen move out. Not built if the median
                     move exceeds max_move_frac x the rim radius (the neck would change shape)
  4. dome warp       the rim displacement is spread over the dome, fading to zero at the tip
  5. close inwards   the rim is capped by a cone to its centerline points (inside the lumen), so
                     the solid runs from the dome tip through the wall into the vessel
  6. one structure   F = artery | solid, voxelised at 2x and downsampled by majority, then closed
                     inside a thin band around the rim (no crease or gap at the neck)
  7. split           label 2 = F minus the artery, inside the core
  8. connectivity    every label-2 piece must face-touch the artery; small detached slivers are
                     dropped, a large detached piece is flagged
  9. checks          on the placed dome (closed by a flat cap at its rim, no cone), reported and
                     flagged but still built (the sample is marked rejected, see REJECT):
                       dome_inside_lumen       > max_parent_frac of the dome inside the parent vessel
                       dome_into_other_vessel  > max_other_frac of the dome inside another vessel:
                                               artery voxels owned by centerline points far along
                                               the graph from the site, or in another piece of the mask
                       not_dome_like           sac shape past the limits of inference/sac_shape.py

Flags that stop the build: no_dome, no_rim, rim_not_projectable, rim_move_too_large, empty,
detached_sac. Warnings only: rim_inside_lumen (projected outwards), split_sac.

Coordinates are mm, frame origin + ijk * spacing with i, j, k = x, y, z = array axes 2, 1, 0 of the
(z, y, x) mask (the image model's patch convention).
"""
import numpy as np
from scipy import ndimage as ndi

FACE = ndi.generate_binary_structure(3, 1)
REJECT = {"dome_inside_lumen", "dome_into_other_vessel", "not_dome_like"}       # built, but not usable


# ------------------------------------------------------------------------------ grids

def mm_to_index(p, origin, spacing):
    """mm points (n, 3) -> fractional (z, y, x) array indices."""
    return ((np.asarray(p) - origin) / spacing)[..., ::-1]


def index_to_mm(zyx, origin, spacing):
    return np.asarray(zyx)[..., ::-1] * spacing + origin


def signed_distance(mask, spacing):
    """Signed distance (mm) to the boundary of a (z, y, x) mask: negative inside."""
    sp = np.asarray(spacing)[::-1]
    return ndi.distance_transform_edt(~mask, sampling=sp) - ndi.distance_transform_edt(mask, sampling=sp)


def sample(field, p, origin, spacing):
    """Trilinear value of a (z, y, x) field at mm points."""
    idx = mm_to_index(p, origin, spacing).T
    return ndi.map_coordinates(field, idx, order=1, mode="nearest")


# ------------------------------------------------------------------------------ dome patch

def _components(n, edges):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    A = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    return connected_components(A, directed=False)[1]


def dome_patch(faces, p_dome, thr=0.5):
    """Faces of the dome: those whose vertices are all dome (p > thr), largest connected piece,
    with the small non-dome islands inside it added back (so the patch has one boundary)."""
    F = np.asarray(faces)
    is_d = (p_dome[F] > thr).all(1)
    # face adjacency through shared edges
    e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
    fid = np.tile(np.arange(len(F)), 3)
    order = np.lexsort((e[:, 1], e[:, 0]))
    e, fid = e[order], fid[order]
    same = (e[1:] == e[:-1]).all(1)
    pairs = np.stack([fid[:-1][same], fid[1:][same]], 1)
    def comps(mask):
        pr = pairs[mask[pairs[:, 0]] & mask[pairs[:, 1]]]
        lab = _components(len(F), pr) if len(pr) else np.arange(len(F))
        lab = np.where(mask, lab, -1)
        return lab
    lab = comps(is_d)
    if (lab >= 0).sum() == 0:
        return np.zeros(len(F), bool)
    big = np.bincount(lab[lab >= 0]).argmax()
    dome = lab == big
    # fill holes: non-dome face islands other than the largest non-dome piece
    lab2 = comps(~dome)
    if (lab2 >= 0).any():
        keep_out = np.bincount(lab2[lab2 >= 0]).argmax()
        dome |= (lab2 >= 0) & (lab2 != keep_out)
    return dome


def boundary_loop(faces):
    """Ordered vertex indices of the longest boundary loop of a triangle patch."""
    F = np.asarray(faces)
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    key = np.sort(e, 1)
    uniq, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    be = e[cnt[inv.ravel()] == 1]                          # directed boundary edges
    nxt = {int(a): int(b) for a, b in be}
    loops, seen = [], set()
    for s in nxt:
        if s in seen:
            continue
        loop, v = [], s
        while v not in seen and v in nxt:
            seen.add(v)
            loop.append(v)
            v = nxt[v]
        loops.append(loop)
    return np.array(max(loops, key=len)) if loops else np.zeros(0, int)


# ------------------------------------------------------------------------------ projection and warp

def project_rim(rim, phi, origin, spacing, centerline, step_mm=0.05, reach_mm=8.0):
    """Each rim point onto the artery wall: march from its nearest centerline point c along the ray
    c -> rim point and take the first crossing from inside (phi < 0) to outside. Returns
    (wall points, centerline points, ok mask); ok is False where the ray never leaves the artery or
    never starts inside it."""
    from scipy.spatial import cKDTree
    _, ic = cKDTree(centerline).query(rim)
    c = centerline[ic]
    d = rim - c
    L = np.linalg.norm(d, axis=1, keepdims=True)
    d = d / np.maximum(L, 1e-9)
    t = np.arange(0.0, reach_mm, step_mm)
    pts = c[:, None] + t[None, :, None] * d[:, None]                       # (R, T, 3)
    v = sample(phi, pts.reshape(-1, 3), origin, spacing).reshape(len(rim), len(t))
    wall, ok = np.zeros_like(rim), np.zeros(len(rim), bool)
    for i in range(len(rim)):
        inside = np.flatnonzero(v[i] < 0)
        if len(inside) == 0:
            continue
        out = np.flatnonzero((v[i] >= 0) & (np.arange(len(t)) > inside[0]))
        if len(out) == 0:
            continue
        k = out[0]
        a, b = v[i, k - 1], v[i, k]                                        # interpolate the crossing
        tt = t[k - 1] + (t[k] - t[k - 1]) * (-a) / max(b - a, 1e-9)
        wall[i], ok[i] = c[i] + tt * d[i], True
    return wall, c, ok


def warp_dome(verts, dome_vidx, rim_vidx, delta, normal):
    """Move the dome so its rim vertices move by delta, fading to zero at the dome tip: each dome
    vertex takes the Gaussian-weighted rim displacement nearby, times (1 - height / max height)^2
    (height along the ostium normal, measured from the rim plane)."""
    V = verts.copy()
    R = verts[rim_vidx]
    D = verts[dome_vidx]
    h = (D - R.mean(0)) @ normal
    H = max(float(h.max()), 1e-6)
    fade = np.clip(1 - np.maximum(h, 0) / H, 0, 1) ** 2
    sig = max(np.median(np.linalg.norm(R - np.roll(R, 1, 0), axis=1)) * 4, 0.3)
    w = np.exp(-0.5 * (np.linalg.norm(D[:, None] - R[None], axis=2) / sig) ** 2) + 1e-12
    disp = (w @ delta) / w.sum(1, keepdims=True)
    V[dome_vidx] = D + fade[:, None] * disp
    V[rim_vidx] = R + delta                                                # rim exactly on the wall
    return V


def closed_solid(verts, dome_faces, rim_vidx, apex_pts):
    """The dome patch closed by a cone from the rim to the rim's centerline points: a fan of
    triangles (rim_k, rim_k+1, apex_k) and (rim_k+1, apex_k+1, apex_k)."""
    V = [verts]
    n = len(verts)
    A = np.asarray(apex_pts)
    V.append(A)
    m = len(rim_vidx)
    tri = []
    for k in range(m):
        a, b = rim_vidx[k], rim_vidx[(k + 1) % m]
        ak, bk = n + k, n + (k + 1) % m
        tri += [(a, b, ak), (b, bk, ak)]
    return np.concatenate(V), np.concatenate([dome_faces, np.array(tri)])


def voxelise(verts, faces, shape, origin, spacing, supersample=2, margin_vox=2):
    """Inside of a closed surface on the (z, y, x) grid: points at supersample x the resolution
    inside the surface's bounding box, majority vote per voxel."""
    import pyvista as pv
    out = np.zeros(shape, bool)
    lo = np.floor(mm_to_index(verts.min(0), origin, spacing)[::-1] - margin_vox).astype(int)   # (x, y, z)
    hi = np.ceil(mm_to_index(verts.max(0), origin, spacing)[::-1] + margin_vox).astype(int)
    lo, hi = np.maximum(lo, 0), np.minimum(hi, np.array(shape)[::-1] - 1)
    if (hi < lo).any():
        return out
    s = supersample
    sub = (np.arange(s) + 0.5) / s - 0.5
    axes = [np.add.outer(np.arange(lo[a], hi[a] + 1), sub).ravel() for a in range(3)]
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    ijk = np.stack([X, Y, Z], -1).reshape(-1, 3)
    pts = ijk * np.asarray(spacing) + np.asarray(origin)
    surf = pv.PolyData(verts, np.c_[np.full(len(faces), 3), faces].ravel())
    ins = pv.PolyData(pts).select_enclosed_points(surf, check_surface=False)["SelectedPoints"].astype(bool)
    nx, ny, nz = (hi - lo + 1)
    vote = ins.reshape(nx, s, ny, s, nz, s).mean((1, 3, 5)) >= 0.5           # (x, y, z)
    out[lo[2]:hi[2] + 1, lo[1]:hi[1] + 1, lo[0]:hi[0] + 1] = vote.transpose(2, 1, 0)
    return out


# ------------------------------------------------------------------------------ merge

def vessel_overlap(dome, art, graph, site, origin, spacing, local_mm):
    """Shares of the dome's voxels inside the parent vessel and inside other vessels. Each artery
    voxel the dome overlaps is owned by its nearest centerline point; it is another vessel's when that
    point is far along the graph from the site (more than twice the straight distance and more than
    local_mm), unreachable, or when the voxel is in another connected piece of the artery mask than
    the one nearest the site."""
    from scipy.sparse.csgraph import dijkstra
    from scipy.spatial import cKDTree
    from dataset.ostium_dataset import adjacency
    n = int(dome.sum())
    ov = dome & art
    if n == 0 or not ov.any():
        return 0.0, 0.0
    pos = graph["pos"]
    comp = ndi.label(art, structure=FACE)[0]
    _, near = ndi.distance_transform_edt(~art, return_indices=True)
    s = np.clip(np.rint(mm_to_index(pos[site], origin, spacing)).astype(int), 0, np.array(art.shape) - 1)
    parent = comp[tuple(near[:, s[0], s[1], s[2]])]
    zyx = np.argwhere(ov)
    _, own = cKDTree(pos).query(index_to_mm(zyx, origin, spacing))
    gd = dijkstra(adjacency(pos, graph["edge_index"]), directed=False, indices=site)
    eu = np.linalg.norm(pos[own] - pos[site], axis=1)
    other = (gd[own] > np.maximum(2 * eu, local_mm)) | (comp[tuple(zyx.T)] != parent)
    return float((~other).sum()) / n, float(other.sum()) / n


def merge_sample(seg, origin, spacing, sample_out, graph, core=None, inside_tol_vox=1.0,
                 inside_frac_max=1 / 3, skip_inside=False, max_move_frac=1.0, local_mm=None, band_vox=2,
                 min_piece_frac=0.1, max_parent_frac=0.8, max_other_frac=0.1):
    """Merge one generated aneurysm into seg (z, y, x; 0 background, 1 artery). Returns
    (new_seg or None, report). new_seg is None when the sample is flagged instead of built."""
    origin, spacing = np.asarray(origin, float), np.asarray(spacing, float)
    vox = float(spacing.min())
    art = seg == 1
    phi = signed_distance(art, spacing)
    rep = dict(flags=[])
    V, F, p = sample_out["verts_mm"], sample_out["faces"], sample_out["p_dome"]
    dmask = dome_patch(F, p)
    if not dmask.any():
        rep["flags"].append("no_dome")
        return None, rep
    DF = F[dmask]
    rim = boundary_loop(DF)
    if len(rim) < 6:
        rep["flags"].append("no_rim")
        return None, rep
    # rim inside the lumen -> flag
    depth = sample(phi, V[rim], origin, spacing)
    frac_in = float((depth < -inside_tol_vox * vox).mean())
    rep.update(rim_points=len(rim), rim_inside_frac=round(frac_in, 3), rim_depth_min_mm=round(float(depth.min()), 2))
    if frac_in > inside_frac_max:
        rep["flags"].append("rim_inside_lumen")
        if skip_inside:
            return None, rep
    # centerline near the site
    pos = graph["pos"]
    site = sample_out["site"]
    if local_mm is None:
        local_mm = 3.0 * max(float(np.ptp(V[rim], 0).max()), 2.0)
    near = np.linalg.norm(pos - pos[site], axis=1) < local_mm
    wall, cpts, ok = project_rim(V[rim], phi, origin, spacing, pos[near])
    rep["rim_projected_frac"] = round(float(ok.mean()), 3)
    if ok.mean() < 0.8:
        rep["flags"].append("rim_not_projectable")
        return None, rep
    # rim points whose ray failed keep their position
    wall[~ok] = V[rim][~ok]
    move = np.linalg.norm(wall - V[rim], axis=1)
    rim_radius = max(float(np.ptp(V[rim], 0).max()) / 2, 1e-6)
    rep.update(rim_move_median_mm=round(float(np.median(move)), 2), rim_move_max_mm=round(float(move.max()), 2),
               rim_move_frac=round(float(np.median(move)) / rim_radius, 3))
    if np.median(move) > max_move_frac * rim_radius:
        rep["flags"].append("rim_move_too_large")
        return None, rep
    dome_v = np.unique(DF)
    V2 = warp_dome(V, dome_v, rim, wall - V[rim], sample_out["frame"][2])
    # close inwards, voxelise, one structure
    SV, SF = closed_solid(V2, DF, rim, cpts)
    solid = voxelise(SV, SF, seg.shape, origin, spacing)
    # checks: where the placed dome itself lies (dome closed flat at its rim, no cone)
    dome = voxelise(*closed_solid(V2, DF, rim, np.repeat(V2[rim].mean(0)[None], len(rim), 0)), seg.shape, origin, spacing)
    f_par, f_oth = vessel_overlap(dome, art, graph, site, origin, spacing, local_mm)
    rep.update(dome_in_parent_frac=round(f_par, 3), dome_in_other_vessel_frac=round(f_oth, 3))
    if f_par > max_parent_frac:
        rep["flags"].append("dome_inside_lumen")
    if f_oth > max_other_frac:
        rep["flags"].append("dome_into_other_vessel")
    Fm = art | solid
    rim_idx = np.rint(mm_to_index(wall, origin, spacing)).astype(int)
    band = np.zeros(seg.shape, bool)
    inb = (rim_idx >= 0).all(1) & (rim_idx < np.array(seg.shape)).all(1)
    band[tuple(rim_idx[inb].T)] = True
    band = ndi.binary_dilation(band, iterations=band_vox)
    closed = ndi.binary_closing(Fm, structure=FACE, iterations=1)
    Fm = Fm | (closed & band)
    lab2 = Fm & ~art & (seg == 0)
    if core is not None:
        cz, cy, cx = core
        keep = np.zeros(seg.shape, bool)
        keep[cz[0]:cz[1], cy[0]:cy[1], cx[0]:cx[1]] = True
        rep["voxels_outside_core"] = int((lab2 & ~keep).sum())
        lab2 &= keep
    # connectivity: keep the main sac piece (largest, must face-touch the artery); other pieces are
    # voxel specks in pits of the artery surface (dropped) unless at least min_piece_frac of the
    # main piece (kept, flagged: a split sac)
    pieces, n = ndi.label(lab2, structure=FACE)
    touch = ndi.binary_dilation(art, FACE)
    sizes = np.asarray(ndi.sum(lab2, pieces, range(1, n + 1))) if n else np.zeros(0)
    if n == 0:
        rep["flags"].append("empty")
        return None, rep
    main = int(sizes.argmax()) + 1
    if not (touch & (pieces == main)).any():
        rep["flags"].append("detached_sac")
        return None, rep
    big = [k + 1 for k in range(n) if k + 1 != main and sizes[k] >= min_piece_frac * sizes[main - 1]]
    if big:
        rep["flags"].append("split_sac")
    keep_lab = [main] + [k for k in big if (touch & (pieces == k)).any()]
    dropped = float(lab2.sum()) - float(sum(sizes[k - 1] for k in keep_lab))
    lab2 = np.isin(pieces, keep_lab)
    det_vox = dropped
    holes = ndi.binary_fill_holes(lab2 | art) & ~(lab2 | art)
    holes &= ndi.binary_dilation(lab2, iterations=2) & (seg == 0)         # only next to the new sac
    lab2 |= holes
    new = seg.copy()
    new[lab2] = 2
    neck = lab2 & touch
    crease = ndi.binary_dilation(lab2, FACE) & touch & (seg == 0) & ~lab2
    rep.update(sac_voxels=int(lab2.sum()), sac_volume_mm3=round(float(lab2.sum()) * float(np.prod(spacing)), 1),
               pieces=len(keep_lab), specks_dropped_voxels=int(det_vox), holes_filled=int(holes.sum()),
               neck_share=round(float(neck.sum()) / max(float(lab2.sum()), 1), 3), crease_voxels=int(crease.sum()))
    if lab2.sum() == 0:
        rep["flags"].append("empty")
        return None, rep
    from inference.sac_shape import check
    shape, past = check(lab2, float(spacing.min()))
    rep["shape"] = {k: round(v, 3) for k, v in shape.items()}
    if past:
        rep["flags"].append("not_dome_like")
        rep["shape_past_limit"] = past
    rep["rejected"] = sorted(REJECT & set(rep["flags"]))
    return new, rep
