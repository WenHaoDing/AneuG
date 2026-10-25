"""Remove the aneurysm sac from an ImperialNHS vessel mesh and patch the vessel under it.

Used by the skeleton pipeline (vessel_edit/skeleton_pipeline.py) so VMTK can extract a
centerline that is not pulled into the sac. Works on local copies of a case folder, never on
the geometry dataset itself.

Data facts this relies on (checked against AneuSeg/IAgents/tools/reconstruction.py):
  - reconstruction4clipping.ply / reconstruction.ply are marching cubes (level 0.7) of the
    region-grown mask (sac + arteries grown from the sac), then 25 Laplacian iterations.
  - Their frame is  origin + ijk * spacing  (sitk origin/spacing, NO direction matrix), so the
    label.nrrd voxels map onto the mesh with that same formula, not with 'space directions'.
  - label.nrrd: 1 = arteries, 2 = this case's sac.

Pipeline (one function per step, so variants can be swapped):
  1. locate_sac        per-vertex label = label of the nearest labelled voxel
  2. cut_sac           remove sac faces, keep the largest island
  3. trim_collar       remove a geodesic collar around the hole (neck bulge, rim defects)
  4. heal              centerline-guided reconstruction (see heal_implicit)
"""
import json
import os
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- loading

def load_case(case_dir, mesh_name="reconstruction4clipping.ply", label=False, mesh=None):
    """A geometry case: big mesh + endpoints (+ label.nrrd only when label=True).

    The sac now comes from the GHD fit (locate_sac_from_fit), so label.nrrd is optional;
    only the old label-2 path (locate_sac) needs it. `mesh` overrides the loaded mesh (e.g.
    a decimated copy)."""
    import trimesh
    d = Path(case_dir)
    case = d.name
    if not (d / mesh_name).exists():
        mesh_name = "reconstruction.ply"
    if mesh is None:
        mesh = trimesh.load(d / mesh_name, process=True)
    ep = np.load(d / "endpoints_manual.npy", allow_pickle=True).item()
    c = dict(case=case, mesh=mesh, mesh_name=mesh_name, label=None, origin=None, spacing=None,
             endpoints=np.asarray(ep["endpoints"], float),
             aneurysm_type=int(ep.get("aneurysm_type", -1)),
             aneurysm_centroid=np.asarray(ep["aneurysm_centroid"], float))
    if label:
        import nrrd
        lab, hdr = nrrd.read(str(d / "label.nrrd"))
        M = np.asarray(hdr["space directions"], float)
        c.update(label=lab, origin=np.asarray(hdr["space origin"], float),
                 spacing=np.linalg.norm(M, axis=1))
    return c


def voxels_to_mesh_frame(ijk, c):
    return c["origin"] + np.asarray(ijk, float) * c["spacing"]


def labelled_points(c, values=(1, 2), pad_mm=8.0):
    """Voxel centres (mesh frame) and labels near the mesh bounding box."""
    lo, hi = c["mesh"].bounds
    lo_ijk = np.floor((lo - pad_mm - c["origin"]) / c["spacing"]).astype(int).clip(0)
    hi_ijk = np.ceil((hi + pad_mm - c["origin"]) / c["spacing"]).astype(int) + 1
    sub = c["label"][lo_ijk[0]:hi_ijk[0], lo_ijk[1]:hi_ijk[1], lo_ijk[2]:hi_ijk[2]]
    ijk = np.argwhere(np.isin(sub, values))
    return voxels_to_mesh_frame(ijk + lo_ijk, c), sub[tuple(ijk.T)]


# ---------------------------------------------------------------- step 1-3

def locate_sac(c):
    """Per-vertex label from the nearest labelled voxel, then majority-vote cleanup on faces."""
    from scipy.spatial import cKDTree
    pts, labs = labelled_points(c)
    _, idx = cKDTree(pts).query(c["mesh"].vertices)
    v_lab = labs[idx]
    f_sac = (v_lab[c["mesh"].faces] == 2).sum(1) >= 2
    f_sac = clean_face_labels(c["mesh"], f_sac)
    return v_lab, f_sac, pts[labs == 2]


def locate_sac_from_fit(c, fit_verts, fit_faces, fit_dome_faces, max_mm=0.75):
    """Sac faces of the big mesh from the placed GHD fit (the generator's definition of the sac).

    A big-mesh vertex is sac when it lies on the placed fit (within max_mm; the fit sits
    0.11-0.16 mm median, <= 0.27 mm p90 from the big mesh) and its closest point there is on a
    dome face (fit_dome_faces: bool per fit face, the cleaned dome patch that also defines the
    ostium). Without the distance limit, vertices of other vessels whose closest fit point
    happens to be on the dome are labelled sac. Then the same cleanup as locate_sac."""
    import trimesh
    fit = trimesh.Trimesh(fit_verts, fit_faces, process=False)
    _, dist, tri = trimesh.proximity.closest_point(fit, c["mesh"].vertices)
    v_sac = np.asarray(fit_dome_faces)[tri] & (dist < max_mm)
    f_sac = v_sac[c["mesh"].faces].sum(1) >= 2
    return clean_face_labels(c["mesh"], f_sac)


def clean_face_labels(mesh, is_sac, passes=3):
    """Majority vote over face neighbours, then keep the largest sac component."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    adj = mesh.face_adjacency
    n = len(mesh.faces)
    A = sp.coo_matrix((np.ones(2 * len(adj)), (np.r_[adj[:, 0], adj[:, 1]],
                                               np.r_[adj[:, 1], adj[:, 0]])), shape=(n, n)).tocsr()
    lab = is_sac.copy()
    for _ in range(passes):
        votes = A @ lab.astype(float)
        deg = np.asarray(A.sum(1)).ravel()
        lab = np.where(votes > deg / 2, True, np.where(votes < deg / 2, False, lab))
    S = A[lab][:, lab]
    ncomp, comp = connected_components(S, directed=False)
    if ncomp > 1:
        keep = np.argmax(np.bincount(comp))
        ids = np.flatnonzero(lab)
        lab = np.zeros(n, bool)
        lab[ids[comp == keep]] = True
    return lab


def vertex_graph(mesh):
    import scipy.sparse as sp
    e = mesh.edges_unique
    w = mesh.edges_unique_length
    n = len(mesh.vertices)
    return sp.coo_matrix((np.r_[w, w], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
                         shape=(n, n)).tocsr()


def geodesic_from(mesh, seed_vertices, limit=None):
    from scipy.sparse.csgraph import dijkstra
    G = vertex_graph(mesh)
    d = dijkstra(G, directed=False, indices=np.unique(seed_vertices), min_only=True,
                 limit=np.inf if limit is None else limit)
    return d


def submesh(mesh, face_mask, largest=True):
    import trimesh
    m = trimesh.Trimesh(mesh.vertices, mesh.faces[face_mask], process=False)
    m.remove_unreferenced_vertices()
    if largest:
        parts = m.split(only_watertight=False)
        if len(parts) > 1:
            m = max(parts, key=lambda p: len(p.faces))
    return m


def removal_zone(c, f_sac, collar_mm):
    """Faces removed = sac faces + every face within collar_mm (geodesic) of them."""
    mesh = c["mesh"]
    sac_v = np.unique(mesh.faces[f_sac])
    d = geodesic_from(mesh, sac_v, limit=collar_mm + 1.0)
    f_rm = f_sac | (d[mesh.faces].min(1) <= collar_mm)
    return f_rm, d


def boundary_loops(mesh):
    import networkx as nx
    e = mesh.edges_sorted
    u, inv, cnt = np.unique(e, axis=0, return_inverse=True, return_counts=True)
    b = u[cnt == 1]
    G = nx.Graph()
    G.add_edges_from(b.tolist())
    return [np.array(list(cc)) for cc in nx.connected_components(G)]


# ---------------------------------------------------------------- rendering

def to_pv(m):
    import pyvista as pv
    return pv.PolyData(np.asarray(m.vertices),
                       np.hstack([np.full((len(m.faces), 1), 3), m.faces]).ravel())


def view_dirs(c, center):
    """Three camera directions: from outside the sac, and two side views."""
    outward = center - c["mesh"].vertices.mean(0)
    outward /= np.linalg.norm(outward) + 1e-9
    a = np.cross(outward, [0, 0, 1.0])
    if np.linalg.norm(a) < 1e-3:
        a = np.cross(outward, [0, 1.0, 0])
    a /= np.linalg.norm(a)
    return [outward, a, -a]


def render_panels(path, panels, center, dirs, dist=22.0, title=None):
    """panels: list of (label, [(mesh_or_points, kwargs), ...]); one row per view direction."""
    import pyvista as pv
    if not os.environ.get("DISPLAY"):
        pv.start_xvfb()
    pl = pv.Plotter(off_screen=True, shape=(len(dirs), len(panels)),
                    window_size=(520 * len(panels), 480 * len(dirs)), border=False)
    for r, dvec in enumerate(dirs):
        for k, (label, items) in enumerate(panels):
            pl.subplot(r, k)
            for obj, kw in items:
                pl.add_mesh(obj, **kw)
            pl.camera.focal_point = center
            pl.camera.position = center + dist * dvec
            up = np.array([0, 0, 1.0]) if abs(dvec[2]) < 0.9 else np.array([0, 1.0, 0])
            pl.camera.up = up
            if r == 0:
                pl.add_text(label, font_size=10)
    if title:
        pl.subplot(0, 0)
        pl.add_text("\n\n" + title, font_size=8)
    pl.screenshot(str(path))
    pl.close()


# ---------------------------------------------------------------- step 4: implicit heal
#
# healed = blend( original ,  (original minus zone)  U  bridged tube )
#   zone  : the removed faces (sac + collar) and the sac voxels
#   tube  : along each centerline path, the cross-sections whose rays touch the zone are
#           replaced by a Hermite bridge between two healthy anchors; the lumen profile R(theta)
#           is measured at both anchors by ray casting and Hermite-interpolated in arc length
#           (value and slope match at the anchors, so the tube joins the real vessel C1).
#   blend : w = smoothstep(distance to zone / blend_mm); w = 1 keeps the original exactly.
# One marching cubes on the blended field gives a single surface: no seam to stitch.

def signed_distance(points, mesh, chunk=2_000_000, return_face=False):
    """Negative inside. Fast winding number for the sign, so small openings are tolerated."""
    import igl
    V = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
    F = np.ascontiguousarray(mesh.faces, dtype=np.int64)
    out = np.empty(len(points))
    face = np.empty(len(points), dtype=np.int64)
    for s in range(0, len(points), chunk):
        r = igl.signed_distance(
            np.ascontiguousarray(points[s:s + chunk], dtype=np.float64), V, F,
            sign_type=igl.SignedDistanceType.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER)
        out[s:s + chunk], face[s:s + chunk] = r[0], r[1]
    return (out, face) if return_face else out


class Zone:
    """Distance to the removed region, continuous everywhere.

    Outside the vessel: distance to the removed faces. Inside: max(0, d_removed - d_kept), i.e.
    zero wherever the removed wall is at least as close as the kept wall (sac interior, lumen
    under the neck), growing smoothly towards the kept wall."""

    def __init__(self, mesh, f_rm, sample_mm=0.08):
        from scipy.spatial import cKDTree
        self.surface_pts = self._samples(mesh, f_rm, sample_mm)
        self.tree = cKDTree(self.surface_pts)
        self.keep_tree = cKDTree(self._samples(mesh, ~f_rm, sample_mm))
        self.mesh = mesh
        self.f_rm = f_rm

    @staticmethod
    def _samples(mesh, fmask, sample_mm):
        import trimesh
        sub = trimesh.Trimesh(mesh.vertices, mesh.faces[fmask], process=False)
        sub.remove_unreferenced_vertices()
        n = max(2000, int(sub.area / sample_mm ** 2))
        pts, _ = trimesh.sample.sample_surface_even(sub, n)
        return np.vstack([pts, sub.vertices])

    def distance(self, x, upper=np.inf, sd=None):
        """sd: optional (signed distance, closest face) of x to the original mesh."""
        reach = 4.0                      # beyond this the kept wall is surely closer
        d, _ = self.tree.query(x, distance_upper_bound=upper + reach)
        f0 = signed_distance(x, self.mesh) if sd is None else sd[0]
        inside = f0 < 0
        dk, _ = self.keep_tree.query(x[inside], distance_upper_bound=reach)
        d[inside] = np.maximum(0.0, d[inside] - np.minimum(dk, reach))
        return np.minimum(d, upper)


def load_paths(npz):
    z = np.load(npz)
    return [(z[f"path{i}_pts"], z[f"path{i}_r"]) for i in range(int(z["n_paths"]))
            if len(z[f"path{i}_pts"]) > 10]


def tangents(pts):
    t = np.gradient(pts, axis=0)
    return t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-12)


def perp(t):
    a = np.array([1.0, 0, 0]) if abs(t[0]) < 0.9 else np.array([0, 1.0, 0])
    u = a - (a @ t) * t
    return u / np.linalg.norm(u)


def transport(ts, u0):
    us = np.empty_like(ts)
    u = u0 - (u0 @ ts[0]) * ts[0]
    for i, t in enumerate(ts):
        u = u - (u @ t) * t
        u /= np.linalg.norm(u)
        us[i] = u
    return us, np.cross(ts, us)


def ray_profiles(f, C, T, U, rmax, K=48, step=0.025):
    """Distance from each centre C[i] to the wall along K in-plane rays (NaN if no exit).

    Returns R (n, K) and the ray sample points (n, K, S, 3) for zone tests."""
    V = np.cross(T, U)
    th = np.linspace(0, 2 * np.pi, K, endpoint=False)
    dirs = np.cos(th)[None, :, None] * U[:, None, :] + np.sin(th)[None, :, None] * V[:, None, :]
    S = int(np.ceil(rmax / step)) + 1
    radii = np.arange(S) * step
    P = C[:, None, None, :] + dirs[:, :, None, :] * radii[None, None, :, None]
    val = f(P.reshape(-1, 3)).reshape(len(C), K, S)
    out = val > 0
    first = np.where(out.any(-1), out.argmax(-1), -1)
    R = np.full((len(C), K), np.nan)
    ok = first > 0
    i0 = np.clip(first - 1, 0, S - 1)
    v0 = np.take_along_axis(val, i0[..., None], -1)[..., 0]
    v1 = np.take_along_axis(val, np.clip(first, 0, S - 1)[..., None], -1)[..., 0]
    R[ok] = radii[i0[ok]] + step * (-v0[ok]) / (v1[ok] - v0[ok] + 1e-12)
    return R, P, first


def clean_profile(R):
    """Replace missing / escaping rays (into another branch) by the circular median."""
    R = R.copy()
    med = np.nanmedian(R)
    bad = ~np.isfinite(R) | (R > 1.8 * med)
    R[bad] = np.nan
    if np.isnan(R).all():
        return np.full_like(R, med)
    idx = np.arange(len(R))
    good = ~np.isnan(R)
    R[~good] = np.interp(idx[~good], idx[good], R[good], period=len(R))
    return R


def lowpass_profile(R, modes=8):
    F = np.fft.rfft(R)
    F[modes + 1:] = 0
    return np.fft.irfft(F, n=len(R))


def hermite(p0, p1, m0, m1, t):
    t = t[:, None] if np.ndim(p0) else t
    h00 = 2 * t ** 3 - 3 * t ** 2 + 1
    h10 = t ** 3 - 2 * t ** 2 + t
    h01 = -2 * t ** 3 + 3 * t ** 2
    h11 = t ** 3 - t ** 2
    return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1


def bridge_path(pts, r, f, zone, contam_tol=0.35, ext_factor=0.75, K=48, log=None):
    """Find the contaminated stretch of one centerline path and bridge it.

    Returns a list of bridges (dicts) or [] if the path never touches the zone."""
    from scipy.ndimage import label as cc_label
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    T = tangents(pts)
    near = zone.distance(pts, upper=8.0) < 3.0 * r + 1.0
    if not near.any():
        return []
    ids = np.flatnonzero(near)
    U = np.array([perp(t) for t in T[ids]])
    rmax = float(2.5 * r[ids].max() + 0.5)
    R, P, first = ray_profiles(f, pts[ids], T[ids], U, rmax, K=K)
    # rays up to the wall (or the whole ray when it never exits) must not touch the zone
    S = P.shape[2]
    upto = np.where(first < 0, S - 1, first)
    mask = np.arange(S)[None, None, :] <= upto[..., None]
    dz = zone.distance(P.reshape(-1, 3), upper=contam_tol + 0.01).reshape(P.shape[:3])
    hit = ((dz < contam_tol) & mask).any(axis=(1, 2))
    contam = np.zeros(len(pts), bool)
    contam[ids[hit]] = True
    contam |= zone.distance(pts, upper=1.0) <= 0.0          # centre inside the sac
    # every zone point within reach of this path must fall inside a bridge: mark the centerline
    # points they project onto (nearest point, within 2.5 radii of the wall)
    from scipy.spatial import cKDTree
    dzp, j = cKDTree(pts).query(zone.surface_pts)
    own = dzp < 2.5 * r[j] + 0.5
    contam[j[own]] = True
    # fill gaps between marked points shorter than one radius, so one sac gives one bridge
    idx = np.flatnonzero(contam)
    for a_, b_ in zip(idx[:-1], idx[1:]):
        if b_ - a_ > 1 and s[b_] - s[a_] < np.median(r[a_:b_ + 1]):
            contam[a_:b_] = True
    lab, n = cc_label(contam)
    bridges = []
    for k in range(1, n + 1):
        seg = np.flatnonzero(lab == k)
        r_loc = np.median(r[max(seg[0] - 20, 0):seg[-1] + 20])
        ext = max(0.5, ext_factor * r_loc)
        ia = np.searchsorted(s, s[seg[0]] - ext) - 1
        ib = np.searchsorted(s, s[seg[-1]] + ext) + 1
        if ia < 2 or ib > len(pts) - 3:
            if log is not None:
                log.append(f"  stretch {s[seg[0]]:.1f}-{s[seg[-1]]:.1f} mm reaches a path end; not bridged")
            continue
        bridges.append(make_bridge(pts, s, T, ia, ib, f, K))
    return bridges


def make_bridge(pts, s, T, ia, ib, f, K, ds=0.02, dslope=0.4, modes=8):
    pa, pb, ta, tb = pts[ia], pts[ib], T[ia], T[ib]
    L = np.linalg.norm(pb - pa)
    lam = np.linspace(0, 1, max(int(L / ds), 20))
    C = hermite(pa, pb, ta * L, tb * L, lam)
    Tb = tangents(C)
    Ub, Vb = transport(Tb, perp(ta))
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]
    # profiles at the anchors and one step further out (for the slope), in the bridge frame
    # points ~dslope beyond each anchor for the radius slope, kept on the line (an anchor can sit
    # near a line end: UPF_P0103.00_ID1_cut1 indexed one past the end)
    ja = int(np.clip(np.searchsorted(s, s[ia] - dslope), 0, ia))
    jb = int(np.clip(np.searchsorted(s, s[ib] + dslope), ib, len(pts) - 1))
    Cs = np.stack([pa, pts[ja], pb, pts[jb]])
    Ts = np.stack([Tb[0], T[ja], Tb[-1], T[jb]])
    Us = np.stack([Ub[0], Ub[0], Ub[-1], Ub[-1]])
    Us = Us - (Us * Ts).sum(1, keepdims=True) * Ts
    Us /= np.linalg.norm(Us, axis=1, keepdims=True)
    rmax = 4.0 * 2.5
    R, _, _ = ray_profiles(f, Cs, Ts, Us, rmax, K=K)
    R = np.stack([lowpass_profile(clean_profile(x), modes) for x in R])
    dRa = (R[0] - R[1]) / max(s[ia] - s[ja], 1e-6)
    dRb = (R[3] - R[2]) / max(s[jb] - s[ib], 1e-6)
    return dict(curve=C, T=Tb, U=Ub, V=Vb, lam=lam, arc=arc, Ra=R[0], Rb=R[2],
                dRa=dRa, dRb=dRb, ia=ia, ib=ib, anchors=(pa, pb), length=arc[-1])


def bridge_field(br, X, mode="profile", cover=2.5, return_weight=False, ramp=0.5):
    """Tube field of one bridge at points X.

    NaN where X is beyond the bridge's end planes or farther than cover * max radius."""
    from scipy.spatial import cKDTree
    _, j = cKDTree(br["curve"]).query(X)
    c, t, u, v = br["curve"][j], br["T"][j], br["U"][j], br["V"][j]
    d = X - c
    along = (d * t).sum(1)
    beyond = ((j == 0) & (along < -1e-3)) | ((j == len(br["curve"]) - 1) & (along > 1e-3))
    dp = d - along[:, None] * t
    rho = np.linalg.norm(dp, axis=1)
    th = np.mod(np.arctan2((dp * v).sum(1), (dp * u).sum(1)), 2 * np.pi)
    K = len(br["Ra"])
    Ls = br["length"]
    Ra, Rb, dRa, dRb = br["Ra"], br["Rb"], br["dRa"], br["dRb"]
    if mode == "ellipse":                       # keep the offset and the ellipticity only
        Ra, Rb, dRa, dRb = (lowpass_profile(x, 2) for x in (Ra, Rb, dRa, dRb))
    Rk = hermite(Ra[None], Rb[None], (dRa * Ls)[None], (dRb * Ls)[None], br["lam"][j])
    if mode == "circle":
        Rx = Rk.mean(1)
    else:
        x = th / (2 * np.pi) * K
        k0 = np.floor(x).astype(int) % K
        a = x - np.floor(x)
        Rx = (1 - a) * Rk[np.arange(len(X)), k0] + a * Rk[np.arange(len(X)), (k0 + 1) % K]
    out = rho - Rx
    Rmax = max(br["Ra"].max(), br["Rb"].max())
    out[beyond | (rho > cover * Rmax)] = np.nan
    if not return_weight:
        return out
    # soft coverage: ramps to 1 within `ramp` mm inside either end plane and over the outer
    # fifth of the radial reach, so switching between tube and original leaves no step
    s_pos = br["arc"][j] + np.where((j == 0) | (j == len(br["curve"]) - 1), along, 0.0)
    inside = np.minimum(s_pos, br["length"] - s_pos)
    wgt = smoothstep(inside / ramp) * smoothstep((cover * Rmax - rho) / (0.2 * cover * Rmax))
    wgt[np.isnan(out)] = 0.0
    return out, wgt


def smooth_min(a, b, k):
    with np.errstate(invalid="ignore"):
        h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
        return b * (1 - h) + a * h - k * h * (1 - h)


def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


MAX_GRID_POINTS = 40e6        # ~10 GB peak; a 42 x 48 x 74 mm AneuX mesh at 0.1 mm needed 164 M


def grid_spacing(mesh, h=0.1, pad=1.0, max_points=MAX_GRID_POINTS):
    """h, coarsened only when the 0.1 mm grid over the mesh box would exceed max_points."""
    ext = mesh.bounds[1] - mesh.bounds[0] + 2 * pad
    return max(h, float((np.prod(ext) / max_points) ** (1 / 3)))


def prepare_grid(mesh, zone, h=0.1, blend_mm=1.5, pad=1.0):
    """Grid, original signed distance and zone distance; shared by every tube mode."""
    lo, hi = mesh.bounds[0] - pad, mesh.bounds[1] + pad
    axes = [np.arange(lo[i], hi[i] + h, h) for i in range(3)]
    G = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    f0, face = signed_distance(G, mesh, return_face=True)
    dz = zone.distance(G, upper=blend_mm + 0.01, sd=(f0, face))
    return lo, axes, G, f0, dz


def heal_implicit(c, mesh, zone, bridges, h=0.1, blend_mm=1.0, carve_mm=0.3, k_union=0.25,
                  mode="profile", pad=1.0, combine="override", grid=None):
    """Blend the original with (original - zone) U tubes, then marching cubes."""
    from skimage import measure
    import trimesh
    if grid is None:
        grid = prepare_grid(mesh, zone, h, blend_mm, pad)
    lo, axes, G, f0, dz = grid
    w = smoothstep(dz / blend_mm)
    act = np.flatnonzero(w < 1)
    X = G[act]
    f_carved = np.maximum(f0[act], carve_mm - dz[act])
    tube = np.full(len(X), np.inf)
    W = np.zeros(len(X))
    for br in bridges:
        g, wb = bridge_field(br, X, mode=mode, return_weight=True)
        g = np.where(np.isnan(g), np.inf, g)
        tube = np.where(np.isinf(tube), g, np.where(np.isinf(g), tube, smooth_min(tube, g, k_union)))
        W = np.maximum(W, wb)
    if combine == "override":
        # where a bridge covers the point the tube alone decides; elsewhere the carved original
        f_heal = np.where(np.isinf(tube), f_carved, W * np.where(np.isinf(tube), 0, tube) + (1 - W) * f_carved)
    else:
        # smooth union: tube U (original minus zone); the neck bulge is inside the zone, so it
        # is carved, while other vessels crossing the bridge's reach are kept
        f_heal = np.where(np.isinf(tube), f_carved, smooth_min(tube, f_carved, k_union))
    f = f0.copy()
    f[act] = w[act] * f0[act] + (1 - w[act]) * f_heal
    vol = f.reshape([len(a) for a in axes])
    vv, ff, _, _ = measure.marching_cubes(vol, level=0.0, spacing=(h, h, h))
    out = trimesh.Trimesh(vv + lo, ff[:, ::-1], process=True)
    parts = out.split(only_watertight=False)
    if len(parts) > 1:
        out = max(parts, key=lambda p: len(p.faces))
    if out.volume < 0:
        out.invert()
    return out, dict(n_active=len(act), grid=[len(a) for a in axes])


# ---------------------------------------------------------------- one-call patch

def dedup_bridges(bridges, tol=0.3):
    """Paths sharing a trunk give the same bridge; keep one per anchor pair."""
    kept = []
    for b in bridges:
        if not any(np.linalg.norm(b["anchors"][0] - k["anchors"][0]) < tol and
                   np.linalg.norm(b["anchors"][1] - k["anchors"][1]) < tol for k in kept):
            kept.append(b)
    return kept


def heal_case(c, paths, collar_mm=1.0, blend_mm=1.5, h=0.1, mode="ellipse", target_faces=None,
              f_sac=None):
    """Remove the sac (+ collar) from c["mesh"] and patch it. Returns (healed, zone, bridges, notes).

    paths: [(pts, radius)] VMTK centerlines of the ORIGINAL mesh (used only to bridge the gap).
    f_sac: sac faces of c["mesh"] (e.g. from locate_sac_from_fit); default: label.nrrd label 2.
    notes contains "reaches a path end" when a stretch could not be bridged (mesh too short)."""
    mesh = c["mesh"]
    if f_sac is None:
        _, f_sac, _ = locate_sac(c)
    f_rm, _ = removal_zone(c, f_sac, collar_mm)
    zone = Zone(mesh, f_rm)
    f = lambda P: signed_distance(P, mesh)
    notes, bridges = [], []
    for pts, r in paths:
        bridges += bridge_path(pts, r, f, zone, log=notes)
    bridges = dedup_bridges(bridges)
    h = grid_spacing(mesh, h)
    if h > 0.1 + 1e-9:
        notes.append(f"grid spacing {h:.3f} mm (box too large for 0.1 mm)")
    grid = prepare_grid(mesh, zone, h, blend_mm)
    healed, _ = heal_implicit(c, mesh, zone, bridges, h=h, blend_mm=blend_mm, mode=mode, grid=grid)
    if target_faces is not None and len(healed.faces) > target_faces:
        healed = decimate(healed, target_faces)
    return healed, zone, bridges, notes, f_sac


def decimate(mesh, target_faces):
    import open3d as o3d
    import trimesh
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),
                                  o3d.utility.Vector3iVector(mesh.faces))
    m = m.simplify_quadric_decimation(int(target_faces))
    out = trimesh.Trimesh(np.asarray(m.vertices), np.asarray(m.triangles), process=True)
    return out
