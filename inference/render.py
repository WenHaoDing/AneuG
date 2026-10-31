"""3D renders of ostium-model inference, for checking by eye (pyvista through Xvfb).

overview(): the patch's artery surface, the graph coloured by site score, the core box and every
sampled aneurysm. sample(): one aneurysm close up from three directions: the dome (red), the rest
of the decoded complex (faint), the ring (blue), the centerline (black) and the site (yellow).
For aneurysm patches the real sac (label 2) is shown in orange.
"""
import numpy as np

_XVFB = False


def _pv():
    global _XVFB
    import pyvista as pv
    if not _XVFB:
        pv.start_xvfb()
        _XVFB = True
    return pv


def label_surface(seg, label, origin, spacing):
    """Marching-cubes surface of seg == label (seg in array order z, y, x) in the mm frame
    origin + ijk * spacing (i, j, k = x, y, z)."""
    from skimage.measure import marching_cubes
    pv = _pv()
    m = (seg == label).astype(np.float32)
    if m.sum() == 0:
        return None
    m = np.pad(m, 1)
    v, f, _, _ = marching_cubes(m, 0.5)
    xyz = (v - 1)[:, ::-1] * np.asarray(spacing) + np.asarray(origin)
    return pv.PolyData(xyz, np.c_[np.full(len(f), 3), f].ravel())


def _graph_lines(g):
    pv = _pv()
    ei = g["edge_index"]
    und = ei[:, ei[0] < ei[1]]
    return pv.PolyData(g["pos"], lines=np.c_[np.full(und.shape[1], 2), und.T].ravel())


def _mesh(verts, faces, mask=None):
    pv = _pv()
    if mask is not None:
        keep = mask[faces].all(1)
        faces = faces[keep]
    if len(faces) == 0:
        return None
    return pv.PolyData(verts, np.c_[np.full(len(faces), 3), faces].ravel())


def _box(lo, hi):
    pv = _pv()
    return pv.Box(bounds=(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))


def overview(path, g, score, outs, artery=None, sac=None, core_box=None, title=""):
    """Patch overview: artery surface, graph coloured by site score (core points), core box, all
    sampled aneurysms (dome red, ring blue). Two viewing directions."""
    pv = _pv()
    pl = pv.Plotter(off_screen=True, shape=(1, 2), window_size=(1600, 800), border=False)
    for c, view in enumerate(((1, -1, 0.6), (-1, -0.3, 0.4))):
        pl.subplot(0, c)
        if artery is not None:
            pl.add_mesh(artery, color="lightgrey", opacity=0.18)
        if sac is not None:
            pl.add_mesh(sac, color="orange", opacity=0.45)
        pts = pv.PolyData(g["pos"])
        pts["site score"] = np.where(g["in_core"], score, np.nan)
        pl.add_mesh(pts, scalars="site score", cmap="magma", point_size=5, render_points_as_spheres=True,
                    nan_color="grey", show_scalar_bar=(c == 0))
        if core_box is not None:
            pl.add_mesh(_box(*core_box), style="wireframe", color="red", line_width=1)
        for o in outs:
            dm = _mesh(o["verts_mm"], o["faces"], o["p_dome"] > 0.5)
            if dm is not None:
                pl.add_mesh(dm, color="crimson")
            pl.add_mesh(pv.Spline(np.r_[o["ring_mm"], o["ring_mm"][:1]], 200).tube(radius=0.08), color="royalblue")
        pl.camera_position = "iso"
        pl.camera.focal_point = tuple(np.mean(g["pos"][g["in_core"]], 0) if g["in_core"].any() else g["pos"].mean(0))
        pl.camera.position = tuple(np.array(pl.camera.focal_point) + 60 * np.asarray(view) / np.linalg.norm(view))
        pl.camera.up = (0, 0, 1)
        if c == 0 and title:
            pl.add_text(title, font_size=10)
    pl.screenshot(str(path))
    pl.close()


def sample(path, g, o, metrics, artery=None, sac=None, title=""):
    """One aneurysm close up, three directions: along the vessel (e1), across it (e2), oblique."""
    pv = _pv()
    R = o["frame"]                                  # rows e1, e2, n (ostium frame)
    c = o["ring_mm"].mean(0)
    span = max(12.0, 2.5 * float(np.ptp(o["verts_mm"], 0).max()))
    dirs = [("along vessel", R[0]), ("across vessel", R[1]), ("oblique", R[0] + R[1] + 0.8 * R[2])]
    pl = pv.Plotter(off_screen=True, shape=(1, 3), window_size=(1800, 640), border=False)
    for k, (name, d) in enumerate(dirs):
        pl.subplot(0, k)
        if artery is not None:
            bounds = (c[0] - span, c[0] + span, c[1] - span, c[1] + span, c[2] - span, c[2] + span)
            pl.add_mesh(artery.clip_box(bounds, invert=False), color="lightgrey", opacity=0.25)
        if sac is not None:
            pl.add_mesh(sac, color="orange", opacity=0.4)
        pl.add_mesh(_graph_lines(g).tube(radius=0.05), color="black")
        rest = _mesh(o["verts_mm"], o["faces"], o["p_dome"] <= 0.5)
        if rest is not None:
            pl.add_mesh(rest, color="steelblue", opacity=0.12)
        dm = _mesh(o["verts_mm"], o["faces"], o["p_dome"] > 0.5)
        if dm is not None:
            pl.add_mesh(dm, color="crimson", opacity=0.85)
        pl.add_mesh(pv.Spline(np.r_[o["ring_mm"], o["ring_mm"][:1]], 200).tube(radius=0.07), color="royalblue")
        pl.add_mesh(pv.Sphere(radius=0.25, center=g["pos"][o["site"]]), color="gold")
        pl.camera.focal_point = tuple(c)
        pl.camera.position = tuple(c + 1.6 * span * d / np.linalg.norm(d))
        pl.camera.up = tuple(R[2] if k < 2 else (0, 0, 1))
        pl.add_text(name, font_size=9, position="upper_right")
        if k == 0:
            items = [f"{a} {b}" for a, b in metrics.items() if a != "flags"]
            txt = title + "\n" + "\n".join("   ".join(items[i:i + 3]) for i in range(0, len(items), 3))
            if metrics.get("flags"):
                txt += "\nFLAGS: " + ", ".join(metrics["flags"])
            pl.add_text(txt, font_size=7, position="upper_left")
    pl.screenshot(str(path))
    pl.close()


COLORS = {1: (200, 200, 200), 2: (31, 79, 216), 3: (255, 140, 26)}       # artery, generated sac, real sac


def _label_surface_rgb(lab, origin_vox0, spacing, n_smooth=25, relax=0.1):
    """One marching-cubes surface of everything labelled (> 0) in lab (z, y, x), smoothed like the
    training surfaces (Laplacian, n_smooth iterations, relaxation relax), each vertex coloured by the
    label of the voxels around it (sac wins over artery)."""
    from skimage.measure import marching_cubes
    pv = _pv()
    m = np.pad((lab > 0).astype(np.float32), 1)
    if m.sum() == 0:
        return None
    v, f, _, _ = marching_cubes(m, 0.5)
    v = v - 1                                                               # back to lab's (z, y, x) index
    L = np.pad(lab, 1)
    vi = v + 1
    best = np.zeros(len(v), np.uint8)
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                idx = np.floor(vi).astype(int) + np.array([dz, dy, dx])
                idx = np.clip(idx, 0, np.array(L.shape) - 1)
                val = L[idx[:, 0], idx[:, 1], idx[:, 2]]
                rank = np.select([val == 2, val == 3, val == 1], [3, 2, 1], 0)  # generated > real > artery
                cur = np.select([best == 2, best == 3, best == 1], [3, 2, 1], 0)
                best = np.where(rank > cur, val, best)
    xyz = v[:, ::-1] * np.asarray(spacing) + np.asarray(origin_vox0)
    mesh = pv.PolyData(xyz, np.c_[np.full(len(f), 3), f].ravel())
    if n_smooth:
        mesh = mesh.smooth(n_iter=n_smooth, relaxation_factor=relax)
    mesh.point_data["rgb"] = np.array([COLORS.get(int(b), (200, 200, 200)) for b in best], np.uint8)
    return mesh


def voxels(path, seg, origin, spacing, focus_mm, title="", metrics=None, half_mm=None, original=None):
    """Merged patch mask around focus_mm, two rows by three directions:
      row 1  voxels: artery (label 1) faint grey, generated sac (label 2) blue, real sac orange
      row 2  marching cubes of artery + sac as one surface, smoothed like the training surfaces,
             coloured by label (same colours)
    With `original`, label-2 voxels already in the original mask (a real sac) are the orange ones."""
    pv = _pv()
    origin, spacing = np.asarray(origin, float), np.asarray(spacing, float)
    sac = seg == 2
    if half_mm is None:
        ext = 0.0
        if sac.any():
            zyx = np.argwhere(sac)
            pts = zyx[:, ::-1] * spacing + origin
            ext = float(np.abs(pts - focus_mm).max())
        half_mm = max(6.0, ext + 4.0)
    c = (np.asarray(focus_mm) - origin) / spacing                                   # (x, y, z) index
    lo = np.maximum(np.floor(c - half_mm / spacing).astype(int), 0)
    hi = np.minimum(np.ceil(c + half_mm / spacing).astype(int), np.array(seg.shape)[::-1] - 1)
    sub = seg[lo[2]:hi[2] + 1, lo[1]:hi[1] + 1, lo[0]:hi[0] + 1]
    lab = sub.astype(np.uint8)
    if original is not None:
        orig = original[lo[2]:hi[2] + 1, lo[1]:hi[1] + 1, lo[0]:hi[0] + 1]
        lab = np.where((sub == 2) & (orig == 2), 3, lab)
    nz, ny, nx = lab.shape
    vox0 = origin + lo * spacing                                                    # centre of lab[0, 0, 0]
    grid = pv.ImageData(dimensions=(nx + 1, ny + 1, nz + 1), spacing=tuple(spacing), origin=tuple(vox0 - spacing / 2))
    grid.cell_data["label"] = lab.ravel(order="C")                                  # x fastest, as VTK
    surf = _label_surface_rgb(lab, vox0, spacing)
    pl = pv.Plotter(off_screen=True, shape=(2, 3), window_size=(1800, 1240), border=False)
    views = [((1, 0, 0), (0, 0, 1)), ((0, 1, 0), (0, 0, 1)), ((1, 1, 1), (0, 0, 1))]
    f = np.asarray(focus_mm, float)
    for row in (0, 1):
        for k, (d, up) in enumerate(views):
            pl.subplot(row, k)
            if row == 0:
                for val, op in ((1, 0.22), (2, 1.0), (3, 1.0)):
                    part = grid.threshold([val - 0.5, val + 0.5], scalars="label")
                    if part.n_cells:
                        pl.add_mesh(part, color=COLORS[val], opacity=op, show_edges=(val != 1),
                                    edge_color="black", line_width=0.3)
            elif surf is not None:
                pl.add_mesh(surf, scalars="rgb", rgb=True, smooth_shading=True, specular=0.2)
            pl.camera.focal_point = tuple(f)
            pl.camera.position = tuple(f + 4.0 * half_mm * np.asarray(d) / np.linalg.norm(d))
            pl.camera.up = up
            pl.reset_camera()                               # fit the cropped region, keep the direction
            pl.add_text((["voxels", "marching cubes"][row] + " · " + ["view along x", "view along y", "oblique"][k]),
                        font_size=9, position="upper_right")
            if row == 0 and k == 0:
                txt = title
                if metrics:
                    items = [f"{a} {b}" for a, b in metrics.items() if a != "flags"]
                    txt += "\n" + "\n".join("   ".join(items[i:i + 3]) for i in range(0, len(items), 3))
                    if metrics.get("flags"):
                        txt += "\nFLAGS: " + ", ".join(metrics["flags"])
                pl.add_text(txt, font_size=7, position="upper_left")
    pl.screenshot(str(path))
    pl.close()


def _smooth_surface(mask, origin, spacing, n_smooth=25, relax=0.1):
    """Smoothed marching-cubes surface of a (z, y, x) mask in mm, or None if empty."""
    m = label_surface(mask.astype(np.uint8), 1, origin, spacing)
    return m.smooth(n_iter=n_smooth, relaxation_factor=relax) if m is not None and n_smooth else m


def patch_overview(path, seg, origin, spacing, sacs, skipped=None, real_sac=None, core_box=None, title="",
                   rejected=None, prompts=None):
    """The whole patch with every generated aneurysm, 2 x 2 views (along x, y, z and oblique).
    seg: (z, y, x) patch mask, its label 1 is drawn as the artery (grey). sacs: {sample: (z, y, x)
    bool mask} of the built sacs, drawn translucent blue and numbered. skipped: {sample: (dome
    verts, faces)} of generated domes that were not built, drawn faint red and numbered. real_sac:
    (z, y, x) bool mask of a real aneurysm (orange). core_box: (lo, hi) mm (red wireframe).
    rejected: {sample: mask} of built sacs that failed a check, drawn translucent red. prompts:
    {sample: mm position} of the prompt each sample was generated for (green ball, "P<sample>")."""
    pv = _pv()
    origin, spacing = np.asarray(origin, float), np.asarray(spacing, float)
    art = _smooth_surface(seg == 1, origin, spacing)
    real = _smooth_surface(real_sac, origin, spacing) if real_sac is not None and real_sac.any() else None
    blue = {k: _smooth_surface(m, origin, spacing, n_smooth=15) for k, m in sacs.items() if m.any()}
    rej = {k: _smooth_surface(m, origin, spacing, n_smooth=15) for k, m in (rejected or {}).items() if m.any()}
    red = {k: pv.PolyData(v, np.c_[np.full(len(f), 3), f].ravel()) for k, (v, f) in (skipped or {}).items() if len(f)}
    centre = origin + (np.array(seg.shape[::-1]) - 1) / 2 * spacing
    views = [("view along x", (1, 0, 0), (0, 0, 1)), ("view along y", (0, 1, 0), (0, 0, 1)),
             ("view along z (from above)", (0, 0, 1), (0, 1, 0)), ("oblique", (1, -1, 0.7), (0, 0, 1))]
    pl = pv.Plotter(off_screen=True, shape=(2, 2), window_size=(2400, 2400), border=False)
    for i, (name, d, up) in enumerate(views):
        pl.subplot(i // 2, i % 2)
        if art is not None:
            pl.add_mesh(art, color=np.array(COLORS[1]) / 255, smooth_shading=True, specular=0.2)
        if real is not None:
            pl.add_mesh(real, color=np.array(COLORS[3]) / 255, smooth_shading=True, opacity=0.7)
        for k, m in blue.items():
            pl.add_mesh(m, color=np.array(COLORS[2]) / 255, smooth_shading=True, opacity=0.55)
        for k, m in rej.items():
            pl.add_mesh(m, color="red", smooth_shading=True, opacity=0.55)
        for k, m in red.items():
            pl.add_mesh(m, color="crimson", opacity=0.25)
        lab_pts = [m.center for m in blue.values()] + [m.center for m in rej.values()] + [m.center for m in red.values()]
        lab_txt = [f"{k}" for k in blue] + [f"{k} (rejected)" for k in rej] + [f"{k} (not built)" for k in red]
        for k, q in (prompts or {}).items():
            pl.add_mesh(pv.Sphere(radius=0.45, center=tuple(q)), color="limegreen")
            lab_pts.append(np.asarray(q, float))
            lab_txt.append(f"P{k}")
        for pt, txt in zip(lab_pts, lab_txt):                 # one actor each: VTK hides overlapping labels
            pl.add_point_labels(np.array([pt]), [txt], font_size=22, point_size=1, shape_opacity=0.6,
                                always_visible=True, text_color="black", shape_color="white")
        if core_box is not None:
            pl.add_mesh(_box(*core_box), style="wireframe", color="red", line_width=1)
        pl.camera.focal_point = tuple(centre)
        pl.camera.position = tuple(centre + 200 * np.asarray(d, float) / np.linalg.norm(d))
        pl.camera.up = up
        pl.reset_camera()
        pl.add_text(name, font_size=11, position="upper_right")
        if i == 0 and title:
            pl.add_text(title, font_size=11, position="upper_left")
    pl.screenshot(str(path))
    pl.close()
