"""
Interactive manual cap-region labeling tool (PyVista). Run this on a machine
WITH A DISPLAY — this workstation is headless, so this script was written but
never actually run here; it's structured to be easy to debug locally (small,
separable functions, defensive checks with printed diagnostics where the
PyVista API is most likely to need a version-specific tweak — see
brush_one_branch's docstring).

Why this exists: dataset/preprocess_endcaps.py's automatic patch definition
(every mesh vertex within PATCH_THRESHOLD_MM=1.0 of GRAPH distance from the
ray-crossing endpoint) is a reasonable proxy, but a FIXED radius doesn't
adapt to how much vessel cross-sections actually vary in size — it over-covers
thin vessels and under-covers wide ones, so it doesn't always match a human's
own sense of where the cap really is. This script lets a human directly
brush/select the true cap region, per branch, for BOTH real cases (refining
preprocess_endcaps.py's automatic output) and synthetic cases (which have no
automatic label at all — there's no real centerline to derive one from).

Per-branch derivation from the brushed selection (no separate interaction
needed beyond the brush itself):
  - in_patch:  the set of vertices belonging to the brushed faces.
  - endpoint:  centroid of those vertices.
  - tangent:   outward surface normal of the brushed patch (SVD of the patch
               points, oriented away from the mesh's own centroid) — the same
               technique already used for real branch directions elsewhere in
               this codebase (MultiCanonicalGHDReconstruct.reconstruct_fused_mesh's
               _outward_normal, utils/../vessel_clipping.py's version of the
               same idea): the cap is roughly a disk cutting across the vessel
               tube, so the normal to that disk approximates the vessel's own
               axial direction reasonably well.

  For SYNTHETIC cases this derivation is used for ALL THREE fields — there's
  no other source. For REAL cases it's used for in_patch ONLY: endpoint and
  tangent are NOT recomputed from the brush, because preprocess_endcaps.py's
  ray-crossing-along-the-real-centerline already gives a better-grounded
  value than the centroid/normal of a rough hand-dragged patch — brushing
  exists to fix the fixed-radius CAP-REGION proxy, not to replace ground-truth
  endpoint/tangent geometry.

Reference geometry shown WHILE brushing (context only, not the label itself):
  - Real cases: the real (clipped) centerline for that branch (dashed-style
    line, matching preprocess_endcaps.py's per-branch black/blue/red
    convention), plus the AUTOMATIC endpoint/tangent (yellow arrow, same
    convention as every sanity panel in this project) and the automatic
    in_patch region (faint dots) — so the labeler can see what the automatic
    method guessed and correct it, rather than starting from nothing.
  - Synthetic cases: no real centerline exists (freshly generated, never
    labeled before). Instead, shows the FIXED canonical opening vertex
    indices (openings.npz, via MultiCanonicalGHDReconstruct._load_openings)
    as faint per-branch markers — these are NOT precise (this exact
    fixed-index assumption is what motivated building a learned endcap
    predictor in the first place; it can drift on a deformed/generated
    shape), they're only there so the labeler knows roughly which physical
    opening is "branch 0" vs. "branch 1" vs. "branch 2".

Real cases: no separate output folder. Manual labels are written back into
the SAME per-case checkpoint under runtime_dataset/AneuG_morpho/<case>.npy
that dataset/preprocess_endcaps.py already produces, as a new manual_in_patch
field ONLY — endpoints/tangents/branch_mask stay the automatic, ray-crossing-
derived values (see above), never overwritten. dataset/morpho_dataset.py's
MorphoDataset prefers manual_in_patch over automatic in_patch when present,
while always using the automatic endpoints/tangents/branch_mask. Resumable: a
case whose record already has manual_in_patch is skipped. (There's only ONE
kind of label source worth a folder split here — automatic vs. manual in_patch
is the same case, just refined — so it doesn't get one. The rare case where a
real case has NO automatic record at all falls back to a full manual_endpoints/
manual_tangents/manual_branch_mask/manual_in_patch set instead, since there's
nothing to defer to; MorphoDataset falls back to those per-field too.)

Synthetic cases: a genuinely different population (no real centerline, no
automatic record, no case in dataset/processed/ to attach to), so they get
their OWN folder, runtime_dataset/AneuG_morpho_synthetic/, using the
plain (non-"manual_"-prefixed) schema fields directly — there's nothing
automatic to preserve alongside a synthetic case. Keeping synthetic cases out
of AneuG_morpho/ also means a real preprocess_endcaps.py rerun (e.g.
after tuning PATCH_THRESHOLD_MM) can't accidentally interact with them.
MorphoDataset accepts a list of roots, so combining real + synthetic for
training is one line: MorphoDataset([AneuG_morpho, AneuG_morpho_synthetic]).

conda activate new   (needs pyvista with a real display / working GL context —
will not work over a plain SSH session without X forwarding or a virtual
framebuffer)
python dataset/label_morpho.py --mode real
python dataset/label_morpho.py --mode synthetic --n-synthetic 20 --seed 0
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.preprocess_assembled import reconstruct
from dataset.auto_uncap import compute_caps
from dataset.dome_label import dome_proposal

MAX_BRANCHES = 3
TYPE_N_OPEN = {0: 3, 1: 2, 2: 2}   # candidate branch count per type — type 2 merged into type 1 elsewhere
                                    # in this pipeline, but the mesh geometry (Sidewall template) is identical
                                    # either way, so labeling only ever needs 2 or 3 branches per case.

DEFAULT_REAL_DIR       = ROOT / "runtime_dataset" / "AneuG_processed"                    # source of real centerlines (assembled corpus)
DEFAULT_ENDCAPS_DIR    = ROOT / "runtime_dataset" / "AneuG_morpho"             # read automatic labels from AND write manual labels into (real mode)
DEFAULT_SYNTHETIC_DIR  = ROOT / "runtime_dataset" / "AneuG_morpho_synthetic"    # write-only, synthetic mode
CANONICAL_ROOT         = ROOT / "dataset" / "canonical"
GHD_VAE_CKPT = ROOT / "runtime" / "tr_checkpoints" / "v2_1" / "stage1" / "ghd_vae_h512_z16_kl0.5" / "epoch_05000.pth"

BRANCH_COLORS = ["black", "blue", "red"]


def _outward_normal(points, mesh_centroid):
    """SVD-based outward surface normal of a point patch — same technique as
    MultiCanonicalGHDReconstruct.reconstruct_fused_mesh's _outward_normal."""
    centroid = points.mean(axis=0)
    _, _, Vt = np.linalg.svd(points - centroid, full_matrices=False)
    normal = Vt[-1]
    if np.dot(normal, centroid - mesh_centroid) < 0:
        normal = -normal
    return normal / np.linalg.norm(normal)


def _faces_to_pv(faces):
    """[F, 3] int array -> PyVista's flat padded face format."""
    return np.hstack([np.full((len(faces), 1), 3), faces]).ravel()


def brush_region(verts, faces, label, add_reference):
    """Brush ONE arbitrary region (used for the dome). Thin wrapper over
    brush_one_branch so the picking/highlight/confirm machinery has a single
    implementation -- the branch version is left untouched."""
    return brush_one_branch(verts, faces, 0, 1, add_reference, label=label)


def brush_one_branch(verts, faces, branch_idx, n_branches, add_reference, label=None):
    """Opens ONE fresh interactive window for a single branch's cap brushing
    (a fresh plotter per branch, not one shared/reused window across
    branches — simpler and more robust to reason about without a display to
    test against locally).

    add_reference(plotter): callback that adds any reference/guide geometry
    (real centerline, automatic endpoint/tangent/patch, or fixed opening
    markers) to the plotter before picking starts — see _real_reference /
    _synthetic_reference below.

    Interaction: LEFT-CLICK-DRAG a box over the mesh surface to select faces
    (through=False restricts selection to the nearest/visible surface, not
    cells behind it — appropriate for "painting" an external cap region);
    repeat (drag again) to add more to the running selection; 'z' clears it
    and starts over; 'c' confirms and closes this branch's window. (Not 'r'
    for clear — enable_cell_picking's own rubber-band-select interactor style
    already uses 'r' as a built-in toggle between camera-rotate and
    box-select mode, so binding "clear" there too would collide with it.)

    _on_pick tries both "original_cell_ids" (PyVista >=0.44's
    enable_rectangle_visible_picking) and "vtkOriginalCellIds" (older
    versions) for picked.cell_data's key holding the original mesh's cell
    indices. If its warning fires anyway (prints picked.array_names), add
    whatever key your PyVista version actually uses.

    Returns a sorted list of picked face (cell) indices, or None if the
    window was closed without confirming a non-empty selection (skip this
    branch/case — rerun the script later to retry, since output is only
    saved once ALL of a case's branches are done).
    """
    import pyvista as pv

    mesh = pv.PolyData(verts, _faces_to_pv(faces))
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color="whitesmoke", opacity=0.55, show_edges=False)
    add_reference(plotter)

    picked_cells = set()
    state = {"highlight": None, "confirmed": False}

    def _refresh_highlight():
        if state["highlight"] is not None:
            plotter.remove_actor(state["highlight"])
            state["highlight"] = None
        if picked_cells:
            sub = mesh.extract_cells(sorted(picked_cells))
            # pickable=False: this highlight is re-added on every pick, so if it stayed
            # pickable a later drag could hit it too and enable_cell_picking would then
            # hand _on_pick a MultiBlock (one block per hit actor) instead of the single
            # mesh it expects -- see _on_pick's MultiBlock handling below for the same
            # reason applied to the static reference geometry.
            state["highlight"] = plotter.add_mesh(sub, color="orange", opacity=0.95,
                                                    show_edges=True, pickable=False)

    def _on_pick(picked):
        if picked is None:
            return
        if isinstance(picked, pv.MultiBlock):
            # Only reached if some non-mesh actor slipped through without pickable=False;
            # combine() merges all hit blocks back into one mesh so the rest of this
            # function doesn't need to special-case it.
            picked = picked.combine()
        if picked.n_cells == 0:
            return
        # PyVista 0.48's enable_rectangle_visible_picking stores original cell indices
        # under "original_cell_ids" (not "vtkOriginalCellIds" as in older versions/docs);
        # try both so this keeps working across versions.
        ids = picked.cell_data.get("original_cell_ids")
        if ids is None:
            ids = picked.cell_data.get("vtkOriginalCellIds")
        if ids is None:
            print(f"[label_morpho] no original-cell-id array in picked.cell_data; "
                  f"available arrays: {picked.array_names}. Edit _on_pick in "
                  f"dataset/label_morpho.py to use the right key for your PyVista version.")
            return
        picked_cells.update(int(i) for i in np.asarray(ids))
        _refresh_highlight()

    def _reset():
        picked_cells.clear()
        _refresh_highlight()

    def _confirm():
        state["confirmed"] = True
        plotter.close()

    plotter.add_text(
        (label + ": " if label else
         f"Branch {branch_idx + 1}/{n_branches}: ") + "LEFT-CLICK-DRAG a box over the "
        f"region (repeat to add more; if it wraps out of view, 'r' toggles ROTATE vs. "
        f"SELECT mode -- through=False only picks the visible surface), 'z' clears, 'c' confirms",
        font_size=11, position="upper_left",
    )
    # show=False: don't let PyVista draw its OWN highlight for the picked cells -- that
    # highlight only ever shows the LATEST drag's selection (not the running total), which
    # fights visually with _refresh_highlight's cumulative orange one above. show_message=True
    # keeps its on-screen instructional text (separate from `show`), since that's still useful
    # and accurate for whatever version is installed locally. 'r' below is otherwise ambiguous:
    # enable_cell_picking's own rubber-band-select interactor style ALSO binds 'r' as a
    # built-in (toggles between camera-rotate and box-select mode) -- binding "clear" to
    # 'r' as well collided with that, so clearing uses 'z' instead to stay unambiguous.
    plotter.enable_cell_picking(callback=_on_pick, through=False, show=False, show_message=True)
    plotter.add_key_event("z", _reset)
    plotter.add_key_event("c", _confirm)
    plotter.show()

    if not state["confirmed"] or not picked_cells:
        return None
    return sorted(picked_cells)


def review_auto_labels(verts, faces, caps, cap_id, dome, branch_points, n_open, case, info):
    """Show the automatic CAP and DOME regions and ask what to keep.

    Returns a set of regions to brush by hand -- {} to accept everything,
    {"caps"}, {"dome"}, or {"caps","dome"} -- or None to skip the case.

    Caps and dome are answered separately because they fail independently: the
    plane cut can be wrong while the dome is fine, and vice versa, and
    re-brushing a region that was already correct is wasted effort.
    """
    import pyvista as pv

    mesh = pv.PolyData(verts, _faces_to_pv(faces))
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color="whitesmoke", opacity=0.45, show_edges=False)

    if dome is not None and dome.any():
        plotter.add_points(verts[dome], color="crimson", point_size=7,
                           render_points_as_spheres=True, pickable=False)
    cap_colors = ["red", "green", "blue"]
    for b in range(n_open):
        m = caps & (cap_id == b)
        if m.any():
            plotter.add_points(verts[m], color=cap_colors[b % 3], point_size=9,
                               render_points_as_spheres=True, pickable=False)
        if b < len(branch_points) and branch_points[b] is not None and len(branch_points[b]) > 1:
            cl = np.asarray(branch_points[b], dtype=float)[:60]
            plotter.add_lines(np.repeat(cl, 2, axis=0)[1:-1],
                              color=BRANCH_COLORS[b % len(BRANCH_COLORS)], width=3)

    state = {"choice": None}

    def _set(choice):
        def _fn():
            state["choice"] = choice
            plotter.close()
        return _fn

    summary = "  ".join(
        f"b{o['opening']}:{o['n_cap_verts']}v" + ("" if o["used_crossing"] else "(no-cross)")
        for o in info)
    dome_txt = (f"dome {dome.mean():.0%} of verts" if dome is not None and dome.any()
                else "dome: NO automatic proposal")
    plotter.add_text(
        f"{case}: AUTOMATIC  caps ({summary})   {dome_txt}\n"
        f"'a' ACCEPT both  |  'c' rebrush CAPS  |  'd' rebrush DOME  |  "
        f"'b' rebrush BOTH  |  'q'/close SKIP",
        font_size=11, position="upper_left",
    )
    plotter.add_key_event("a", _set(frozenset()))
    plotter.add_key_event("c", _set(frozenset({"caps"})))
    plotter.add_key_event("d", _set(frozenset({"dome"})))
    plotter.add_key_event("b", _set(frozenset({"caps", "dome"})))
    plotter.show()
    return state["choice"]


def _add_arrow_and_point(plotter, origin, direction, color_point, color_arrow="yellow", scale=2.0):
    import pyvista as pv
    # pickable=False: reference-only geometry must never be selectable, or a brush drag
    # near it turns _on_pick's `picked` into a MultiBlock instead of the mesh's own
    # UnstructuredGrid (see _on_pick).
    plotter.add_mesh(pv.Sphere(radius=0.15, center=origin), color=color_point, pickable=False)
    plotter.add_mesh(pv.Arrow(start=origin, direction=direction, scale=scale), color=color_arrow, pickable=False)


def _real_reference(plotter, branch_idx, centerline_pts, auto_endpoint, auto_tangent, auto_patch_pts):
    """Real-case reference: dashed-style real centerline (per-branch color),
    automatic endpoint (same color, sphere) + tangent (yellow arrow, matching
    every sanity panel's convention in this project), automatic patch (faint dots)."""
    import pyvista as pv
    c = BRANCH_COLORS[branch_idx % len(BRANCH_COLORS)]
    if centerline_pts is not None and len(centerline_pts) > 1:
        line = pv.lines_from_points(centerline_pts[:60])   # cap length shown, matches preprocess_endcaps.py
        plotter.add_mesh(line, color=c, line_width=3, pickable=False)
    if auto_endpoint is not None and auto_tangent is not None:
        _add_arrow_and_point(plotter, auto_endpoint, auto_tangent, color_point=c)
    if auto_patch_pts is not None and len(auto_patch_pts) > 0:
        plotter.add_points(auto_patch_pts, color=c, point_size=6, opacity=0.35, pickable=False)


def _synthetic_reference(plotter, branch_idx, opening_pts):
    """Synthetic-case reference: faint markers at the FIXED canonical opening
    indices for this branch slot — approximate, for branch identity only."""
    c = BRANCH_COLORS[branch_idx % len(BRANCH_COLORS)]
    if opening_pts is not None and len(opening_pts) > 0:
        plotter.add_points(opening_pts, color=c, point_size=10, opacity=0.5, pickable=False)
        plotter.add_text(
            f"faint {c} dots (branch {branch_idx}) = approximate canonical opening, NOT precise",
            position="lower_left", font_size=9,
        )


def _derive_label(verts, faces, picked_face_ids, mesh_centroid):
    patch_faces = faces[picked_face_ids]
    patch_vertex_idx = np.unique(patch_faces.ravel())
    patch_pts = verts[patch_vertex_idx]
    endpoint = patch_pts.mean(axis=0).astype(np.float32)
    tangent = _outward_normal(patch_pts, mesh_centroid).astype(np.float32)
    return patch_vertex_idx, endpoint, tangent


def save_label_sanity(verts, faces, in_patch, dome, case, save_path, n_open=3,
                      endpoints=None, tangents=None, branch_mask=None):
    """Render the labelled regions so a bulk auto-accept pass is still
    inspectable afterwards. Matplotlib/Agg only -- no GL, so this works on the
    headless box where --auto-accept is actually used."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    cap_any = np.zeros(len(verts), dtype=bool)
    for b in range(min(n_open, in_patch.shape[0])):
        cap_any |= in_patch[b]
    fig = plt.figure(figsize=(19, 6.6))
    colors = ["#3cb44b", "#4363d8", "#f58231"]
    for k, (el, az) in enumerate(((18, 30), (18, 150), (55, 270))):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        plain = ~dome[faces].any(1) & ~cap_any[faces].any(1)
        ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces[plain],
                        color="lightgray", edgecolor="none", alpha=0.30)
        dm = dome[faces].all(1)
        if dm.any():
            ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces[dm],
                            color="crimson", edgecolor="none", alpha=0.75)
        for b in range(min(n_open, in_patch.shape[0])):
            if in_patch[b].any():
                ax.scatter(*verts[in_patch[b]].T, s=7, color=colors[b % 3])
        # Cross-section normal per opening: the tangent the model has to
        # predict. Drawn at the cap's own endpoint, scaled to the mesh so it
        # stays visible on small and large cases alike -- an arrow pointing
        # the wrong way is far easier to catch here than in the numbers.
        if endpoints is not None and tangents is not None:
            L = 0.18 * float((verts.max(0) - verts.min(0)).max())
            for b in range(min(n_open, len(endpoints))):
                if branch_mask is not None and not bool(branch_mask[b]):
                    continue
                o = np.asarray(endpoints[b], dtype=float)
                d = np.asarray(tangents[b], dtype=float)
                if not np.isfinite(d).all() or np.linalg.norm(d) < 1e-8:
                    continue
                d = d / np.linalg.norm(d)
                ax.quiver(o[0], o[1], o[2], d[0], d[1], d[2], length=L,
                          color=colors[b % 3], linewidth=2.2, arrow_length_ratio=0.28)
                ax.scatter(*o, s=45, color=colors[b % 3], edgecolor="black", zorder=6)
        lo, hi = verts.min(0), verts.max(0)
        c, r = (lo + hi) / 2, float((hi - lo).max() / 2)
        ax.set_xlim(c[0]-r, c[0]+r); ax.set_ylim(c[1]-r, c[1]+r); ax.set_zlim(c[2]-r, c[2]+r)
        ax.view_init(elev=el, azim=az); ax.set_axis_off()
        ax.set_title(f"elev {el} azim {az}", fontsize=8)
    fig.suptitle(f"{case}   dome (crimson) {dome.mean():.1%} of vertices   |   "
                 f"caps (green/blue/orange) {cap_any.mean():.1%}", fontsize=11)
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=140)
    plt.close(fig)
    return save_path


def _save_real_record(rec, endcaps_path, endcaps_rec, manual_in_patch,
                      manual_endpoints, manual_tangents, manual_branch_mask,
                      source, manual_dome=None, dome_source="none"):
    """Write manual_in_patch (and, only when there is no automatic record to
    defer to, the brush/auto-derived endpoint geometry) back into the case."""
    has_ground_truth = endcaps_rec is not None
    out = dict(endcaps_rec) if has_ground_truth else {
        "case": rec["case"], "aneurysm_type": int(rec["aneurysm_type"]),
        "phi": np.asarray(rec["ghd"]["phi"], dtype=np.float32),
    }
    out["manual_in_patch"] = manual_in_patch
    out["in_patch_source"] = source          # "auto" or "manual" -- provenance
    if manual_dome is not None:
        out["manual_dome"] = np.asarray(manual_dome, dtype=bool)
        out["dome_source"] = dome_source
    if not has_ground_truth:
        out["manual_endpoints"] = manual_endpoints
        out["manual_tangents"] = manual_tangents
        out["manual_branch_mask"] = manual_branch_mask
    np.save(endcaps_path, out, allow_pickle=True)
    print(f"[label_morpho] updated {endcaps_path} ({source} labels)")
    return True


def label_real_case(rec, endcaps_path, endcaps_rec, use_auto=True, plane_frac=0.5,
                    assembled_root=None, auto_accept=False, sanity_dir=None):
    """rec: runtime/dataset/processed/<case>.npy checkpoint (dict). endcaps_rec:
    runtime_dataset/AneuG_morpho/<case>.npy record (dict) to update IN PLACE
    and re-save to endcaps_path.

    Brushing only refines manual_in_patch (the cap-region classification
    target) — it does NOT overwrite endpoints/tangents. Those are already
    derived from ray-crossing along the REAL patient centerline
    (preprocess_endcaps.py's find_branch_endpoint), which is better-grounded
    than the centroid/normal of a hand-dragged patch; brushing exists to fix
    the fixed-radius cap-region proxy, not to replace ground-truth geometry.
    (Fallback: if this case somehow has no automatic record at all — endcaps_rec
    is None, shouldn't normally happen since preprocess_endcaps.py covers every
    real case — there's no ground truth to defer to, so the brush-derived
    endpoint/tangent become manual_endpoints/manual_tangents/manual_branch_mask
    as the only available source; MorphoDataset falls back to those per-field
    when the plain endpoints/tangents/branch_mask are absent.)"""
    case = rec["case"]
    atype = int(rec["aneurysm_type"])
    n_open = TYPE_N_OPEN.get(atype, MAX_BRANCHES)
    has_ground_truth = endcaps_rec is not None

    verts, faces = reconstruct(rec)
    mesh_centroid = verts.mean(axis=0)
    branch_points = rec["clipped_centerline"]["branch_points"][:n_open]

    manual_in_patch = np.zeros((MAX_BRANCHES, len(verts)), dtype=bool)
    manual_endpoints = np.zeros((MAX_BRANCHES, 3), dtype=np.float32)
    manual_tangents = np.zeros((MAX_BRANCHES, 3), dtype=np.float32)
    manual_branch_mask = np.zeros(MAX_BRANCHES, dtype=bool)

    # Automatic pass first: dataset/auto_uncap.py cuts each opening with a
    # plane whose normal is the branch tangent, restricted to that opening's
    # geodesic region with the dome excluded. It is right for the large
    # majority of cases, so brushing is only reached when a human looks at it
    # and says no.
    manual_dome = np.zeros(len(verts), dtype=bool)
    dome_source = "none"
    rebrush = {"caps", "dome"}          # what still needs a human; auto may clear it

    if use_auto:
        try:
            caps, cap_id, info, _, _ = compute_caps(rec, verts, faces, plane_frac=plane_frac)
        except Exception as exc:
            print(f"[label_morpho] {case}: automatic caps unavailable ({exc}) -- brushing")
            caps = cap_id = info = None
        # No geometry directory -> no dome proposal. That is the normal answer
        # for a GENERATED shape, so it falls through to brushing rather than
        # failing; only real cases have dome_sac.ply / label.nrrd to derive from.
        dome_auto = None
        # An existing label WINS over recomputing. Recomputation needs the raw
        # geometry (dome_sac.ply / label.nrrd), which is deliberately not part
        # of what gets downloaded for labelling -- so on the labelling machine
        # the already-computed mask is the only source, and re-deriving it
        # would force every dome to be brushed from scratch.
        if endcaps_rec is not None and endcaps_rec.get("manual_dome") is not None:
            dome_auto = np.asarray(endcaps_rec["manual_dome"], dtype=bool)
        else:
            try:
                dome_auto = dome_proposal(rec.get("dataset", ""), case, verts, assembled_root)
            except Exception as exc:
                print(f"[label_morpho] {case}: automatic dome unavailable ({exc})")

        if caps is not None:
            if auto_accept:
                # Non-interactive: take the automatic proposals as-is. For bulk
                # labelling of a corpus whose proposals have already been spot-
                # checked; a case missing either proposal is SKIPPED rather than
                # saved half-labelled, since there is no human here to brush it.
                if dome_auto is None:
                    print(f"[label_morpho] {case}: no dome proposal -- skipped "
                          f"(--auto-accept cannot brush)")
                    return False
                choice = frozenset()
            else:
                choice = review_auto_labels(verts, faces, caps, cap_id, dome_auto,
                                            branch_points, n_open, case, info)
            if choice is None:
                print(f"[label_morpho] {case}: skipped at review -- not saved.")
                return False
            rebrush = set(choice)
            if "caps" not in rebrush:
                for b in range(n_open):
                    m = caps & (cap_id == b)
                    manual_in_patch[b, np.flatnonzero(m)] = True
                    if m.any():
                        pts = verts[m]
                        manual_endpoints[b] = pts.mean(axis=0)
                        manual_tangents[b] = _outward_normal(pts, mesh_centroid)
                        manual_branch_mask[b] = True
            if "dome" not in rebrush and dome_auto is not None:
                manual_dome = np.asarray(dome_auto, dtype=bool)
                dome_source = "auto"
            elif dome_auto is None:
                rebrush.add("dome")     # nothing to accept -- must be brushed

    # HARD GUARD: --auto-accept runs headless, where any PyVista window
    # segfaults the process and takes the whole bulk pass down with it (this
    # happened at 124/525). Anything that would need a human is skipped and
    # named, never brushed.
    if auto_accept and rebrush:
        print(f"[label_morpho] {case}: needs manual work for {sorted(rebrush)} "
              f"-- skipped (--auto-accept cannot brush)")
        return False

    if "dome" in rebrush:
        picked = brush_region(
            verts, faces, f"{case} DOME",
            add_reference=lambda p, d=(dome_auto if use_auto else None):
                (p.add_points(verts[d], color="crimson", point_size=7,
                              render_points_as_spheres=True, pickable=False)
                 if d is not None and d.any() else None),
        )
        if picked is None:
            print(f"[label_morpho] {case}: dome skipped (no confirmed selection) -- not saved.")
            return False
        manual_dome[np.unique(faces[picked].ravel())] = True
        dome_source = "manual"

    if "caps" not in rebrush:
        ok = _save_real_record(rec, endcaps_path, endcaps_rec, manual_in_patch,
                               manual_endpoints, manual_tangents,
                               manual_branch_mask, "auto", manual_dome, dome_source)
        if ok and sanity_dir is not None:
            save_label_sanity(verts, faces, manual_in_patch, manual_dome, case,
                              Path(sanity_dir) / f"{case}.png", n_open,
                              endpoints=manual_endpoints, tangents=manual_tangents,
                              branch_mask=manual_branch_mask)
        return ok

    for b in range(n_open):
        cl_pts = branch_points[b] if b < len(branch_points) else None
        auto_ep = endcaps_rec["endpoints"][b] if has_ground_truth else None
        auto_tg = endcaps_rec["tangents"][b] if has_ground_truth else None
        auto_patch_pts = (verts[endcaps_rec["in_patch"][b]]
                          if has_ground_truth and endcaps_rec["in_patch"][b].any() else None)

        picked = brush_one_branch(
            verts, faces, b, n_open,
            add_reference=lambda p, b=b, cl=cl_pts, ae=auto_ep, at=auto_tg, ap=auto_patch_pts:
                _real_reference(p, b, cl, ae, at, ap),
        )
        if picked is None:
            print(f"[label_morpho] {case} branch {b}: skipped (no confirmed selection) — case not saved.")
            return False

        patch_idx, endpoint, tangent = _derive_label(verts, faces, picked, mesh_centroid)
        manual_in_patch[b, patch_idx] = True
        manual_endpoints[b] = endpoint
        manual_tangents[b] = tangent
        manual_branch_mask[b] = True

    ok = _save_real_record(rec, endcaps_path, endcaps_rec, manual_in_patch,
                           manual_endpoints, manual_tangents, manual_branch_mask,
                           "manual", manual_dome, dome_source)
    if ok and sanity_dir is not None:
        save_label_sanity(verts, faces, manual_in_patch, manual_dome, case,
                          Path(sanity_dir) / f"{case}.png", n_open,
                          endpoints=manual_endpoints, tangents=manual_tangents,
                          branch_mask=manual_branch_mask)
    return ok


def label_synthetic_case(case_id, atype, phi, multi_recon, synthetic_dir):
    """No automatic record exists for a fresh synthetic sample, so this writes
    a brand-new record directly into synthetic_dir (runtime_dataset/AneuG_morpho_synthetic/,
    kept separate from the real cases' AneuG_morpho/ — see module docstring),
    using the plain (non-"manual_"-prefixed) schema — there's nothing automatic
    to preserve alongside it."""
    # Synthetic samples come from the GHD VAE decoder: phi only, with no s_can
    # and no Stage-2 pose, so they are NOT rebuilt the way a fitted case is.
    # Imported locally -- the real path deliberately no longer depends on the
    # old preprocess_ImperialNHS reconstruction.
    from dataset.preprocess_ImperialNHS import _reconstruct_ghd_numpy
    verts, faces = _reconstruct_ghd_numpy(
        {"aneurysm_type": atype, "ghd": {"phi": phi}}, denormalize_shape=True)
    mesh_centroid = verts.mean(axis=0)
    n_open = TYPE_N_OPEN.get(atype, MAX_BRANCHES)
    opening_indices = multi_recon._load_openings(atype)   # list of index tensors, len == n_open for this type

    endpoints = np.zeros((MAX_BRANCHES, 3), dtype=np.float32)
    tangents = np.zeros((MAX_BRANCHES, 3), dtype=np.float32)
    branch_mask = np.zeros(MAX_BRANCHES, dtype=bool)
    in_patch = np.zeros((MAX_BRANCHES, len(verts)), dtype=bool)

    for b in range(n_open):
        idx = opening_indices[b].detach().cpu().numpy() if b < len(opening_indices) else None
        opening_pts = verts[idx] if idx is not None else None

        picked = brush_one_branch(
            verts, faces, b, n_open,
            add_reference=lambda p, b=b, op=opening_pts: _synthetic_reference(p, b, op),
        )
        if picked is None:
            print(f"[label_morpho] {case_id} branch {b}: skipped (no confirmed selection) — case not saved.")
            return False

        patch_idx, endpoint, tangent = _derive_label(verts, faces, picked, mesh_centroid)
        endpoints[b] = endpoint
        tangents[b] = tangent
        branch_mask[b] = True
        in_patch[b, patch_idx] = True

    np.save(synthetic_dir / f"{case_id}.npy", {
        "case": case_id, "aneurysm_type": atype,
        "phi": np.asarray(phi, dtype=np.float32),
        "endpoints": endpoints, "tangents": tangents,
        "branch_mask": branch_mask, "in_patch": in_patch,
        "is_synthetic": True,
    }, allow_pickle=True)
    print(f"[label_morpho] saved {case_id}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["real", "synthetic"], required=True)
    parser.add_argument("--real-dir", default=str(DEFAULT_REAL_DIR))
    parser.add_argument("--endcaps-dir", default=str(DEFAULT_ENDCAPS_DIR),
                         help="Real mode only: read automatic labels from here and write manual labels back "
                              "into the same per-case record — no separate output folder.")
    parser.add_argument("--synthetic-dir", default=str(DEFAULT_SYNTHETIC_DIR),
                         help="Synthetic mode only: where labeled synthetic cases are written. Kept separate "
                              "from --endcaps-dir since synthetic cases have no automatic counterpart there.")
    parser.add_argument("--case", action="append", dest="cases", help="Real mode: label only these cases.")
    parser.add_argument("--n-synthetic", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", help="Only used in synthetic mode, to run the frozen GHD VAE.")
    parser.add_argument("--no-auto", dest="auto", action="store_false",
                        help="Real mode: skip the automatic cap pass and brush every case "
                             "by hand, as before.")
    parser.add_argument("--plane-frac", type=float, default=0.5,
                        help="Real mode: where the automatic cutting plane sits between the "
                             "centerline start (0) and where the centerline leaves the mesh "
                             "(1). Higher clips less. Default 0.5.")
    parser.add_argument("--auto-accept", action="store_true",
                        help="Accept the automatic cap+dome proposals with NO interactive "
                             "review. For bulk labelling on a headless machine; a case "
                             "missing either proposal is skipped, not half-saved.")
    parser.add_argument("--assembled-root", default=str(ROOT / "runtime_dataset" / "assembled"),
                        help="Assembled corpus, where the precomputed dome_mask.npy lives.")
    parser.add_argument("--no-sanity", action="store_true",
                        help="Skip the per-case label render. They are the only way to "
                             "inspect a bulk --auto-accept pass after the fact.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after this many cases (smoke tests).")
    parser.set_defaults(auto=True)
    args = parser.parse_args()

    if args.mode == "real":
        endcaps_dir = Path(args.endcaps_dir)
        endcaps_dir.mkdir(parents=True, exist_ok=True)
        real_dir = Path(args.real_dir)
        paths = ([real_dir / f"{c}.npy" for c in args.cases] if args.cases
                 else sorted(real_dir.glob("*.npy")))
        n_done = 0
        for path in paths:
            if args.limit is not None and n_done >= args.limit:
                break
            if not path.exists():
                print(f"[SKIP] {path} not found"); continue
            case = path.stem
            endcaps_path = endcaps_dir / f"{case}.npy"
            endcaps_rec = np.load(endcaps_path, allow_pickle=True).item() if endcaps_path.exists() else None
            # Resume skips a case only when a HUMAN has already judged it.
            # An auto-accepted record is exactly what the interactive pass
            # exists to review, so it must not be skipped here.
            if (endcaps_rec is not None and "manual_in_patch" in endcaps_rec
                    and endcaps_rec.get("in_patch_source") != "auto"
                    and endcaps_rec.get("dome_source") != "auto"):
                continue   # already reviewed by hand -- resumable
            rec = np.load(path, allow_pickle=True).item()
            if label_real_case(rec, endcaps_path, endcaps_rec,
                               use_auto=args.auto, plane_frac=args.plane_frac,
                               assembled_root=args.assembled_root,
                               auto_accept=args.auto_accept,
                               sanity_dir=(None if args.no_sanity
                                           else Path(args.endcaps_dir) / "sanity")):
                n_done += 1
        print(f"Done. Manual labels written into {endcaps_dir}")

    else:
        import torch
        from utils.generate_synthetic import load_ghd_vae
        from models.multi_canonical_ghd_reconstruct import MultiCanonicalGHDReconstruct

        synthetic_dir = Path(args.synthetic_dir)
        synthetic_dir.mkdir(parents=True, exist_ok=True)

        device = torch.device(args.device)
        multi_recon = MultiCanonicalGHDReconstruct(CANONICAL_ROOT, device=device)
        ghd_vae, ghd_mean, ghd_std, ghd_input_dim = load_ghd_vae(GHD_VAE_CKPT, device)
        for p in ghd_vae.parameters():
            p.requires_grad_(False)

        g = torch.Generator(device="cpu").manual_seed(args.seed)
        types = torch.randint(0, 2, (args.n_synthetic,), generator=g)   # type 2 merged into 1 -- never sampled
        z_ghd = torch.randn(args.n_synthetic, ghd_vae.latent_dim, generator=g)
        with torch.no_grad():
            ghd_n, _ = ghd_vae.decode(z_ghd.to(device), types.to(device))
            phi_all = (ghd_n * ghd_std[:, :ghd_input_dim] + ghd_mean[:, :ghd_input_dim]).reshape(args.n_synthetic, -1, 3)

        for i in range(args.n_synthetic):
            case_id = f"synthetic_seed{args.seed}_{i:04d}"
            if (synthetic_dir / f"{case_id}.npy").exists():
                continue   # resumable
            label_synthetic_case(case_id, int(types[i]), phi_all[i].cpu().numpy(), multi_recon, synthetic_dir)
        print(f"Done. Manual labels written into {synthetic_dir}")


if __name__ == "__main__":
    main()
