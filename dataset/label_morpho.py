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
case is skipped once its record's "reviewed" flag is set -- written whenever
a human made the call interactively, be it a full hand-brush or pressing 'a'
in review_auto_labels to accept the automatic proposal as-is (see main()'s
skip condition for the exact rule, including the pre-"reviewed" fallback for
older records). A --auto-accept record, with no human involved, is NOT
skipped: that's the population the interactive pass exists to review. (There's
only ONE
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
python dataset/label_morpho.py --mode synthetic \
  --ghd-vae runtime_train/ghd_vae/stage1/*/epoch_05000.pth

  
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

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


def _outward_normal(points, mesh_centroid, reference=None):
    """SVD-based outward surface normal of a point patch.

    SVD fixes the normal only UP TO SIGN (the last right-singular vector is
    equally valid negated), so the sign must come from somewhere else.

    reference: a direction already known to point outward for this opening --
    in practice compute_caps' per-opening centerline tangent, oriented along
    the branch running AWAY from the aneurysm. When given it decides the sign.

    Without one the fallback is the old heuristic: assume the vector from the
    whole mesh's centroid to the patch centroid points outward. That fails
    exactly where it matters -- a short stub cap near the mesh centre gives a
    tiny, noise-dominated reference vector; a curved vessel bending back inward,
    or a bifurcation whose mesh centroid sits between the branches, can put the
    patch on the "wrong" side. Those are the ones that come out pointing INTO
    the shape, so prefer a reference whenever one exists.
    """
    centroid = points.mean(axis=0)
    _, _, Vt = np.linalg.svd(points - centroid, full_matrices=False)
    normal = Vt[-1]
    if reference is not None:
        ref = np.asarray(reference, dtype=float)
        if np.isfinite(ref).all() and np.linalg.norm(ref) > 1e-8:
            if np.dot(normal, ref) < 0:
                normal = -normal
            return normal / np.linalg.norm(normal)
    if np.dot(normal, centroid - mesh_centroid) < 0:
        normal = -normal
    return normal / np.linalg.norm(normal)


def _faces_to_pv(faces):
    """[F, 3] int array -> PyVista's flat padded face format."""
    return np.hstack([np.full((len(faces), 1), 3), faces]).ravel()


def _safe_show(plotter):
    """plotter.show(), swallowing a known PyVista/VTK teardown crash.

    Every key callback in this file that ends a window (_set in
    review_auto_labels, _confirm in brush_one_branch) calls plotter.close()
    from INSIDE the VTK interactor loop that show() starts. On PyVista 0.46 /
    VTK 9.2, show() resumes right after that loop and unconditionally does
    `self.render_window.IsCurrent()` with no None-guard -- but our own
    plotter.close() already destroyed render_window, so that line raises
    `AttributeError: 'NoneType' object has no attribute 'IsCurrent'`. By the
    time it fires, our callback has already written whatever state (choice /
    confirmed / picked_cells) the caller needs, so this is safe to ignore --
    re-raised if the message doesn't match, so a genuinely different
    AttributeError still surfaces.
    """
    try:
        plotter.show()
    except AttributeError as exc:
        if "IsCurrent" not in str(exc):
            raise


class _Reject:
    """Returned by brush_one_branch when the human judged the SHAPE itself unfit
    to label, as opposed to closing the window (which means 'not now')."""
    def __repr__(self):
        return "REJECT"


REJECT = _Reject()


def brush_region(verts, faces, label, add_reference, allow_reject=False):
    """Brush ONE arbitrary region (used for the dome). Thin wrapper over
    brush_one_branch so the picking/highlight/confirm machinery has a single
    implementation -- the branch version is left untouched."""
    return brush_one_branch(verts, faces, 0, 1, add_reference, label=label,
                            allow_reject=allow_reject)


def brush_one_branch(verts, faces, branch_idx, n_branches, add_reference, label=None,
                     allow_reject=False):
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
    state = {"highlight": None, "confirmed": False, "rejected": False}

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
            tqdm.write(f"[label_morpho] no original-cell-id array in picked.cell_data; "
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

    def _reject():
        # Distinct from closing the window. Closing means "not now" and leaves the
        # case to be offered again; this means "this shape is not worth labelling"
        # and is recorded so it is never offered again.
        state["rejected"] = True
        plotter.close()

    # "branch {branch_idx}" (0-indexed) matches the array index this patch lands in
    # (in_patch[branch_idx], endpoints[branch_idx], ...) and the label/color shown for
    # this same branch in review_auto_labels' overview window and the reference geometry
    # below -- "(k/n)" alongside it is just the human-friendly "Nth of N windows" count.
    header = label if label else f"Branch {branch_idx} ({branch_idx + 1}/{n_branches})"
    tqdm.write(f"\n[label_morpho] {header}: brush window opened")
    tqdm.write("    LEFT-CLICK-DRAG  = select a box of faces (repeat to add more)")
    tqdm.write("    'r'              = toggle ROTATE vs. SELECT mode")
    tqdm.write("    'z'              = clear the current selection")
    tqdm.write("    'c'              = confirm selection and close this window")
    if allow_reject:
        tqdm.write("    'x'              = REJECT this shape as unrealistic (recorded; "
                   "never offered again)")
    tqdm.write("    close window     = skip for now (case not saved; rerun later to retry)")
    plotter.add_text(
        header + ": LEFT-CLICK-DRAG a box over the "
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
    if allow_reject:
        plotter.add_key_event("x", _reject)
    _safe_show(plotter)

    if state["rejected"]:
        return REJECT
    if not state["confirmed"] or not picked_cells:
        return None
    return sorted(picked_cells)


def review_auto_labels(verts, faces, caps, cap_id, dome, branch_points, n_open, case, info,
                       endpoints=None, tangents=None, branch_mask=None, allow_reject=False):
    """Show the automatic CAP and DOME regions and ask what to keep.

    Returns a set of regions to brush by hand -- {} to accept everything,
    {"caps"}, {"dome"}, or {"caps","dome"} -- or None to skip the case.

    Caps and dome are answered separately because they fail independently: the
    plane cut can be wrong while the dome is fine, and vice versa, and
    re-brushing a region that was already correct is wasted effort.

    endpoints/tangents/branch_mask: the ground-truth per-branch values already
    stored in the checkpoint (preprocess_endcaps.py's ray-crossing derivation
    -- see module docstring), when this case has one. Drawn as the same
    yellow-arrow-at-a-sphere convention as _add_arrow_and_point/_real_reference
    use during per-branch brushing, so the tangent is visible at the overview
    stage too, not only once you're already inside a branch's brush window.
    """
    import pyvista as pv

    mesh = pv.PolyData(verts, _faces_to_pv(faces))
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color="whitesmoke", opacity=0.45, show_edges=False)

    if dome is not None and dome.any():
        plotter.add_points(verts[dome], color="crimson", point_size=7,
                           render_points_as_spheres=True, pickable=False)
    cap_colors = ["red", "green", "blue"]
    label_pts, label_txt = [], []
    for b in range(n_open):
        m = caps & (cap_id == b)
        anchor = None
        if m.any():
            plotter.add_points(verts[m], color=cap_colors[b % 3], point_size=9,
                               render_points_as_spheres=True, pickable=False)
            anchor = verts[m].mean(axis=0)
        if b < len(branch_points) and branch_points[b] is not None and len(branch_points[b]) > 1:
            cl = np.asarray(branch_points[b], dtype=float)[:60]
            plotter.add_lines(np.repeat(cl, 2, axis=0)[1:-1],
                              color=BRANCH_COLORS[b % len(BRANCH_COLORS)], width=3)
            anchor = cl[0]   # branch's own centerline start -- most reliable anchor when present
        if (endpoints is not None and tangents is not None
                and b < len(endpoints) and (branch_mask is None or bool(branch_mask[b]))):
            _add_arrow_and_point(plotter, endpoints[b], tangents[b], color_point=cap_colors[b % 3])
            anchor = np.asarray(endpoints[b], dtype=float)   # ground-truth endpoint wins if present
        else:
            # No stored ground truth for this branch (the common case for this corpus --
            # see _has_auto_ground_truth) -- fall back to compute_caps' own plane normal.
            fb_origin, fb_tangent = _fallback_auto_tangent(b, caps, cap_id, info, verts)
            if fb_origin is not None:
                _add_arrow_and_point(plotter, fb_origin, fb_tangent, color_point=cap_colors[b % 3])
                anchor = fb_origin
        if anchor is not None:
            label_pts.append(anchor)
            label_txt.append(f"branch {b}")
    # Numeric labels so branch identity doesn't rely on distinguishing similar colors --
    # this is the SAME "branch {b}" index each per-branch brush window will show in its
    # own title (see brush_one_branch's header), so it carries over across windows.
    if label_pts:
        plotter.add_point_labels(np.asarray(label_pts), label_txt, font_size=16,
                                 text_color="black", shape_color="white", shape_opacity=0.7,
                                 always_visible=True, pickable=False)

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
    has_ground_truth_tangents = endpoints is not None and tangents is not None
    tqdm.write(f"\n[label_morpho] {case}: reviewing AUTOMATIC labels")
    tangent_note = ("   (yellow arrows = ground-truth tangent per branch)" if has_ground_truth_tangents
                    else "   (yellow arrows = compute_caps' plane-normal tangent -- "
                         "no ground truth for this case)" if info else "")
    tqdm.write(f"    caps: {summary}" + ("" if summary else "  (none)") + tangent_note)
    tqdm.write(f"    {dome_txt}")
    tqdm.write("    'a' = ACCEPT both caps and dome (keep automatic, no brushing)")
    tqdm.write("    'c' = rebrush CAPS only (keep the automatic dome; you'll then be asked which "
          "branch(es) -- blank/'all' for every branch, or e.g. '1' or '0,2' for just those)")
    tqdm.write("    'd' = rebrush DOME only (keep the automatic caps)")
    tqdm.write("    'b' = rebrush BOTH caps and dome (same per-branch prompt for caps)")
    tqdm.write("    'q' or close window = SKIP this case (not saved; retry later)")
    if allow_reject:
        tqdm.write("    'x' = REJECT this shape as unrealistic (recorded; never offered again)")
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
    if allow_reject:
        # Distinct from 'q'. 'q' means "not now" and re-offers the case later;
        # this means "this SHAPE is not worth labelling" and is recorded so it
        # never comes back, and so the rejection rate per generator stays
        # measurable.
        plotter.add_key_event("x", _set(REJECT))
    _safe_show(plotter)
    return state["choice"]


def _ask_branches(case, n_open):
    """Which cap(s) actually need a hand. Often only one is wrong, and
    re-brushing a correct one is wasted effort."""
    if n_open <= 1:
        return set(range(n_open))
    resp = input(f"[label_morpho] {case}: rebrush which branch(es)? comma-separated "
                 f"indices 0-{n_open - 1} (e.g. '1' or '0,2'), or blank/'all' for all "
                 f"{n_open}: ").strip().lower()
    if not resp or resp in ("all", "a"):
        return set(range(n_open))
    try:
        sel = {int(t) for t in resp.replace(" ", "").split(",") if t}
    except ValueError:
        sel = set()
    sel = {b for b in sel if 0 <= b < n_open}
    if not sel:
        tqdm.write(f"[label_morpho] {case}: couldn't parse {resp!r} -- rebrushing all.")
        return set(range(n_open))
    tqdm.write(f"[label_morpho] {case}: rebrushing branch(es) {sorted(sel)}; "
               f"keeping the sensor's prediction for the rest.")
    return sel


def _sensor_proposal(sensor, phi, atype, n_verts, n_open, device):
    """Run the morphology sensor and shape its output like an automatic label.

    Returns (caps, cap_id, dome, endpoints, tangents, info) in exactly the form
    review_auto_labels already consumes for real cases, so the triage window is
    the same one, with the sensor standing in for the geometric pipeline.
    """
    import torch
    from models.morphoformer import region_from_probs
    with torch.no_grad():
        ep, tg, loc, tok, extra = sensor(
            torch.as_tensor(phi, dtype=torch.float32, device=device)[None],
            torch.as_tensor([atype], dtype=torch.long, device=device))
    nv = min(n_verts, int(tok[0].sum()))
    caps = np.zeros(n_verts, dtype=bool)
    cap_id = np.full(n_verts, -1, dtype=int)
    info = []
    for b in range(n_open):
        idx = region_from_probs(loc[0, b, :nv].detach().cpu().numpy())
        caps[idx] = True
        cap_id[idx] = b
        info.append({"opening": b, "n_cap_verts": int(idx.size), "used_crossing": True})
    dome = np.zeros(n_verts, dtype=bool)
    if "dome_logits" in extra:
        d = torch.sigmoid(extra["dome_logits"][0, :nv]).detach().cpu().numpy() > 0.5
        dome[:nv] = d
    return (caps, cap_id, dome,
            ep[0].detach().cpu().numpy(), tg[0].detach().cpu().numpy(), info)


def _add_arrow_and_point(plotter, origin, direction, color_point, color_arrow="yellow", scale=2.0):
    import pyvista as pv
    # pickable=False: reference-only geometry must never be selectable, or a brush drag
    # near it turns _on_pick's `picked` into a MultiBlock instead of the mesh's own
    # UnstructuredGrid (see _on_pick).
    plotter.add_mesh(pv.Sphere(radius=0.15, center=origin), color=color_point, pickable=False)
    plotter.add_mesh(pv.Arrow(start=origin, direction=direction, scale=scale), color=color_arrow, pickable=False)


def _fallback_auto_tangent(b, caps, cap_id, info, verts):
    """Tangent to show when this case has no preprocess_endcaps.py ground truth
    (see _has_auto_ground_truth) -- which is the common case for most of this
    corpus, not the rare one the module docstring assumed. Falls back to the
    SAME per-opening normal compute_caps itself used to orient this opening's
    cutting plane (branch_tangent(br), stashed in info[b]["normal"]) -- less
    reliable than real ray-crossing ground truth, but still an automatic value,
    worth showing rather than no arrow at all. Origin is this opening's own
    cap-vertex centroid when compute_caps found any, else the plane point it
    cut against. Returns (origin, tangent), or (None, None) if info doesn't
    cover this branch (e.g. compute_caps failed entirely -- caps is None)."""
    if caps is None or info is None or b >= len(info):
        return None, None
    o = info[b]
    tangent = np.asarray(o["normal"], dtype=float)
    m = caps & (cap_id == b)
    origin = verts[m].mean(axis=0) if m.any() else np.asarray(o["plane_point"], dtype=float)
    return origin, tangent


def _real_reference(plotter, branch_idx, centerline_pts, auto_endpoint, auto_tangent, auto_patch_pts):
    """Real-case reference: dashed-style real centerline (per-branch color),
    automatic endpoint (same color, sphere) + tangent (yellow arrow, matching
    every sanity panel's convention in this project), automatic patch (faint dots)."""
    import pyvista as pv
    c = BRANCH_COLORS[branch_idx % len(BRANCH_COLORS)]
    anchor = None
    if centerline_pts is not None and len(centerline_pts) > 1:
        line = pv.lines_from_points(centerline_pts[:60])   # cap length shown, matches preprocess_endcaps.py
        plotter.add_mesh(line, color=c, line_width=3, pickable=False)
        anchor = np.asarray(centerline_pts[0], dtype=float)
    if auto_endpoint is not None and auto_tangent is not None:
        _add_arrow_and_point(plotter, auto_endpoint, auto_tangent, color_point=c)
        anchor = np.asarray(auto_endpoint, dtype=float)   # ground-truth endpoint wins if present
    if auto_patch_pts is not None and len(auto_patch_pts) > 0:
        plotter.add_points(auto_patch_pts, color=c, point_size=6, opacity=0.35, pickable=False)
    if anchor is not None:
        # Same "branch {branch_idx}" index/anchor as review_auto_labels' overview window,
        # so identity carries over even though only THIS branch's geometry is shown here.
        plotter.add_point_labels([anchor], [f"branch {branch_idx}"], font_size=16,
                                 text_color="black", shape_color="white", shape_opacity=0.7,
                                 always_visible=True, pickable=False)


def _synthetic_reference(plotter, branch_idx, opening_pts):
    """Synthetic-case reference: faint markers at the FIXED canonical opening
    indices for this branch slot — approximate, for branch identity only."""
    c = BRANCH_COLORS[branch_idx % len(BRANCH_COLORS)]
    if opening_pts is not None and len(opening_pts) > 0:
        plotter.add_points(opening_pts, color=c, point_size=10, opacity=0.5, pickable=False)
        plotter.add_point_labels([np.asarray(opening_pts, dtype=float).mean(axis=0)],
                                 [f"branch {branch_idx}"], font_size=16, text_color="black",
                                 shape_color="white", shape_opacity=0.7,
                                 always_visible=True, pickable=False)
        plotter.add_text(
            f"faint {c} dots (branch {branch_idx}) = approximate canonical opening, NOT precise",
            position="lower_left", font_size=9,
        )


def _derive_label(verts, faces, picked_face_ids, mesh_centroid, reference=None):
    """reference: this opening's outward centerline tangent (compute_caps'
    info[b]["normal"]) when the case has one. A hand-brushed patch is where the
    mesh-centroid sign heuristic is least reliable -- the brush covers a
    slightly different region than the automatic cut -- so orient against the
    reference at derivation rather than leaving it to _repair_tangents to
    notice afterwards."""
    patch_faces = faces[picked_face_ids]
    patch_vertex_idx = np.unique(patch_faces.ravel())
    patch_pts = verts[patch_vertex_idx]
    endpoint = patch_pts.mean(axis=0).astype(np.float32)
    tangent = _outward_normal(patch_pts, mesh_centroid,
                              reference=reference).astype(np.float32)
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


TANGENT_WARN_DEG = 60.0


def rerender_sanity(rec, endcaps_rec, sanity_dir, plane_frac=0.5,
                    warn_deg=TANGENT_WARN_DEG, warn_only=False):
    """Redraw one case's sanity image from its EXISTING label, touching nothing.

    Read-only on purpose: the point is to re-inspect labels (in particular the
    SVD patch normal stored as manual_tangents) without risking a rewrite of
    work that was done by hand.

    Also cross-checks that stored tangent against the automatic one. The
    automatic value is NOT kept in the record -- only manual_tangents is -- but
    it is recomputable from the case's branch_points via compute_caps, whose
    per-opening "normal" is the centerline tangent that oriented the cut plane.
    Two independent estimates of the same direction: a large angle between them
    means at least one is wrong, and the render is the only way to tell which.

    Returns (path, warnings) where warnings is a list of human-readable strings.
    """
    case = rec["case"]
    atype = int(rec["aneurysm_type"])
    n_open = TYPE_N_OPEN.get(atype, MAX_BRANCHES)
    verts, faces = reconstruct(rec)

    in_patch = np.asarray(endcaps_rec["manual_in_patch"], dtype=bool)
    dome = np.asarray(endcaps_rec.get("manual_dome",
                                      np.zeros(len(verts), dtype=bool)), dtype=bool)
    tang = np.asarray(endcaps_rec.get("manual_tangents"), dtype=float)
    eps = np.asarray(endcaps_rec.get("manual_endpoints"), dtype=float)
    bmask = np.asarray(endcaps_rec.get("manual_branch_mask",
                                       np.ones(MAX_BRANCHES, bool)), dtype=bool)

    def _render():
        return save_label_sanity(verts, faces, in_patch, dome, case,
                                 Path(sanity_dir) / f"{case}.png", n_open,
                                 endpoints=eps, tangents=tang, branch_mask=bmask)

    auto_normal = {}
    try:
        _c, _cid, info, _l, _d = compute_caps(rec, verts, faces, plane_frac=plane_frac)
        for o in info:
            auto_normal[int(o["opening"])] = np.asarray(o["normal"], dtype=float)
    except Exception as exc:
        # No centerline for this case, so there is no branch direction to check
        # against -- fall back to asking the MESH which side is out.
        ws = []
        for b in range(min(n_open, len(tang))):
            if not bool(bmask[b]) or np.linalg.norm(tang[b]) < 1e-8:
                continue
            sign, used = _outward_by_ray(verts, faces, eps[b], tang[b])
            if sign < 0:
                ws.append(f"{case}: branch {b} POINTS INWARD by mesh ray test "
                          f"at {used}mm (no centerline) -- rebrush or re-save to correct")
            elif sign == 0:
                ws.append(f"{case}: branch {b} direction UNDECIDED -- no probe distance "
                          f"in {list(RAY_STEPS)} could separate inside from outside "
                          f"(no centerline) -- INSPECT")
        if not ws:
            ws = [f"{case}: no centerline; mesh ray test says all tangents point outward "
                  f"({type(exc).__name__})"]
        return _render(), ws

    warns = []
    for b in range(min(n_open, len(tang))):
        if not bool(bmask[b]):
            continue
        t = tang[b]
        if not np.isfinite(t).all() or np.linalg.norm(t) < 1e-8:
            warns.append(f"{case}: branch {b} has no usable stored tangent")
            continue
        a = auto_normal.get(b)
        if a is None or np.linalg.norm(a) < 1e-8:
            warns.append(f"{case}: branch {b} has no automatic tangent to check against")
            continue
        # SIGNED, deliberately. Both estimates are independently oriented
        # OUTWARD, so a ~180 deg disagreement means one points INTO the mesh --
        # precisely the failure worth catching. abs() would hide a flip as 0 deg.
        cos = float(np.dot(t / np.linalg.norm(t), a / np.linalg.norm(a)))
        ang = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
        if ang > warn_deg:
            kind = "POINTS INWARD (flipped)" if ang > 120 else "disagrees"
            warns.append(f"{case}: branch {b} stored tangent {kind}: {ang:.0f} deg from the "
                         f"automatic one (> {warn_deg:.0f}) -- INSPECT the sanity image")
    # warn_only: a clean case needs no picture -- rendering all 523 to look at
    # a handful buries the ones that matter (and costs ~20 min of matplotlib).
    if warn_only and not warns:
        return None, warns
    return _render(), warns


# Probe distances, tried largest first. A step wider than the local wall
# overshoots: the inward probe passes clean through and lands OUTSIDE too, so
# both sides read "outside" and the test cannot decide. Backing off to a
# shorter step keeps the inward probe within the lumen. Starting small instead
# would be worse -- a probe shorter than the mesh's own surface roughness sits
# ambiguously on the boundary.
RAY_STEPS = (0.5, 0.25, 0.05)


def _outward_by_ray(verts, faces, origin, direction, steps=RAY_STEPS, mesh=None):
    """Which way is out, decided by the MESH rather than by any heuristic.

    Steps off the cap centroid along the candidate normal, both ways, and asks
    the closed mesh which probe is inside. The side that lands OUTSIDE is out.

    Tries each distance in `steps` in order and takes the first that gives a
    clear answer, because a single fixed distance fails at both ends: too wide
    and the inward probe punches through a thin wall (both probes outside),
    too narrow and both sit ambiguously on the surface.

    Returns (verdict, step_used): verdict is +1 already outward, -1 flip it, or
    0 when no step could decide -- which callers must treat as "unknown",
    never as "fine".
    """
    import trimesh

    d = np.asarray(direction, dtype=float)
    n = np.linalg.norm(d)
    if n < 1e-8:
        return 0, None
    d = d / n
    o = np.asarray(origin, dtype=float)
    if mesh is None:
        mesh = trimesh.Trimesh(vertices=np.asarray(verts, dtype=float),
                               faces=np.asarray(faces), process=False)
    for step in steps:
        inside = mesh.contains(np.vstack([o + step * d, o - step * d]))
        if bool(inside[0]) != bool(inside[1]):
            return (-1 if inside[0] else 1), step
    return 0, None


def _canonical_tangents(tangents, branch_mask, info, case, verts=None, faces=None,
                        endpoints=None, warn_deg=TANGENT_WARN_DEG):
    """Put the GROUND-TRUTH tangent in place, per branch, before saving.

    Priority, deliberately:

      1. the CENTERLINE tangent (compute_caps' info[b]["normal"]) whenever the
         branch has one. This is ground truth -- it comes from the vessel's own
         skeleton -- whereas the SVD normal of a hand-brushed cap only
         approximates it and inherits whatever the brush happened to cover. So
         the centerline REPLACES the SVD value rather than merely correcting
         its sign.

      2. the patch SVD normal, kept only where no centerline exists, with its
         direction settled by the mesh ray test.

    A large disagreement between the two is still reported, but it is no longer
    a correctness problem: the stored value is the centerline's either way.
    """
    """Flip any tangent pointing INTO the mesh, in place, before saving.

    Belt and braces: the accept and brush paths already orient against
    compute_caps' normal, but a record can also reach save carrying a tangent
    loaded from an OLDER file written before that fix. Running the check here
    means every case that passes through the interactive flow comes out with
    outward tangents, whichever path produced them.
    """
    replaced, flipped, by_ray, diverged = [], [], [], []
    have = {int(o["opening"]): np.asarray(o["normal"], dtype=float)
            for o in (info or [])}
    for b in range(len(tangents)):
        if not bool(branch_mask[b]):
            continue
        t = np.asarray(tangents[b], dtype=float)
        a = have.get(b)
        if a is not None and np.linalg.norm(a) > 1e-8:
            a = a / np.linalg.norm(a)
            if np.linalg.norm(t) > 1e-8:
                cos = float(np.dot(t / np.linalg.norm(t), a))
                ang = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
                if ang > warn_deg:
                    diverged.append(f"{b}:{ang:.0f}deg")
                if ang > 1e-3:
                    replaced.append(b)
            else:
                replaced.append(b)
            tangents[b] = a                      # centerline wins outright
        elif (verts is not None and endpoints is not None and b < len(endpoints)
              and np.linalg.norm(t) > 1e-8):
            # no centerline for this opening -- keep the SVD normal, but let
            # the mesh settle which way it points
            sign, used = _outward_by_ray(verts, faces, endpoints[b], t)
            if sign < 0:
                tangents[b] = -t
                flipped.append(b); by_ray.append(f"{b}@{used}mm")
            elif sign > 0:
                by_ray.append(f"{b}@{used}mm")
    if replaced:
        note = f"  (differed by {', '.join(diverged)})" if diverged else ""
        tqdm.write(f"[label_morpho] {case}: tangent(s) set from centerline "
                   f"on branch {replaced}{note}")
    if flipped:
        tqdm.write(f"[label_morpho] {case}: no centerline -- SVD tangent flipped outward "
                   f"on branch {flipped} (ray test {by_ray})")
    return replaced + flipped


def _has_auto_ground_truth(endcaps_rec):
    """True iff endcaps_rec holds preprocess_endcaps.py's automatic endpoints/
    tangents/branch_mask/in_patch -- NOT just "a record file exists for this
    case". A case that hit the no-ground-truth fallback (see label_real_case's
    docstring) gets a record saved too, but one holding only manual_endpoints/
    manual_tangents/manual_branch_mask/manual_in_patch -- endcaps_rec is not
    None there either, so callers must check for the actual keys, not None-ness,
    or a rerun crashes indexing endcaps_rec["endpoints"] that was never written."""
    return endcaps_rec is not None and "endpoints" in endcaps_rec and "tangents" in endcaps_rec


def _save_real_record(rec, endcaps_path, endcaps_rec, manual_in_patch,
                      manual_endpoints, manual_tangents, manual_branch_mask,
                      source, manual_dome=None, dome_source="none", reviewed=False):
    """Write manual_in_patch (and, only when there is no automatic record to
    defer to, the brush/auto-derived endpoint geometry) back into the case.

    reviewed: True whenever a HUMAN made this call interactively -- pressing
    'a'/'c'/'d'/'b' in review_auto_labels, or brushing by hand -- as opposed to
    a headless --auto-accept pass nobody looked at. source=="auto" alone can't
    tell those apart (both leave the automatic value unchanged), so main()'s
    resume check needs this separate flag to skip a case a human already
    judged "good as automatic" without re-showing it every run.
    """
    out = dict(endcaps_rec) if endcaps_rec is not None else {
        "case": rec["case"], "aneurysm_type": int(rec["aneurysm_type"]),
        "phi": np.asarray(rec["ghd"]["phi"], dtype=np.float32),
    }
    out["manual_in_patch"] = manual_in_patch
    out["in_patch_source"] = source          # "auto" or "manual" -- provenance
    out["reviewed"] = bool(reviewed)
    if manual_dome is not None:
        out["manual_dome"] = np.asarray(manual_dome, dtype=bool)
        out["dome_source"] = dome_source
    if not _has_auto_ground_truth(endcaps_rec):
        out["manual_endpoints"] = manual_endpoints
        out["manual_tangents"] = manual_tangents
        out["manual_branch_mask"] = manual_branch_mask
    np.save(endcaps_path, out, allow_pickle=True)
    tqdm.write(f"[label_morpho] updated {endcaps_path} ({source} labels)")
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
    has_ground_truth = _has_auto_ground_truth(endcaps_rec)

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
    caps = cap_id = info = None         # stay None if use_auto is False or compute_caps fails
    caps_to_brush = set(range(n_open))  # branch indices actually needing a brush window; auto may narrow it

    if use_auto:
        try:
            caps, cap_id, info, _, _ = compute_caps(rec, verts, faces, plane_frac=plane_frac)
        except Exception as exc:
            tqdm.write(f"[label_morpho] {case}: automatic caps unavailable ({exc}) -- brushing")
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
                tqdm.write(f"[label_morpho] {case}: automatic dome unavailable ({exc})")

        if caps is not None:
            if auto_accept:
                # Non-interactive: take the automatic proposals as-is. For bulk
                # labelling of a corpus whose proposals have already been spot-
                # checked; a case missing either proposal is SKIPPED rather than
                # saved half-labelled, since there is no human here to brush it.
                if dome_auto is None:
                    tqdm.write(f"[label_morpho] {case}: no dome proposal -- skipped "
                          f"(--auto-accept cannot brush)")
                    return False
                choice = frozenset()
            else:
                choice = review_auto_labels(
                    verts, faces, caps, cap_id, dome_auto, branch_points, n_open, case, info,
                    endpoints=endcaps_rec["endpoints"] if has_ground_truth else None,
                    tangents=endcaps_rec["tangents"] if has_ground_truth else None,
                    branch_mask=endcaps_rec["branch_mask"] if has_ground_truth else None,
                )
            if choice is None:
                tqdm.write(f"[label_morpho] {case}: skipped at review -- not saved.")
                return False
            rebrush = set(choice)
            caps_to_brush = set(range(n_open)) if "caps" in rebrush else set()
            if caps_to_brush and n_open > 1:
                # Ask which branch(es) actually need a hand -- often only one cap
                # was wrong, and re-brushing an already-correct one is wasted effort.
                resp = input(
                    f"[label_morpho] {case}: rebrush which branch(es)? comma-separated "
                    f"indices 0-{n_open - 1} (e.g. '1' or '0,2'), or blank/'all' for all "
                    f"{n_open}: ").strip().lower()
                if resp and resp not in ("all", "a"):
                    try:
                        sel = {int(tok) for tok in resp.replace(" ", "").split(",") if tok}
                    except ValueError:
                        sel = set()
                    sel = {b for b in sel if 0 <= b < n_open}
                    if sel:
                        caps_to_brush = sel
                        tqdm.write(f"[label_morpho] {case}: rebrushing branch(es) "
                              f"{sorted(caps_to_brush)}; keeping automatic for the rest.")
                    else:
                        tqdm.write(f"[label_morpho] {case}: couldn't parse {resp!r} as branch "
                              f"indices -- rebrushing all {n_open} branches.")
            # Accept the automatic in_patch/endpoint/tangent for every branch NOT
            # selected for rebrushing (covers both "caps" never in rebrush at all,
            # and "caps" in rebrush but the human narrowed it to a subset above).
            for b in range(n_open):
                if b in caps_to_brush:
                    continue
                m = caps & (cap_id == b)
                manual_in_patch[b, np.flatnonzero(m)] = True
                if m.any():
                    pts = verts[m]
                    manual_endpoints[b] = pts.mean(axis=0)
                    # info[b]["normal"] is compute_caps' centerline tangent for
                    # this opening -- already outward, so it fixes the SVD sign.
                    _ref = (np.asarray(info[b]["normal"], dtype=float)
                            if info is not None and b < len(info) else None)
                    manual_tangents[b] = _outward_normal(pts, mesh_centroid,
                                                         reference=_ref)
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
        tqdm.write(f"[label_morpho] {case}: needs manual work for {sorted(rebrush)} "
              f"-- skipped (--auto-accept cannot brush)")
        return False

    # Announce the brushing order up front -- DOME always opens first, THEN caps branches
    # 0, 1, (2) in that order (see the "dome" block immediately below, then the "for b in
    # sorted(caps_to_brush)" loop after it). Worth saying explicitly since each window's own
    # header only names ITSELF ("<case> DOME" / "Branch b (k/n)"), not where it sits in the
    # sequence -- most visible for a case with no automatic record at all (caps is None),
    # where EVERYTHING needs brushing and there's no review_auto_labels summary beforehand.
    if "dome" in rebrush or caps_to_brush:
        order = (["DOME"] if "dome" in rebrush else []) + (
            [f"caps branch(es) {sorted(caps_to_brush)}"] if caps_to_brush else [])
        tqdm.write(f"[label_morpho] {case}: brushing needed, in this order -- "
                   f"{' then '.join(order)}")

    if "dome" in rebrush:
        picked = brush_region(
            verts, faces, f"{case} DOME",
            add_reference=lambda p, d=(dome_auto if use_auto else None):
                (p.add_points(verts[d], color="crimson", point_size=7,
                              render_points_as_spheres=True, pickable=False)
                 if d is not None and d.any() else None),
        )
        if picked is None:
            tqdm.write(f"[label_morpho] {case}: dome skipped (no confirmed selection) -- not saved.")
            return False
        manual_dome[np.unique(faces[picked].ravel())] = True
        dome_source = "manual"

    if not caps_to_brush:
        _canonical_tangents(manual_tangents, manual_branch_mask, info, case,
                         verts=verts, faces=faces, endpoints=manual_endpoints)
        ok = _save_real_record(rec, endcaps_path, endcaps_rec, manual_in_patch,
                               manual_endpoints, manual_tangents,
                               manual_branch_mask, "auto", manual_dome, dome_source,
                               reviewed=not auto_accept)
        if ok and sanity_dir is not None:
            save_label_sanity(verts, faces, manual_in_patch, manual_dome, case,
                              Path(sanity_dir) / f"{case}.png", n_open,
                              endpoints=manual_endpoints, tangents=manual_tangents,
                              branch_mask=manual_branch_mask)
        return ok

    for b in sorted(caps_to_brush):
        cl_pts = branch_points[b] if b < len(branch_points) else None
        auto_ep = endcaps_rec["endpoints"][b] if has_ground_truth else None
        auto_tg = endcaps_rec["tangents"][b] if has_ground_truth else None
        auto_patch_pts = (verts[endcaps_rec["in_patch"][b]]
                          if has_ground_truth and endcaps_rec["in_patch"][b].any() else None)
        if auto_ep is None:
            # No stored ground truth (the common case for this corpus) -- same
            # compute_caps plane-normal fallback used in review_auto_labels.
            auto_ep, auto_tg = _fallback_auto_tangent(b, caps, cap_id, info, verts)

        picked = brush_one_branch(
            verts, faces, b, n_open,
            add_reference=lambda p, b=b, cl=cl_pts, ae=auto_ep, at=auto_tg, ap=auto_patch_pts:
                _real_reference(p, b, cl, ae, at, ap),
        )
        if picked is None:
            tqdm.write(f"[label_morpho] {case} branch {b}: skipped (no confirmed selection) — case not saved.")
            return False

        _ref = (np.asarray(info[b]["normal"], dtype=float)
                if info is not None and b < len(info) else None)
        patch_idx, endpoint, tangent = _derive_label(verts, faces, picked, mesh_centroid,
                                                     reference=_ref)
        manual_in_patch[b, patch_idx] = True
        manual_endpoints[b] = endpoint
        manual_tangents[b] = tangent
        manual_branch_mask[b] = True

    # "manual" when every branch got brushed, "mixed" when caps_to_brush was narrowed to a
    # subset (see the branch-selection prompt above) and the rest kept the automatic value.
    in_patch_source = "manual" if caps_to_brush == set(range(n_open)) else "mixed"
    _canonical_tangents(manual_tangents, manual_branch_mask, info, case,
                     verts=verts, faces=faces, endpoints=manual_endpoints)
    ok = _save_real_record(rec, endcaps_path, endcaps_rec, manual_in_patch,
                           manual_endpoints, manual_tangents, manual_branch_mask,
                           in_patch_source, manual_dome, dome_source, reviewed=not auto_accept)
    if ok and sanity_dir is not None:
        save_label_sanity(verts, faces, manual_in_patch, manual_dome, case,
                          Path(sanity_dir) / f"{case}.png", n_open,
                          endpoints=manual_endpoints, tangents=manual_tangents,
                          branch_mask=manual_branch_mask)
    return ok


def label_synthetic_case(case_id, atype, phi, multi_recon, synthetic_dir,
                         scale=1.0, provenance=None, sensor=None, device=None):
    """No automatic record exists for a fresh synthetic sample, so this writes
    a brand-new record directly into synthetic_dir (runtime_dataset/AneuG_morpho_synthetic/,
    kept separate from the real cases' AneuG_morpho/ — see module docstring),
    using the plain (non-"manual_"-prefixed) schema — there's nothing automatic
    to preserve alongside it."""
    # Synthetic samples come from the GHD VAE decoder: phi only, with no s_can
    # and no fitted Stage-2 pose, so they are NOT rebuilt the way a fitted case is.
    #
    # This MUST use multi_recon, i.e. exactly what MorphoFormer sees via
    # to_pyg_batch: (canonical + eigvec @ phi) * norm_canonical, verified
    # identical to 0.000e+00. It used to call preprocess_ImperialNHS.
    # _reconstruct_ghd_numpy, whose norm_canonical carries a legacy
    # * 1.10 * 2.50. Both add the same canonical template, so the discrepancy
    # is not a global scale that a brush would be blind to -- it multiplies the
    # DEFORMATION by exactly 2.75, i.e. the human would have been brushing a
    # caricature of the shape and every saved endpoint would have landed in a
    # frame the model never sees.
    verts = multi_recon._reconstruct_verts_np(phi, atype)
    faces = multi_recon.get(atype).canonical_Meshes.faces_packed().detach().cpu().numpy()
    mesh_centroid = verts.mean(axis=0)
    n_open = TYPE_N_OPEN.get(atype, MAX_BRANCHES)
    opening_indices = multi_recon._load_openings(atype)   # list of index tensors, len == n_open for this type

    endpoints = np.zeros((MAX_BRANCHES, 3), dtype=np.float32)
    tangents = np.zeros((MAX_BRANCHES, 3), dtype=np.float32)
    branch_mask = np.zeros(MAX_BRANCHES, dtype=bool)
    in_patch = np.zeros((MAX_BRANCHES, len(verts)), dtype=bool)

    # Dome first, then caps -- same order the real path announces, so the
    # human's muscle memory carries over between the two modes. There is no
    # automatic proposal to accept here: a synthetic shape has no dome_sac.ply
    # and no label.nrrd, so the dome is always brushed from scratch.
    if provenance:
        ckpt = Path(provenance["ghd_vae"]).parent.name if provenance.get("ghd_vae") else "?"
        amp = provenance.get("z_amp", "?")
        amp_txt = f"{amp:.2f}" if isinstance(amp, (int, float)) else str(amp)
        tqdm.write(f"[label_morpho] {case_id}: checkpoint {ckpt}, z_amp {amp_txt}")
    def _tombstone():
        # Tombstone, not silence. Writing the rejection keeps the case out of every
        # later run, and the rejection RATE per generator is itself the validity
        # statistic for that generator -- discarding these would throw that away.
        np.save(synthetic_dir / f"{case_id}.npy", {
            "case": case_id, "aneurysm_type": atype,
            "phi": np.asarray(phi, dtype=np.float32),
            "is_synthetic": True, "rejected": True,
            "provenance": provenance or {},
        }, allow_pickle=True)
        tqdm.write(f"[label_morpho] {case_id}: REJECTED as unrealistic -- recorded, "
                   f"will not be offered again.")

    dome = None
    caps_to_brush = set(range(n_open))
    src = {"caps": "manual", "dome": "manual"}

    if sensor is not None:
        # THE POINT OF THE SENSOR PASS. This labelling exists to fix the sensor
        # where it is wrong on generated shapes, so showing its own prediction
        # first turns the job from "brush every shape" into "brush the ones it
        # got wrong". Cases accepted as-is carry little training signal, but
        # they cost nothing to keep and they guard against forgetting when
        # mixed with the corrections.
        p_caps, p_cap_id, p_dome, p_ep, p_tg, info = _sensor_proposal(
            sensor, phi, atype, len(verts), n_open, device)
        choice = review_auto_labels(
            verts, faces, p_caps, p_cap_id, p_dome, [None] * n_open, n_open,
            case_id, info, endpoints=p_ep, tangents=p_tg,
            branch_mask=np.array([True] * n_open + [False] * (MAX_BRANCHES - n_open)),
            allow_reject=True)
        if choice is REJECT:
            _tombstone(); return False
        if choice is None:
            tqdm.write(f"[label_morpho] {case_id}: skipped -- not saved, will be re-offered.")
            return False
        if "dome" not in choice:
            dome = p_dome
            src["dome"] = "sensor"
        caps_to_brush = _ask_branches(case_id, n_open) if "caps" in choice else set()
        for b in range(n_open):
            if b not in caps_to_brush:
                endpoints[b] = p_ep[b]
                tangents[b] = p_tg[b]
                branch_mask[b] = True
                in_patch[b, p_cap_id == b] = True
        if not caps_to_brush:
            src["caps"] = "sensor"
        elif len(caps_to_brush) < n_open:
            src["caps"] = "mixed"

    if dome is None:
        tqdm.write(f"[label_morpho] {case_id}: brushing DOME"
                   + (f" then caps {sorted(caps_to_brush)}" if caps_to_brush else ""))
        picked = brush_region(verts, faces, f"{case_id} DOME", add_reference=lambda p: None,
                              allow_reject=sensor is None)
        if picked is REJECT:
            _tombstone(); return False
        if picked is None:
            tqdm.write(f"[label_morpho] {case_id}: dome skipped (no confirmed selection) -- case not saved.")
            return False
        dome = np.zeros(len(verts), dtype=bool)
        dome[np.unique(faces[picked].ravel())] = True

    for b in sorted(caps_to_brush):
        idx = opening_indices[b].detach().cpu().numpy() if b < len(opening_indices) else None
        opening_pts = verts[idx] if idx is not None else None

        picked = brush_one_branch(
            verts, faces, b, n_open,
            add_reference=lambda p, b=b, op=opening_pts: _synthetic_reference(p, b, op),
        )
        if picked is None:
            tqdm.write(f"[label_morpho] {case_id} branch {b}: skipped (no confirmed selection) — case not saved.")
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
        # "manual_dome" is the key MorphoDataset._dome_for looks for first, so a
        # synthetic record is picked up by exactly the same path as a real one.
        "manual_dome": dome, "dome_source": src["dome"],
        "tangent_source": src["caps"], "in_patch_source": src["caps"],
        "reviewed": True,
        # Which parts the human actually corrected. The corrected ones are where
        # the training signal is; "sensor" everywhere means this shape taught the
        # model nothing new, which is worth being able to count later.
        "label_source": dict(src),
        # Stage-2 pose. A synthetic shape is decoded straight into the canonical
        # frame, so its rotation and translation are identity by construction.
        # log_scale is NOT identity: the stage-1 VAE has withscale=True and
        # generates exp(log_scale) alongside phi, so the real fitted quantity has
        # a generated counterpart and is recorded rather than invented.
        "ghd": {"w_rot": np.zeros(3, dtype=np.float32),
                "log_scale": np.array([np.log(max(float(scale), 1e-6))], dtype=np.float32),
                "t_vec": np.zeros(3, dtype=np.float32)},
        "is_synthetic": True,
        # Enough to regenerate this exact shape, and to filter the corpus later
        # by generator config or by how extreme the sample was.
        "provenance": provenance or {},
    }, allow_pickle=True)
    tqdm.write(f"[label_morpho] saved {case_id}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["real", "synthetic"], default="real")
    parser.add_argument("--real-dir", default=str(DEFAULT_REAL_DIR))
    parser.add_argument("--endcaps-dir", default=str(DEFAULT_ENDCAPS_DIR),
                         help="Real mode only: read automatic labels from here and write manual labels back "
                              "into the same per-case record — no separate output folder.")
    parser.add_argument("--synthetic-dir", default=str(DEFAULT_SYNTHETIC_DIR),
                         help="Synthetic mode only: where labeled synthetic cases are written. Kept separate "
                              "from --endcaps-dir since synthetic cases have no automatic counterpart there.")
    parser.add_argument("--case", action="append", dest="cases", help="Real mode: label only these cases.")
    parser.add_argument("--n-synthetic", type=int, default=20)
    parser.add_argument("--ghd-vae", nargs="+", default=[str(GHD_VAE_CKPT)],
                        help="One or more stage-1 GHD VAE checkpoints. Several generators are "
                             "pooled so the fine-tuned sensor learns synthetic shapes in general "
                             "rather than one generator's artefacts.")
    parser.add_argument("--z-amp", nargs="+", type=float, default=None,
                        help="Scale(s) on z ~ N(0,1), one fixed value per group (every checkpoint "
                             "crossed with every amplitude) instead of the default random range. "
                             "Passing this explicitly disables --z-amp-range, even the default.")
    parser.add_argument("--z-amp-range", nargs=2, type=float, default=[1.0, 5.0], metavar=("LOW", "HIGH"),
                        help="Draw a fresh amplitude per generated shape, uniformly from [LOW, HIGH] "
                             "-- so within one run some draws land near the prior (typical shapes, "
                             "amp near 1.0) and some land out near HIGH (extreme ones), rather than "
                             "every shape in a group sharing one fixed amplitude. Default 1.0-5.0 "
                             "-- this is the default mode; pass --z-amp instead for the old fixed-"
                             "list behaviour.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu",
                        help="Synthetic mode only: runs the frozen GHD VAE and, if given, the sensor.")
    parser.add_argument("--pool-dir", default=None,
                        help="Label a PRE-GENERATED pool (scripts/generate/gen_synthetic_pool.py) "
                             "instead of sampling here. Preferred: the set being labelled is then a "
                             "fixed artefact rather than a function of which day the labeller ran.")
    parser.add_argument("--sensor", default=None,
                        help="morphology_sensor checkpoint. When given, its prediction is shown "
                             "FIRST as a proposal and you only brush what it got wrong -- which is "
                             "the whole reason for this labelling round.")
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
    parser.add_argument("--fix-tangents", action="store_true",
                        help="Flip stored tangents that point INTO the mesh, judged against "
                             "compute_caps' centerline tangent. Rewrites .npy files -- run "
                             "--rerender afterwards to see the result.")
    parser.add_argument("--rerender", action="store_true",
                        help="Regenerate sanity images from the EXISTING labels and report "
                             "tangents that disagree with the automatic estimate. Writes only "
                             "PNGs -- never touches a .npy, so hand-labelled work is safe.")
    parser.add_argument("--warn-only", action="store_true",
                        help="With --rerender, draw ONLY the cases that trigger a warning. "
                             "Every case is still checked; the clean ones just get no PNG.")
    parser.add_argument("--sanity-dir", default=None,
                        help="Where --rerender writes (default: <endcaps-dir>/sanity).")
    parser.add_argument("--no-sanity", action="store_true",
                        help="Skip the per-case label render. They are the only way to "
                             "inspect a bulk --auto-accept pass after the fact.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after this many cases (smoke tests).")
    parser.set_defaults(auto=True)
    args = parser.parse_args()

    if args.fix_tangents:
        # Repair pass for records written before _outward_normal took an
        # outward reference: flip any stored tangent that points INTO the mesh,
        # judged against compute_caps' centerline tangent for that opening.
        # Writes .npy, so it is a separate opt-in mode -- never folded into
        # --rerender, which must stay read-only over hand-labelled work.
        endcaps_dir, real_dir = Path(args.endcaps_dir), Path(args.real_dir)
        names = args.cases or [q.stem for q in sorted(endcaps_dir.glob("*.npy"))]
        if args.limit:
            names = names[:args.limit]
        n_fix = n_case = n_skip = 0
        for case in tqdm(names, desc="fix-tangents"):
            rp, ep = real_dir / f"{case}.npy", endcaps_dir / f"{case}.npy"
            if not (rp.exists() and ep.exists()):
                continue
            rec = np.load(rp, allow_pickle=True).item()
            erec = np.load(ep, allow_pickle=True).item()
            verts, faces = reconstruct(rec)
            try:
                _c, _cid, info, _l, _d = compute_caps(rec, verts, faces,
                                                      plane_frac=args.plane_frac)
            except Exception:
                # No centerline: the SVD normal stays, but the mesh ray test can
                # still settle its direction, so this is NOT a skip any more.
                info = None
            tang = np.asarray(erec["manual_tangents"], dtype=float)
            bmask = np.asarray(erec.get("manual_branch_mask",
                                        np.ones(MAX_BRANCHES, bool)), dtype=bool)
            eps = np.asarray(erec.get("manual_endpoints"), dtype=float)
            before = tang.copy()
            _canonical_tangents(tang, bmask, info, case, verts=verts, faces=faces,
                                endpoints=eps)
            flipped = [b for b in range(len(tang))
                       if not np.allclose(before[b], tang[b], atol=1e-6)]
            if flipped:
                erec["manual_tangents"] = tang.astype(np.float32)
                erec["tangent_source"] = "centerline" if info else "svd+ray"
                erec["tangent_fixed"] = sorted(int(b) for b in flipped)
                np.save(ep, erec, allow_pickle=True)
                n_fix += len(flipped); n_case += 1
                tqdm.write(f"[FIXED] {case}: flipped branch(es) {flipped}")
        print(f"\nUpdated {n_fix} tangent(s) across {n_case} case(s) "
              f"(centerline where available, mesh ray test otherwise).")
        return

    if args.rerender:
        # Read-only pass: redraw every existing label's sanity image and report
        # tangents that disagree with the automatic estimate. No .npy is opened
        # for writing anywhere in this branch.
        endcaps_dir = Path(args.endcaps_dir)
        real_dir = Path(args.real_dir)
        sanity_dir = Path(args.sanity_dir) if args.sanity_dir else endcaps_dir / "sanity"
        names = (args.cases if args.cases else
                 [p.stem for p in sorted(endcaps_dir.glob("*.npy"))])
        if args.limit:
            names = names[:args.limit]
        all_warns, n_ok, n_fail, n_drawn = [], 0, 0, 0
        for case in tqdm(names, desc="rerender"):
            rp, ep = real_dir / f"{case}.npy", endcaps_dir / f"{case}.npy"
            if not (rp.exists() and ep.exists()):
                tqdm.write(f"[label_morpho] {case}: missing record -- skipped")
                continue
            try:
                _path, warns = rerender_sanity(
                    np.load(rp, allow_pickle=True).item(),
                    np.load(ep, allow_pickle=True).item(),
                    sanity_dir, plane_frac=args.plane_frac,
                    warn_only=args.warn_only)
                n_ok += 1
                if _path is not None:
                    n_drawn += 1
                for w in warns:
                    tqdm.write(f"[WARN] {w}")
                all_warns += warns
            except Exception as exc:
                n_fail += 1
                tqdm.write(f"[FAIL] {case}: {type(exc).__name__}: {exc}")
        print(f"\nChecked {n_ok} case(s); rendered {n_drawn} into {sanity_dir}; "
              f"{n_fail} failed.")
        print(f"{len(all_warns)} warning(s) -- these are the cases to eyeball:")
        for w in all_warns:
            print(f"  {w}")
        return

    if args.mode == "real":
        endcaps_dir = Path(args.endcaps_dir)
        endcaps_dir.mkdir(parents=True, exist_ok=True)
        real_dir = Path(args.real_dir)
        paths = ([real_dir / f"{c}.npy" for c in args.cases] if args.cases
                 else sorted(real_dir.glob("*.npy")))
        n_done = n_skip = n_missing = 0
        # dynamic_ncols: keep the bar itself on one line. Every diagnostic in this file goes
        # through tqdm.write() rather than print() -- tqdm.write() clears the bar, writes the
        # line, then redraws the bar, so the two never leave stale/duplicate-looking bar lines
        # behind each other the way interleaved plain print()s + a live '\r'-redrawn bar do.
        pbar = tqdm(paths, desc="Labeling", unit="case", dynamic_ncols=True)
        for path in pbar:
            if args.limit is not None and n_done >= args.limit:
                break
            pbar.set_postfix(done=n_done, skipped=n_skip, missing=n_missing, case=path.stem)
            if not path.exists():
                tqdm.write(f"[SKIP] {path} not found"); n_missing += 1; continue
            case = path.stem
            endcaps_path = endcaps_dir / f"{case}.npy"
            endcaps_rec = np.load(endcaps_path, allow_pickle=True).item() if endcaps_path.exists() else None
            # Resume skips a case only when a HUMAN has already judged it -- either the
            # explicit "reviewed" flag (set whenever label_real_case ran interactively,
            # even if the human's verdict was 'a' ACCEPT and the value stayed "auto"),
            # or, for records saved before that flag existed, the old signal: both
            # fields hold a "manual" (non-"auto") source, which only a full hand-brush
            # could have produced. A headless --auto-accept record (reviewed=False,
            # source="auto") is exactly what the interactive pass exists to review, so
            # it must not be skipped here.
            if endcaps_rec is not None and "manual_in_patch" in endcaps_rec and (
                    endcaps_rec.get("reviewed")
                    or (endcaps_rec.get("in_patch_source") != "auto"
                        and endcaps_rec.get("dome_source") != "auto")):
                n_skip += 1
                continue   # already reviewed -- resumable
            rec = np.load(path, allow_pickle=True).item()
            if label_real_case(rec, endcaps_path, endcaps_rec,
                               use_auto=args.auto, plane_frac=args.plane_frac,
                               assembled_root=args.assembled_root,
                               auto_accept=args.auto_accept,
                               sanity_dir=(None if args.no_sanity
                                           else Path(args.endcaps_dir) / "sanity")):
                n_done += 1
        pbar.set_postfix(done=n_done, skipped=n_skip, missing=n_missing)
        pbar.close()
        tqdm.write(f"Done. {n_done} labeled, {n_skip} already reviewed, {n_missing} missing. "
              f"Manual labels written into {endcaps_dir}")

    else:
        import torch
        from utils.generate_synthetic import load_ghd_vae
        from models.multi_canonical_ghd_reconstruct import MultiCanonicalGHDReconstruct

        synthetic_dir = Path(args.synthetic_dir)
        synthetic_dir.mkdir(parents=True, exist_ok=True)

        device = torch.device(args.device)
        multi_recon = MultiCanonicalGHDReconstruct(CANONICAL_ROOT, device=device)

        sensor = None
        if args.sensor:
            from models.morphoformer import MorphoFormer
            sck = torch.load(args.sensor, map_location=device, weights_only=False)
            if not sck["args"].get("predict_dome"):
                parser.error("--sensor has no dome head: that is an uncapper, not a "
                             "morphology_sensor, and it cannot propose a dome.")
            sensor = MorphoFormer(multi_recon, **sck["args"]).to(device)
            sensor.load_state_dict(sck["model"]); sensor.eval()
            tqdm.write(f"[label_morpho] sensor: {Path(args.sensor).parent.name} "
                       f"epoch {sck.get('epoch')} -- its prediction is the proposal; "
                       f"brush only what it gets wrong")

        if args.pool_dir:
            pool = sorted(Path(args.pool_dir).glob("*.npy"))
            if not pool:
                parser.error(f"--pool-dir is empty: {args.pool_dir}")
            plan = []
            for q in pool:
                r = np.load(q, allow_pickle=True).item()
                plan.append({"case_id": r["case"], "atype": int(r["aneurysm_type"]),
                             "phi": np.asarray(r["phi"], dtype=np.float32),
                             "scale": float(r.get("scale", 1.0)),
                             "provenance": r.get("provenance", {})})
            # Interleave generators: labelling gets abandoned partway, so any
            # PREFIX of the session has to stay a balanced sample.
            np.random.default_rng(args.seed).shuffle(plan)
            tqdm.write(f"[label_morpho] labelling pool of {len(plan)} from {args.pool_dir}")

        if not args.pool_dir:
            # Default mode: one group per checkpoint, each shape within it draws its own random
            # amplitude below (so a single run mixes typical and extreme shapes). Passing --z-amp
            # explicitly opts back into the old fixed-list behaviour (every checkpoint crossed
            # with every amplitude, every shape in a group sharing that one value) and disables
            # --z-amp-range entirely, even its default.
            random_amp = args.z_amp is None
            groups = ([(Path(c), None) for c in args.ghd_vae] if random_amp else
                      [(Path(c), float(a)) for c in args.ghd_vae for a in args.z_amp])
            for c, _ in groups:
                if not c.exists():
                    parser.error(f"--ghd-vae not found: {c}")
            base, extra = divmod(args.n_synthetic, len(groups))
            plan = []
            for gi, (ckpt_path, z_amp) in enumerate(groups):
                n_g = base + (1 if gi < extra else 0)
                if n_g == 0:
                    continue
                ghd_vae, ghd_mean, ghd_std, ghd_input_dim = load_ghd_vae(ckpt_path, device)
                for prm in ghd_vae.parameters():
                    prm.requires_grad_(False)
                tag = f"{ckpt_path.parent.name}_arand" if random_amp else f"{ckpt_path.parent.name}_a{z_amp:g}"

                g = torch.Generator(device="cpu").manual_seed(args.seed + 1000 * gi)
                types = torch.randint(0, 2, (n_g,), generator=g)   # type 2 merged into 1 -- never sampled
                # One amplitude per shape when random (uniform over [LOW, HIGH]), else every
                # shape in this group shares the fixed --z-amp value -- same as before.
                z_amp_g = (torch.empty(n_g).uniform_(*args.z_amp_range, generator=g) if random_amp
                           else torch.full((n_g,), z_amp))
                z_ghd = torch.randn(n_g, ghd_vae.latent_dim, generator=g) * z_amp_g.unsqueeze(-1)
                with_scale = ghd_mean.numel() > ghd_input_dim
                with torch.no_grad():
                    out = ghd_vae.decode(z_ghd.to(device), types.to(device))
                    # The stage-1 VAE is trained withscale=True, so decode returns the
                    # generated exp(log_scale) alongside phi. It used to be discarded
                    # here; it is the synthetic counterpart of the real fitted Stage-2
                    # scale, so it is kept and written into the record.
                    ghd_n, scale_n = out if isinstance(out, tuple) else (out, None)
                    phi_all = (ghd_n * ghd_std[:, :ghd_input_dim]
                               + ghd_mean[:, :ghd_input_dim]).reshape(n_g, -1, 3).cpu().numpy()
                    scale_all = ((scale_n * ghd_std[:, ghd_input_dim:]
                                  + ghd_mean[:, ghd_input_dim:]).squeeze(1).cpu().numpy()
                                 if with_scale and scale_n is not None
                                 else np.ones(n_g, dtype=np.float32))
                amp_txt = (f"z_amp in [{z_amp_g.min():.2f}, {z_amp_g.max():.2f}]" if random_amp
                          else f"z_amp {z_amp:g}")
                tqdm.write(f"[label_morpho] {tag}: {n_g} shape(s), {amp_txt}, generated scale mean "
                           f"{float(np.mean(scale_all)):.4f}")
                for i in range(n_g):
                    amp_i = float(z_amp_g[i])
                    case_id = (f"synthetic_{tag}_a{amp_i:.2f}_seed{args.seed}_{i:04d}" if random_amp
                              else f"synthetic_{tag}_seed{args.seed}_{i:04d}")
                    plan.append({
                        "case_id": case_id,
                        "atype": int(types[i]), "phi": phi_all[i], "scale": float(scale_all[i]),
                        "provenance": {"ghd_vae": str(ckpt_path), "z_amp": amp_i,
                                       "seed": int(args.seed), "index": int(i),
                                       "z": z_ghd[i].cpu().numpy().astype(np.float32)},
                    })

            # Interleave the groups. Labelling is slow and gets abandoned partway, so
            # the order has to make any PREFIX of the session a balanced sample across
            # generators and amplitudes -- brushing group by group would mean stopping
            # early leaves the corpus skewed to whichever config happened to be first.
            np.random.default_rng(args.seed).shuffle(plan)

        n_done = n_skip = 0
        pbar = tqdm(plan, desc="Labeling", unit="case", dynamic_ncols=True)
        for item in pbar:
            case_id = item["case_id"]
            pbar.set_postfix(done=n_done, skipped=n_skip, case=case_id)
            if (synthetic_dir / f"{case_id}.npy").exists():
                n_skip += 1
                continue   # resumable
            if label_synthetic_case(case_id, item["atype"], item["phi"], multi_recon,
                                    synthetic_dir, scale=item["scale"],
                                    provenance=item["provenance"],
                                    sensor=sensor, device=device):
                n_done += 1
        pbar.set_postfix(done=n_done, skipped=n_skip)
        pbar.close()
        tqdm.write(f"Done. {n_done} labeled, {n_skip} already existed. Manual labels written into {synthetic_dir}")

        # How often a generator's shapes get rejected IS that generator's validity
        # rate, so report it rather than leaving it buried in the tombstones.
        tally = {}
        for q in sorted(synthetic_dir.glob("*.npy")):
            r = np.load(q, allow_pickle=True).item()
            if not r.get("is_synthetic"):
                continue
            prov = r.get("provenance") or {}
            amp_val = prov.get("z_amp", "?")
            # Bucket to the nearest 0.5 so a continuous --z-amp-range draw still groups with
            # its neighbours here -- otherwise every sample gets its own row (kept+rej == 1
            # each), and the whole point of this table (reject rate BY amplitude) is lost.
            # A --z-amp value already a multiple of 0.5 (the common case) is unaffected.
            amp_key = round(amp_val * 2) / 2 if isinstance(amp_val, (int, float)) else amp_val
            key = (Path(prov.get("ghd_vae", "?")).parent.name, amp_key)
            kept, rej = tally.get(key, (0, 0))
            tally[key] = (kept + (0 if r.get("rejected") else 1), rej + (1 if r.get("rejected") else 0))
        if tally:
            tqdm.write(f"\n{'generator':<34}{'z_amp':>7}{'kept':>7}{'rejected':>10}{'reject rate':>13}")
            for (cfg, amp), (kept, rej) in sorted(tally.items()):
                tot = kept + rej
                tqdm.write(f"{cfg:<34}{str(amp):>7}{kept:>7}{rej:>10}"
                           f"{(rej / tot if tot else 0):>12.1%}")


if __name__ == "__main__":
    main()
