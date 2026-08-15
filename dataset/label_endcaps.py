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
the SAME per-case checkpoint under runtime/dataset/processed_endcaps/<case>.npy
that dataset/preprocess_endcaps.py already produces, as a new manual_in_patch
field ONLY — endpoints/tangents/branch_mask stay the automatic, ray-crossing-
derived values (see above), never overwritten. dataset/endcap_dataset.py's
EndcapDataset prefers manual_in_patch over automatic in_patch when present,
while always using the automatic endpoints/tangents/branch_mask. Resumable: a
case whose record already has manual_in_patch is skipped. (There's only ONE
kind of label source worth a folder split here — automatic vs. manual in_patch
is the same case, just refined — so it doesn't get one. The rare case where a
real case has NO automatic record at all falls back to a full manual_endpoints/
manual_tangents/manual_branch_mask/manual_in_patch set instead, since there's
nothing to defer to; EndcapDataset falls back to those per-field too.)

Synthetic cases: a genuinely different population (no real centerline, no
automatic record, no case in dataset/processed/ to attach to), so they get
their OWN folder, runtime/dataset/processed_endcaps_synthetic/, using the
plain (non-"manual_"-prefixed) schema fields directly — there's nothing
automatic to preserve alongside a synthetic case. Keeping synthetic cases out
of processed_endcaps/ also means a real preprocess_endcaps.py rerun (e.g.
after tuning PATCH_THRESHOLD_MM) can't accidentally interact with them.
EndcapDataset accepts a list of roots, so combining real + synthetic for
training is one line: EndcapDataset([processed_endcaps, processed_endcaps_synthetic]).

conda activate new   (needs pyvista with a real display / working GL context —
will not work over a plain SSH session without X forwarding or a virtual
framebuffer)
python dataset/label_endcaps.py --mode real
python dataset/label_endcaps.py --mode synthetic --n-synthetic 20 --seed 0
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.preprocess_ImperialNHS import _reconstruct_ghd_numpy

MAX_BRANCHES = 3
TYPE_N_OPEN = {0: 3, 1: 2, 2: 2}   # candidate branch count per type — type 2 merged into type 1 elsewhere
                                    # in this pipeline, but the mesh geometry (Sidewall template) is identical
                                    # either way, so labeling only ever needs 2 or 3 branches per case.

DEFAULT_REAL_DIR       = ROOT / "runtime" / "dataset" / "processed"                     # source of real centerlines
DEFAULT_ENDCAPS_DIR    = ROOT / "runtime" / "dataset" / "processed_endcaps"             # read automatic labels from AND write manual labels into (real mode)
DEFAULT_SYNTHETIC_DIR  = ROOT / "runtime" / "dataset" / "processed_endcaps_synthetic"    # write-only, synthetic mode
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


def brush_one_branch(verts, faces, branch_idx, n_branches, add_reference):
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
            print(f"[label_endcaps] no original-cell-id array in picked.cell_data; "
                  f"available arrays: {picked.array_names}. Edit _on_pick in "
                  f"dataset/label_endcaps.py to use the right key for your PyVista version.")
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
        f"Branch {branch_idx + 1}/{n_branches}: LEFT-CLICK-DRAG a box over the cap "
        f"region (repeat to add more), 'z' clears, 'c' confirms",
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


def label_real_case(rec, endcaps_path, endcaps_rec):
    """rec: runtime/dataset/processed/<case>.npy checkpoint (dict). endcaps_rec:
    runtime/dataset/processed_endcaps/<case>.npy record (dict) to update IN PLACE
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
    as the only available source; EndcapDataset falls back to those per-field
    when the plain endpoints/tangents/branch_mask are absent.)"""
    case = rec["case"]
    atype = int(rec["aneurysm_type"])
    n_open = TYPE_N_OPEN.get(atype, MAX_BRANCHES)
    has_ground_truth = endcaps_rec is not None

    verts, faces = _reconstruct_ghd_numpy(rec, denormalize_shape=True)
    mesh_centroid = verts.mean(axis=0)
    branch_points = rec["clipped_centerline"]["branch_points"][:n_open]

    manual_in_patch = np.zeros((MAX_BRANCHES, len(verts)), dtype=bool)
    manual_endpoints = np.zeros((MAX_BRANCHES, 3), dtype=np.float32)
    manual_tangents = np.zeros((MAX_BRANCHES, 3), dtype=np.float32)
    manual_branch_mask = np.zeros(MAX_BRANCHES, dtype=bool)

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
            print(f"[label_endcaps] {case} branch {b}: skipped (no confirmed selection) — case not saved.")
            return False

        patch_idx, endpoint, tangent = _derive_label(verts, faces, picked, mesh_centroid)
        manual_in_patch[b, patch_idx] = True
        manual_endpoints[b] = endpoint
        manual_tangents[b] = tangent
        manual_branch_mask[b] = True

    endcaps_rec = dict(endcaps_rec) if has_ground_truth else {
        "case": case, "aneurysm_type": atype,
        "phi": np.asarray(rec["ghd"]["phi"], dtype=np.float32),
    }
    endcaps_rec["manual_in_patch"] = manual_in_patch
    if not has_ground_truth:
        endcaps_rec["manual_endpoints"] = manual_endpoints
        endcaps_rec["manual_tangents"] = manual_tangents
        endcaps_rec["manual_branch_mask"] = manual_branch_mask
    np.save(endcaps_path, endcaps_rec, allow_pickle=True)
    print(f"[label_endcaps] updated {endcaps_path} with manual labels")
    return True


def label_synthetic_case(case_id, atype, phi, multi_recon, synthetic_dir):
    """No automatic record exists for a fresh synthetic sample, so this writes
    a brand-new record directly into synthetic_dir (runtime/dataset/processed_endcaps_synthetic/,
    kept separate from the real cases' processed_endcaps/ — see module docstring),
    using the plain (non-"manual_"-prefixed) schema — there's nothing automatic
    to preserve alongside it."""
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
            print(f"[label_endcaps] {case_id} branch {b}: skipped (no confirmed selection) — case not saved.")
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
    print(f"[label_endcaps] saved {case_id}")
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
    args = parser.parse_args()

    if args.mode == "real":
        endcaps_dir = Path(args.endcaps_dir)
        endcaps_dir.mkdir(parents=True, exist_ok=True)
        real_dir = Path(args.real_dir)
        paths = ([real_dir / f"{c}.npy" for c in args.cases] if args.cases
                 else sorted(real_dir.glob("*.npy")))
        for path in paths:
            if not path.exists():
                print(f"[SKIP] {path} not found"); continue
            case = path.stem
            endcaps_path = endcaps_dir / f"{case}.npy"
            endcaps_rec = np.load(endcaps_path, allow_pickle=True).item() if endcaps_path.exists() else None
            if endcaps_rec is not None and "manual_in_patch" in endcaps_rec:
                continue   # already manually labeled — resumable
            rec = np.load(path, allow_pickle=True).item()
            label_real_case(rec, endcaps_path, endcaps_rec)
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
