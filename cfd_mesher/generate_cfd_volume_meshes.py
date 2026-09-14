"""Tetrahedralize CFD-ready surfaces into boundary-tagged volume meshes for a solver.

Takes the surface stage's deliverable (a closed-except-openings .obj) and its fusion
bookkeeping npz, converts to .vtp, brings the surface's roughness into the physiological
range if it is outside it, optionally extends the inlet/outlet openings for a
developed-flow length, tetrahedralizes at an edge length adaptive to the shape's volume
and area, relabels wall/inlet/outlet boundary ids to a fixed policy, and exports the
bookkeeping a Fluent run needs (inlet node coordinates, and for two-outlet cases a
flow-split ratio).

Mesh parameters match AneuSeg/cfd_meshing/get_mesh_aneux_tune_elements_v3.py exactly,
including the vmtkmeshgenerator boundary-layer settings in vmtk_backend.py.

This is the volume stage; it reads what generate_cfd_surface_meshes.py writes.

    python -m cfd_mesher.generate_cfd_volume_meshes                 # batch, no prompts
    python -m cfd_mesher.generate_cfd_volume_meshes \
        --surface_root /path/to/surfaces --save_root /path/to/volumes --overwrite

Per case the following are written into <save_root>/<case>/ (save_root defaults to
--surface_root itself, i.e. in place next to the surface stage's output):

    (every source file)         copied from <surface_root>/<case>/ unless --no-copy_case_files
    ghd_surface.vtp             the surface .obj converted to .vtp
    regularization.json         the roughness assessment and what, if anything, was done
    ghd_surface_regularized.vtp only if the surface was rougher than the reference
    ghd_surface_extended.vtp    only if --extend_inlet/--extend_outlets is set
    mesh.vtu                    the tagged volume mesh
    mesh.msh                    the same mesh, converted to Fluent format
    inlet_centroids.csv         every inlet node's coordinates, scaled by --inlet_scale
    flowsplit_ratio.txt         two-outlet cases only: normalized per-outlet flow split

Requires the vmtk CLI on PATH, not just the vmtk python package (vmtkmeshgenerator is
shelled out to) -- true in the vmtk_autogen conda environment the surface stage already
documents using.

    conda activate vmtk_autogen
    python -m cfd_mesher.generate_cfd_volume_meshes --help

ROUGHNESS. Before meshing, each surface is measured against mesh_regularizer's reference
model without being modified. A surface within the typical range, or smoother than it,
is meshed as it is. Only a rougher one is regularized, and the result is used only if it
keeps every opening in place, since relabelling depends on the caps. Run from the repo
root so mesh_regularizer imports; --no-regularize skips the step entirely.

GENERATED COHORTS. scripts/generate/gen_merged_cohort.py writes the
ghd_forward_fusion_info.npz this stage reads, with openings in branch order (inlet
first). If the surfaces are remeshed by other software first, keep that npz in each
case folder; the processed surface is read as geomagic_processed.obj by default.
"""

import os
import json
import shutil
import argparse

from .config import SURFACE_SAVE_ROOT, VOLUME_SAVE_ROOT

# The Geomagic-processed surface, as in get_mesh_aneux_tune_elements_v3.py
# (obj_prefix = "geomagic_processed"). The surface stage writes
# ghd_smoothed_reconstruction.obj, which goes through Geomagic before reaching here;
# pass --surface_filename ghd_smoothed_reconstruction.obj to mesh it directly.
SURFACE_FILENAME = "geomagic_processed.obj"
FUSION_INFO_FILENAME = "ghd_forward_fusion_info.npz"

# Calibrated for an average of 3 M volume elements on the AneuGv2 generated cohort.
#
# The previous values, tuned on AneuX real data (get_mesh_aneux_tune_elements_v3.py),
# were base_edge 0.13, ref_volume 320, ref_area 320, vol_exponent 0.30,
# area_exponent 0.50. On this cohort they already averaged 3.0 M, but unevenly:
# bifurcated shapes rarely exceed either reference, so they were never coarsened and
# came in at 2.5 M, while the larger sidewall shapes were coarsened too little and
# averaged 3.5 M. Measured on five cases they spanned 1.72 to 3.88 M.
#
# Element count fits N = 27.4 V/e^3 + 36.1 A/e^2 (V mm^3, A mm^2, e mm) to within 5%
# over seven meshes, one of them re-meshed at a second edge length. Solving against
# the volume and area of all 650 cases for a 3.0 M mean with the least spread gives
# the values below. vol_exponent 0.33 is the natural choice: edge growing as V^(1/3)
# holds V/e^3 constant. Predicted: mean 3.00 M, 5th-95th percentile 2.62-3.14 M,
# bifurcated 2.96 / sidewall 3.05. Re-meshed, the same five cases span 2.15 to
# 3.19 M, each within 5% of prediction.
#
# The mechanism only coarsens shapes above the references; it never refines below
# base_edge. A shape smaller than both therefore still falls short -- the smallest
# case here meshes to 2.15 M.
DEFAULT_MESH_PARAMS = dict(
    base_edge=0.1194,
    ref_volume=150.0,
    ref_area=300.0,
    vol_exponent=0.33,
    area_exponent=0.70,
    max_edge=1.0,
)


def _check_vmtk_available():
    """Fail fast if the vmtk CLI is missing, rather than failing identically on every
    case partway through inside a shelled-out vmtkmeshgenerator call."""
    if shutil.which("vmtkmeshgenerator") is None:
        raise SystemExit(
            "vmtkmeshgenerator is not on PATH; this pipeline shells out to the vmtk CLI.\n"
            "  Use the 'vmtk_autogen' conda environment (or wherever vmtk's CLI tools are "
            "installed), not just a Python env with the vmtk package importable.")


def _volume_mesh():
    """Imported lazily so --help works even where vmtk is not installed."""
    try:
        from . import volume_mesh as vm
        from . import vmtk_backend as backend
        return vm, backend
    except ImportError as exc:                                  # pragma: no cover
        raise SystemExit(
            "cfd_mesher needs vmtk (and vtk, pandas, pyvista) importable for the volume "
            "stage; the 'vmtk_autogen' conda environment has these.\n"
            "  underlying import error: %s" % exc)


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--surface_root", default=SURFACE_SAVE_ROOT,
                   help="per-case surface stage output (.obj + fusion npz)")
    p.add_argument("--save_root", default=VOLUME_SAVE_ROOT or None,
                   help="output root; default is --surface_root itself (in place)")
    p.add_argument("--surface_filename", default=SURFACE_FILENAME)
    p.add_argument("--npz_filename", default=FUSION_INFO_FILENAME)

    p.add_argument("--base_edge", type=float, default=DEFAULT_MESH_PARAMS["base_edge"])
    p.add_argument("--ref_volume", type=float, default=DEFAULT_MESH_PARAMS["ref_volume"])
    p.add_argument("--ref_area", type=float, default=DEFAULT_MESH_PARAMS["ref_area"])
    p.add_argument("--vol_exponent", type=float, default=DEFAULT_MESH_PARAMS["vol_exponent"])
    p.add_argument("--area_exponent", type=float, default=DEFAULT_MESH_PARAMS["area_exponent"])
    p.add_argument("--max_edge", type=float, default=DEFAULT_MESH_PARAMS["max_edge"])
    p.add_argument("--inflation", action=argparse.BooleanOptionalAction, default=True,
                   help="add boundary-layer sublayers at the wall")

    p.add_argument("--regularize", action=argparse.BooleanOptionalAction, default=True,
                   help="measure roughness against the physiological reference and smooth "
                        "only surfaces rougher than it")
    p.add_argument("--regularizer_tolerance", type=float, default=1.0,
                   help="SD above the reference mean a target scale may sit before the "
                        "surface counts as too rough")
    p.add_argument("--regularizer_remesher", default="auto",
                   choices=["auto", "pymeshlab", "vmtk"],
                   help="auto prefers pymeshlab and falls back to vmtk. The two gave the "
                        "same verdict to within 0.15 SD on generated shapes")
    p.add_argument("--max_cap_shift", type=float, default=0.5,
                   help="mm a cap centroid may move under regularization before the "
                        "regularized surface is rejected in favour of the original")

    p.add_argument("--extend_inlet", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--inlet_extension_length", type=float, default=3.0)
    p.add_argument("--extend_outlets", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--outlet_extension_length_one_outlet", type=float, default=3.0)
    p.add_argument("--outlet_extension_length_two_outlets", type=float, default=3.0)

    p.add_argument("--flow_split_exponent", type=float, default=2.45)
    p.add_argument("--inlet_scale", type=float, default=0.001,
                   help="scale factor applied to inlet node coordinates in the CSV export")

    p.add_argument("--copy_case_files", action=argparse.BooleanOptionalAction, default=True,
                   help="copy every file in the source case folder into the output case "
                        "folder, so the volume disk holds complete cases")
    p.add_argument("--overwrite", action="store_true",
                   help="reprocess cases whose mesh.vtu and mesh.msh already both exist")
    p.add_argument("--cases", nargs="*", default=None,
                   help="only these case names (default: everything under --surface_root)")
    p.add_argument("--exclude", nargs="*", default=[], help="case names to skip")
    p.add_argument("--limit", type=int, default=None, help="stop after N cases")
    return p


def main():
    args = build_parser().parse_args()
    _check_vmtk_available()
    vm, backend = _volume_mesh()

    save_root = args.save_root or args.surface_root
    cases = args.cases or sorted(os.listdir(args.surface_root))
    cases = [c for c in cases if c not in set(args.exclude)]
    print("%d candidate case(s) | surface_root %s | save_root %s"
          % (len(cases), args.surface_root, save_root))

    done, skipped, failed, attempted = 0, 0, [], 0
    for case in cases:
        case_surface_dir = os.path.join(args.surface_root, case)
        obj_path = os.path.join(case_surface_dir, args.surface_filename)
        npz_path = os.path.join(case_surface_dir, args.npz_filename)
        save_dir = os.path.join(save_root, case)
        vtp_path = os.path.join(save_dir, "ghd_surface.vtp")
        vtu_path = os.path.join(save_dir, "mesh.vtu")
        msh_path = os.path.join(save_dir, "mesh.msh")

        if not os.path.isdir(case_surface_dir):
            continue
        if not os.path.exists(obj_path):
            print("[%s] no %s, skipping" % (case, args.surface_filename))
            skipped += 1
            continue
        if not os.path.exists(npz_path):
            print("[%s] no %s, skipping" % (case, args.npz_filename))
            skipped += 1
            continue
        if os.path.exists(vtu_path) and os.path.exists(msh_path) and not args.overwrite:
            skipped += 1
            continue

        attempted += 1
        os.makedirs(save_dir, exist_ok=True)
        print("[%s]" % case, flush=True)

        # Copy the whole source case folder alongside the volume mesh, so each output
        # case is self-contained on the volume disk: GHD coefficients, latent, opening
        # data, metadata and the processed surface travel with their mesh. Files
        # only, and never over an existing copy unless --overwrite.
        if args.copy_case_files and os.path.abspath(save_dir) != os.path.abspath(case_surface_dir):
            for name in sorted(os.listdir(case_surface_dir)):
                src = os.path.join(case_surface_dir, name)
                dst = os.path.join(save_dir, name)
                if os.path.isfile(src) and (args.overwrite or not os.path.exists(dst)):
                    shutil.copy2(src, dst)
        try:
            # -- obj -> vtp --------------------------------------------------------
            if not os.path.exists(vtp_path) or args.overwrite:
                # merged, not a plain pv.read: see volume_mesh.read_surface
                vm.read_surface(obj_path).save(vtp_path)

            surface_for_meshing = vtp_path

            # -- roughness: measure, and smooth only if rougher than the reference ---
            if args.regularize:
                reg_path = os.path.join(save_dir, "ghd_surface_regularized.vtp")
                used, report = vm.regularize_surface_if_needed(
                    surface_file=obj_path, output_file=reg_path,
                    report_path=os.path.join(save_dir, "regularization.json"),
                    tolerance=args.regularizer_tolerance,
                    remesher=args.regularizer_remesher,
                    max_cap_shift=args.max_cap_shift, overwrite=args.overwrite)
                a = report["assessment"]
                print("  roughness: %s (worst target %+.2f SD) -> %s"
                      % (a["verdict"], a["worst_target_excess_sd"], report["action"]))
                if report["action"] == "rejected":
                    print("  ! %s" % report["reason"])
                if report["action"] == "regularized":
                    surface_for_meshing = used

            # -- optional inlet/outlet flow extension -------------------------------
            if args.extend_inlet or args.extend_outlets:
                extended_path = os.path.join(save_dir, "ghd_surface_extended.vtp")
                if not os.path.exists(extended_path) or args.overwrite:
                    inlet_c, outlet_c = vm.load_opening_info_from_npz(npz_path)
                    vm.extend_inlet_and_outlets(
                        surface_file=surface_for_meshing, output_file=extended_path,
                        inlet_centroid=inlet_c, outlet_centroids=outlet_c,
                        extend_inlet=args.extend_inlet,
                        inlet_extension_length=args.inlet_extension_length,
                        extend_outlets=args.extend_outlets,
                        outlet_extension_length_one_outlet=args.outlet_extension_length_one_outlet,
                        outlet_extension_length_two_outlets=args.outlet_extension_length_two_outlets)
                surface_for_meshing = extended_path

            # -- volume meshing (adaptive edge from shape volume + area) ------------
            if not os.path.exists(vtu_path) or args.overwrite:
                vol, area = vm.get_surface_geometry_stats(surface_for_meshing)
                edge = vm.compute_adaptive_edge(
                    vol, area, base_edge=args.base_edge, ref_volume=args.ref_volume,
                    ref_area=args.ref_area, vol_exponent=args.vol_exponent,
                    area_exponent=args.area_exponent)
                print("  volume=%.2f area=%.2f edge=%.4f (base=%.4f)"
                      % (vol, area, edge, args.base_edge))
                backend.cfdmesher_custom(surface_for_meshing, vtu_path, edge, args.max_edge,
                                         "y" if args.inflation else "n")
                vm.report_mesh_stats(vtu_path)

            # -- relabel wall/inlet/outlet ids by npz cpcd endpoints ----------------
            vm.relabel_inlet_outlets_v2(mesh_file=vtu_path, npz_path=npz_path)

            # -- export inlet node coordinates for a Fluent UDF ---------------------
            inlet_csv = os.path.join(save_dir, "inlet_centroids.csv")
            if not os.path.exists(inlet_csv) or args.overwrite:
                vm.scan_inlet_nodes(vtu_path, csv_path=inlet_csv, scale_factor=args.inlet_scale)

            # -- export flow-split ratio for two-outlet cases -----------------------
            vm.write_flowsplit_ratio(vtu_path, exponent=args.flow_split_exponent,
                                     output_path=os.path.join(save_dir, "flowsplit_ratio.txt"))

            # -- write Fluent mesh ---------------------------------------------------
            if not os.path.exists(msh_path) or args.overwrite:
                backend.write_msh_single(vtu_path, msh_path)
        except Exception as exc:
            print("  failed: %r" % exc)
            failed.append({"case": case, "error": repr(exc)})
            if args.limit and attempted >= args.limit:
                break
            continue

        done += 1
        if args.limit and attempted >= args.limit:
            break

    print("meshed %d, skipped %d, failed %d" % (done, skipped, len(failed)))
    if failed:
        path = os.path.join(save_root, "volume_mesh_failures.json")
        os.makedirs(save_root, exist_ok=True)
        with open(path, "w") as f:
            json.dump(failed, f, indent=2)
        print("failures -> %s" % path)


if __name__ == "__main__":
    main()
