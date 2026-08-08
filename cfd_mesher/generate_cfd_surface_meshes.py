"""Generate CFD-ready surface meshes from an aneurysm complex and its centreline.

Takes a clipped aneurysm complex (either a GHD-fitted mesh or the segmentation-derived
clipped reconstruction), propagates the vessel branches outward along the clipped
centreline, fuses them into the complex, and writes a watertight-except-for-openings
surface suitable for volume meshing.

This is the surface stage; the resulting mesh is what the volume mesher consumes to
produce the CFD tetrahedral mesh.

    python -m cfd_mesher.generate_cfd_surface_meshes                 # automatic
    python -m cfd_mesher.generate_cfd_surface_meshes --no-automatic  # tune per case
    python -m cfd_mesher.generate_cfd_surface_meshes \
        --post_dir /path/to/PostTr --ghd_dir /path/to/fitting_results \
        --save_root /path/to/output --mesh_source ghd

Per case the following are written into <save_root>/<case>/:

    ghd_reconstructed.obj            the propagated complex before fusion
    ghd_merged_reconstruction.obj    complex fused with the propagated branches
    ghd_smoothed_reconstruction.obj  the deliverable, seams and openings smoothed
    ghd_forward_fusion_info.npz      fusion bookkeeping for downstream stages
    ghd_fusion_params.json           the parameters this case actually used
"""

import os
import json
import shutil
import argparse
import urllib.parse

from .config import POST_DIR, GHD_DIR, SURFACE_SAVE_ROOT

# Source of the complex surface. 'ghd' uses the GHD-fitted mesh, which is smooth and
# topologically regular; 'clipped' uses the raw clipped reconstruction straight from the
# segmentation, which follows the image more closely but carries its noise.
MESH_SOURCES = {
    "ghd": ("ghd_dir", "ghd_fitted_uncapped_world.obj"),
    "clipped": ("post_dir", "clipped_reconstruction.ply"),
}

COPY_ALONGSIDE = ["forward_fusion_info.npz", "branch_ranking.npy"]

DEFAULT_PARAMS = dict(
    flaw_opening_min_size=5,
    init_step=1,
    max_cl_length=None,
    ring_downsample_ratio=1,
    smooth_n_rings=5,
    smooth_n_iter=10,
)


def _check_obj_export():
    """Fail fast if pyvista cannot write .obj.

    vessel_reconstruct saves the merged and smoothed surfaces as .obj through
    pyvista.PolyData.save, which only gained .obj support in later releases: 0.42 raises
    "Invalid file extension", 0.47 works. Without this check every case fails identically
    partway through, which reads like a data problem rather than an environment one.
    """
    import tempfile
    import pyvista as pv
    try:
        pv.Sphere().save(os.path.join(tempfile.mkdtemp(), "_probe.obj"))
    except Exception:
        raise SystemExit(
            "pyvista %s cannot write .obj files, which this pipeline requires.\n"
            "  Use an environment with a newer pyvista (0.47 works; 0.42 does not).\n"
            "  During development this ran under the 'vmtk_autogen' environment."
            % pv.__version__)


def _vessel_reconstruct():
    """Imported lazily so --help still works before the module is vendored in here."""
    try:
        from . import vessel_reconstruct as vr
        return vr
    except ImportError as exc:                                  # pragma: no cover
        raise SystemExit(
            "cfd_mesher needs vessel_reconstruct.py (and its patching.py dependency) "
            "alongside this file; they have not been moved into this package yet.\n"
            "  original: AneuSeg/AneuG/Aneu_GHD/utils/{vessel_reconstruct,patching}.py\n"
            "  underlying import error: %s" % exc)


def ask_params(defaults):
    """Prompt for propagation parameters; a bare Enter keeps the current value."""
    print("=== Adjust parameters (Enter keeps default) ===")
    params = {}
    for key, default in defaults.items():
        raw = input("  %s [%s]: " % (key, default)).strip()
        if not raw:
            params[key] = default
        elif isinstance(default, bool):
            params[key] = raw.lower() in ("y", "yes", "true", "1")
        elif key == "max_cl_length":
            vals = [float(x) for x in raw.split() if x.lower() != "none"]
            params[key] = None if not vals else vals[0] if len(vals) == 1 else vals
        else:
            params[key] = type(default)(raw)
    return params


def copy_alongside(src_dir, dst_dir, filenames):
    """Carry the fusion bookkeeping next to the outputs, without clobbering."""
    for name in filenames:
        src, dst = os.path.join(src_dir, name), os.path.join(dst_dir, name)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--post_dir", default=POST_DIR,
                   help="per-case clipped centrelines and fusion info")
    p.add_argument("--ghd_dir", default=GHD_DIR,
                   help="per-case GHD fitting results; required for --mesh_source ghd")
    p.add_argument("--save_root", default=SURFACE_SAVE_ROOT,
                   help="output root; one subfolder per case, named by the case")
    p.add_argument("--mesh_source", default="ghd", choices=sorted(MESH_SOURCES),
                   help="which surface to propagate from (default: ghd)")
    p.add_argument("--automatic", action=argparse.BooleanOptionalAction, default=True,
                   help="run each case once with the default parameters and no prompts")
    p.add_argument("--extrude_outlets", action=argparse.BooleanOptionalAction, default=True,
                   help="extend the outlets straight, giving CFD a developed-flow length")
    p.add_argument("--extrude_inlet", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--min_torsion", action=argparse.BooleanOptionalAction, default=False,
                   help="minimise torsion when propagating the branch frames")
    p.add_argument("--overwrite", action="store_true",
                   help="reprocess cases whose smoothed mesh already exists")
    p.add_argument("--cases", nargs="*", default=None,
                   help="only these case names (default: everything under --post_dir)")
    p.add_argument("--exclude", nargs="*", default=[],
                   help="case names to skip")
    p.add_argument("--limit", type=int, default=None, help="stop after N cases")
    return p


def main():
    args = build_parser().parse_args()
    _check_obj_export()
    vr = _vessel_reconstruct()

    source_root_arg, source_filename = MESH_SOURCES[args.mesh_source]
    source_root = getattr(args, source_root_arg)
    if not source_root or not os.path.isdir(source_root):
        raise SystemExit("--mesh_source %s reads from --%s, which is not a directory: %r"
                         % (args.mesh_source, source_root_arg, source_root))

    cases = args.cases or sorted(os.listdir(args.post_dir))
    cases = [c for c in cases if c not in set(args.exclude)]
    print("%d candidate case(s) | source %s (%s) | save_root %s"
          % (len(cases), args.mesh_source, source_filename, args.save_root))

    done, skipped, failed, attempted = 0, 0, [], 0
    for case in cases:
        case_post = os.path.join(args.post_dir, case)
        save_dir = os.path.join(args.save_root, case)
        smoothed = os.path.join(save_dir, "ghd_smoothed_reconstruction.obj")
        complex_mesh = os.path.join(source_root, case, source_filename)

        if not os.path.isdir(case_post):
            continue
        if not os.path.exists(complex_mesh):
            print("[%s] no %s, skipping" % (case, source_filename))
            skipped += 1
            continue
        if os.path.exists(smoothed) and not args.overwrite:
            skipped += 1
            continue

        attempted += 1
        os.makedirs(save_dir, exist_ok=True)
        copy_alongside(case_post, save_dir, COPY_ALONGSIDE)

        params = dict(DEFAULT_PARAMS,
                      extrude_outlets=args.extrude_outlets,
                      extrude_inlet=args.extrude_inlet,
                      min_torsion=args.min_torsion)
        print("[%s]" % case)
        try:
            while True:
                vr.ghd_forward_mesh_fusion(
                    save_dir=save_dir,
                    r_ghd_mesh_filename=complex_mesh,
                    r_clipped_cl_filename=os.path.join(case_post, "clipped_centerline.npy"),
                    r_forward_fusion_info_filename=os.path.join(
                        case_post, "forward_fusion_info.npz"),
                    r_branch_ranking_filename=os.path.join(case_post, "branch_ranking.npy"),
                    w_ghd_reconstructed_filename=os.path.join(
                        save_dir, "ghd_reconstructed.obj"),
                    w_ghd_forward_fusion_info_filename=os.path.join(
                        save_dir, "ghd_forward_fusion_info.npz"),
                    w_merged_mesh_filename=os.path.join(
                        save_dir, "ghd_merged_reconstruction.obj"),
                    w_smoothed_mesh_filename=smoothed,
                    **params)
                if args.automatic:
                    break
                print("  surface:\n    file://%s" % urllib.parse.quote(smoothed, safe="/"))
                if input("Redo propagation? (y/n, default n): ").lower() != "y":
                    break
                params = ask_params(params)
        except Exception as exc:
            print("  failed: %r" % exc)
            failed.append({"case": case, "error": repr(exc)})
            if args.limit and attempted >= args.limit:
                break
            continue

        # Parameters go next to the mesh they produced, so a case is self-describing.
        with open(os.path.join(save_dir, "ghd_fusion_params.json"), "w") as f:
            json.dump(params, f, indent=2)
        done += 1
        if args.limit and attempted >= args.limit:
            break

    print("generated %d, skipped %d, failed %d" % (done, skipped, len(failed)))
    if failed:
        path = os.path.join(args.save_root, "surface_mesh_failures.json")
        os.makedirs(args.save_root, exist_ok=True)
        with open(path, "w") as f:
            json.dump(failed, f, indent=2)
        print("failures -> %s" % path)


if __name__ == "__main__":
    main()
