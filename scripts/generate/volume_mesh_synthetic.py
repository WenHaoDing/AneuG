"""
Tetrahedralize synthetic aneurysm bundles into boundary-tagged volume meshes.

This is the synthetic-data counterpart to cfd_mesher.generate_cfd_volume_meshes, which
targets real segmented cases. It reuses the same core machinery
(cfd_mesher.volume_mesh / cfd_mesher.vmtk_backend: extension, adaptive edge,
relabeling, flow-split, Fluent export) but computes its own inlet/outlet reference
points, since a synthetic bundle (scripts/generate/generate_synthetic.py, optionally
mesh-regularized by scripts/generate/regularize_synthetic.py) has no
*_forward_fusion_info.npz -- no cpcd_glo, no opening_centroids.

Where the reference points come from instead: every sample of a given aneurysm type is
a deformation of the SAME canonical template, and dataset/canonical/<Type>/openings.npz
records that template's `num_openings` fixed, ordered vertex-index sets. Every sample's
own ghd_fitted_world.obj carries the identical vertex indexing (it IS that template,
just deformed), so indexing it at openings.npz's indices gives this sample's own opening
centroids in world space -- with the same opening ordering as every other sample of that
type. Which opening is the inlet is therefore a fixed property of the aneurysm type, not
something re-derived per sample.

IMPORTANT -- see INLET_OPENING_INDEX below: this was verified geometrically for
Bifurcated only (opening 0 is the widest of the three rings on the canonical template --
1.15 vs 0.93 vs 0.81 mm -- consistent with it being the parent vessel feeding two
narrower daughter branches). Sidewall's two openings measure within 2% of each other
(1.41 vs 1.44 mm), so there is no geometric signal to check against; index 0 = inlet is
carried over there only for consistency with this codebase's existing "index/branch 0 =
inlet" convention (opening_centroids[0], cpcd_glo[0] in the real-data pipeline), not
because it has been independently verified. If Sidewall cases come out with backwards
flow, this is the first place to look.

    conda activate vmtk_autogen
    python scripts/generate/volume_mesh_synthetic.py --bundle_root v2/synthetic_v1

Per case (synth_NNNNN_aneurysm1/), written in place alongside the export bundle:

    ghd_surface.vtp             the meshed surface (regularized if present) as .vtp
    ghd_surface_extended.vtp    only if --extend_inlet/--extend_outlets is set
    mesh.vtu                    the tagged volume mesh
    mesh.msh                    the same mesh, converted to Fluent format
    inlet_centroids.csv         every inlet node's coordinates, scaled by --inlet_scale
    flowsplit_ratio.txt         two-outlet cases only: normalized per-outlet flow split
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path = [p for p in sys.path if Path(p or ".").resolve() != ROOT]
sys.path.insert(0, str(ROOT))

CANONICAL_ROOT = ROOT / "dataset" / "canonical"
CANONICAL_FOLDER_BY_ANEU_TYPE = {"bifurcation": "Bifurcated", "sidewall": "Sidewall"}

# Which canonical opening index is the inlet, per canonical folder. Verified for
# Bifurcated (widest ring); assumed, NOT verified, for Sidewall -- see module docstring.
INLET_OPENING_INDEX = {"Bifurcated": 0, "Sidewall": 0}

DOME_FILENAME = "ghd_fitted_world.obj"
LANDMARKS_FILENAME = "landmarks.npz"
RAW_SURFACE_FILENAME = "merge_mesh_world.obj"
REGULARIZED_SURFACE_FILENAME = "merge_mesh_world_regularized.obj"

# Same values cfd_mesher.generate_cfd_volume_meshes defaults to; see that module for
# where they came from (get_mesh_aneux_tune_elements_v3.py's actually-run __main__).
DEFAULT_MESH_PARAMS = dict(
    base_edge=0.13, ref_volume=320.0, ref_area=320.0,
    vol_exponent=0.30, area_exponent=0.50, max_edge=1.0,
)


def _check_vmtk_available():
    if shutil.which("vmtkmeshgenerator") is None:
        raise SystemExit(
            "vmtkmeshgenerator is not on PATH; this pipeline shells out to the vmtk CLI.\n"
            "  Use the 'vmtk_autogen' conda environment (or wherever vmtk's CLI tools are "
            "installed), not just a Python env with the vmtk package importable.")


def _volume_mesh():
    """Imported lazily so --help works even where vmtk is not installed."""
    try:
        from cfd_mesher import volume_mesh as vm
        from cfd_mesher import vmtk_backend as backend
        return vm, backend
    except ImportError as exc:                                  # pragma: no cover
        raise SystemExit(
            "This script needs vmtk (and vtk, pandas, pyvista, trimesh) importable; the "
            "'vmtk_autogen' conda environment has these.\n"
            "  underlying import error: %s" % exc)


def opening_centroids_for_sample(sample_dir, canonical_root=CANONICAL_ROOT):
    """(inlet_centroid, outlet_centroids) for one synth_*_aneurysm1/ sample, computed
    from its own ghd_fitted_world.obj against the canonical template's fixed opening
    definitions. See module docstring for why this is where reference points for a
    synthetic sample come from, and the inlet-index caveat for Sidewall.
    """
    import trimesh

    lm = np.load(os.path.join(sample_dir, LANDMARKS_FILENAME), allow_pickle=True)
    aneu_type = str(lm["aneu_type"])
    canonical_folder = CANONICAL_FOLDER_BY_ANEU_TYPE.get(aneu_type)
    if canonical_folder is None:
        raise RuntimeError("Unknown aneu_type %r in %s" % (aneu_type, sample_dir))

    openings = np.load(os.path.join(canonical_root, canonical_folder, "openings.npz"))
    n_openings = int(openings["num_openings"])

    # process=False: vertex order must match openings.npz's indices exactly, which
    # trimesh's default vertex-merging/reindexing would silently break.
    dome = trimesh.load(os.path.join(sample_dir, DOME_FILENAME), process=False)
    verts = np.asarray(dome.vertices)
    centroids = [verts[openings["indices_%d" % i]].mean(axis=0) for i in range(n_openings)]

    inlet_idx = INLET_OPENING_INDEX[canonical_folder]
    inlet_centroid = centroids[inlet_idx]
    outlet_centroids = [c for i, c in enumerate(centroids) if i != inlet_idx]
    return inlet_centroid, outlet_centroids


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle_root", required=True,
                   help="a generate_synthetic.py --out directory (contains synth_*_aneurysm1/)")
    p.add_argument("--canonical_root", default=str(CANONICAL_ROOT),
                   help="dataset/canonical directory (Bifurcated/Sidewall templates)")
    p.add_argument("--prefer_regularized", action=argparse.BooleanOptionalAction, default=True,
                   help="mesh %s if present, else fall back to %s"
                        % (REGULARIZED_SURFACE_FILENAME, RAW_SURFACE_FILENAME))

    p.add_argument("--base_edge", type=float, default=DEFAULT_MESH_PARAMS["base_edge"])
    p.add_argument("--ref_volume", type=float, default=DEFAULT_MESH_PARAMS["ref_volume"])
    p.add_argument("--ref_area", type=float, default=DEFAULT_MESH_PARAMS["ref_area"])
    p.add_argument("--vol_exponent", type=float, default=DEFAULT_MESH_PARAMS["vol_exponent"])
    p.add_argument("--area_exponent", type=float, default=DEFAULT_MESH_PARAMS["area_exponent"])
    p.add_argument("--max_edge", type=float, default=DEFAULT_MESH_PARAMS["max_edge"])
    p.add_argument("--inflation", action=argparse.BooleanOptionalAction, default=True,
                   help="add boundary-layer sublayers at the wall")

    p.add_argument("--extend_inlet", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--inlet_extension_length", type=float, default=3.0)
    p.add_argument("--extend_outlets", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--outlet_extension_length_one_outlet", type=float, default=3.0)
    p.add_argument("--outlet_extension_length_two_outlets", type=float, default=3.0)

    p.add_argument("--flow_split_exponent", type=float, default=2.45)
    p.add_argument("--inlet_scale", type=float, default=0.001,
                   help="scale factor applied to inlet node coordinates in the CSV export")

    p.add_argument("--overwrite", action="store_true",
                   help="reprocess cases whose mesh.vtu and mesh.msh already both exist")
    p.add_argument("--cases", nargs="*", default=None,
                   help="only these case names (default: everything under --bundle_root)")
    p.add_argument("--exclude", nargs="*", default=[], help="case names to skip")
    p.add_argument("--limit", type=int, default=None, help="stop after N cases")
    return p


def main():
    args = build_parser().parse_args()
    _check_vmtk_available()
    vm, backend = _volume_mesh()

    cases = args.cases or sorted(os.listdir(args.bundle_root))
    cases = [c for c in cases if c not in set(args.exclude)]
    print("%d candidate case(s) | bundle_root %s" % (len(cases), args.bundle_root))

    done, skipped, failed, attempted = 0, 0, [], 0
    for case in cases:
        sample_dir = os.path.join(args.bundle_root, case)
        dome_path = os.path.join(sample_dir, DOME_FILENAME)
        landmarks_path = os.path.join(sample_dir, LANDMARKS_FILENAME)
        regularized_path = os.path.join(sample_dir, REGULARIZED_SURFACE_FILENAME)
        raw_path = os.path.join(sample_dir, RAW_SURFACE_FILENAME)
        vtp_path = os.path.join(sample_dir, "ghd_surface.vtp")
        vtu_path = os.path.join(sample_dir, "mesh.vtu")
        msh_path = os.path.join(sample_dir, "mesh.msh")

        if not os.path.isdir(sample_dir):
            continue
        if not (os.path.exists(dome_path) and os.path.exists(landmarks_path)):
            print("[%s] missing %s or %s, skipping" % (case, DOME_FILENAME, LANDMARKS_FILENAME))
            skipped += 1
            continue
        surface_path = (regularized_path if args.prefer_regularized
                                            and os.path.exists(regularized_path) else raw_path)
        if not os.path.exists(surface_path):
            print("[%s] no %s, skipping" % (case, os.path.basename(surface_path)))
            skipped += 1
            continue
        if os.path.exists(vtu_path) and os.path.exists(msh_path) and not args.overwrite:
            skipped += 1
            continue

        attempted += 1
        print("[%s] meshing %s" % (case, os.path.basename(surface_path)))
        try:
            inlet_c, outlet_c = opening_centroids_for_sample(sample_dir, args.canonical_root)

            if not os.path.exists(vtp_path) or args.overwrite:
                import pyvista as pv
                pv.read(surface_path).save(vtp_path)

            surface_for_meshing = vtp_path
            if args.extend_inlet or args.extend_outlets:
                extended_path = os.path.join(sample_dir, "ghd_surface_extended.vtp")
                surface_for_meshing = extended_path
                if not os.path.exists(extended_path) or args.overwrite:
                    vm.extend_inlet_and_outlets(
                        surface_file=vtp_path, output_file=extended_path,
                        inlet_centroid=inlet_c, outlet_centroids=outlet_c,
                        extend_inlet=args.extend_inlet,
                        inlet_extension_length=args.inlet_extension_length,
                        extend_outlets=args.extend_outlets,
                        outlet_extension_length_one_outlet=args.outlet_extension_length_one_outlet,
                        outlet_extension_length_two_outlets=args.outlet_extension_length_two_outlets)

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

            vm.relabel_inlet_outlets_by_centroids(vtu_path, inlet_c, outlet_c)

            inlet_csv = os.path.join(sample_dir, "inlet_centroids.csv")
            if not os.path.exists(inlet_csv) or args.overwrite:
                vm.scan_inlet_nodes(vtu_path, csv_path=inlet_csv, scale_factor=args.inlet_scale)

            vm.write_flowsplit_ratio(vtu_path, exponent=args.flow_split_exponent,
                                     output_path=os.path.join(sample_dir, "flowsplit_ratio.txt"))

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
        path = os.path.join(args.bundle_root, "volume_mesh_failures.json")
        with open(path, "w") as f:
            json.dump(failed, f, indent=2)
        print("failures -> %s" % path)


if __name__ == "__main__":
    main()
