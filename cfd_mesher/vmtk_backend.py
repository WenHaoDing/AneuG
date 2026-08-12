"""Thin wrappers around the vmtk CLI and vmtkscripts mesh I/O.

Ported from AneuSeg/cfd_meshing/vmtk_cfdmesher.py and vmtk_vtu2msh.py. Only the
single-file entry points the volume-meshing pipeline actually calls are kept
(cfdmesher_multiple/cfdmesher_single/write_msh_multi were unused by the driving
script and are not ported).

vmtkmeshgenerator is shelled out to rather than driven through vmtkscripts because
that is how the original was built and verified; vmtk's own Python bindings for the
mesh generator are not as commonly exercised. This means the vmtk CLI binaries (not
just the `vmtk` python package) must be on PATH -- true in the `vmtk_autogen` conda
environment this package already documents using for the surface stage.
"""

import os

import vmtk


def cfdmesher_custom(ifile, ofile, edge, max_edge, inflation):
    """Tetrahedralize `ifile` (a closed-except-openings surface) into `ofile` (.vtu).

    `edge` / `max_edge` bound the target element size; `inflation` is "y" to add
    boundary-layer sublayers at the wall (skipped on caps) or "n" for a uniform mesh.
    """
    if inflation == "y":
        arg = (
            ' vmtkmeshgenerator -ifile "%s" -edgelength %s -maxedgelength %s '
            ' -boundarylayer 1 -thicknessfactor 0.5 -sublayers 4 -sublayerratio 0.8 '
            ' -boundarylayeroncaps 0 -tetrahedralize 1 -ofile "%s"'
            % (ifile, edge, max_edge, ofile)
        )
    else:
        arg = ' vmtkmeshgenerator -ifile "%s" -edgelength %s -ofile "%s"' % (ifile, edge, ofile)
    if os.system(arg) != 0:
        raise RuntimeError("vmtkmeshgenerator failed on %r (see console output above)" % ifile)


def write_msh_single(ifile, ofile):
    """Convert a vmtk-readable mesh (.vtu) to Fluent .msh."""
    mesh_reader = vmtk.vmtkmeshreader.vmtkMeshReader()
    mesh_reader.InputFileName = ifile
    mesh_reader.Execute()

    writer = vmtk.vmtkmeshwriter.vmtkMeshWriter()
    writer.Mesh = mesh_reader.Mesh
    writer.OutputFileName = ofile
    writer.Execute()
