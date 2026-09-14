"""Filesystem locations for the CFD meshing stages.

Every path can be overridden by an environment variable so the package is not tied to one
workstation; the defaults reflect the layout used during development.

    export CFDMESH_POST_DIR=/path/to/PostTr
    export CFDMESH_SURFACE_SAVE_ROOT=/path/to/surfaces
"""

import os


def _env(name, default):
    return os.environ.get(name, default)


PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

# -- surface stage ----------------------------------------------------------------------
# per-case clipped centrelines, fusion info and branch rankings
POST_DIR = _env("CFDMESH_POST_DIR",
                "/media/yaplab2/HDD Storage/wenhao/AneuSeg/PostTr_debug/AneuX_CFD")
# per-case GHD fitting results, read when --mesh_source ghd
GHD_DIR = _env("CFDMESH_GHD_DIR",
               "/media/yaplab2/HDD Storage/wenhao/AneuSeg/AneuG/Aneu_GHD/fitting_results/job1")
SURFACE_SAVE_ROOT = _env(
    "CFDMESH_SURFACE_SAVE_ROOT",
    "/media/yaplab2/HDD Storage/wenhao/AneuSeg/cfd_meshing/processed_shapes/aneux_cfd_batch1")

# -- volume stage -------------------------------------------------------------------------
# Volume meshes go to the wd8tb disk, which has the room for them: a tetrahedral mesh
# with boundary layers runs to millions of cells per case. The volume stage reads the
# surface .obj and fusion npz from SURFACE_SAVE_ROOT/<case>/ and writes everything it
# produces under VOLUME_SAVE_ROOT/<case>/. Set this to "" to write in place instead.
VOLUME_SAVE_ROOT = _env("CFDMESH_VOLUME_SAVE_ROOT",
                        "/media/yaplab2/wd8tb/wenhao/angioflow/cfd/AneuGv2")
