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

# -- volume stage (populated when the vmtk mesher is moved in) ---------------------------
VOLUME_SAVE_ROOT = _env("CFDMESH_VOLUME_SAVE_ROOT", "")
