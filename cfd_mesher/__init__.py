"""CFD meshing: turn an aneurysm complex plus centreline into meshes a solver can run.

Two stages. The surface stage propagates the vessel branches along the clipped centreline
and fuses them into the complex, producing a closed-except-for-openings surface. The
volume stage (to be moved in) meshes that surface into tetrahedra with vmtk.

    python -m cfd_mesher.generate_cfd_surface_meshes --help
"""

from .config import (POST_DIR, GHD_DIR, SURFACE_SAVE_ROOT, VOLUME_SAVE_ROOT,
                     PACKAGE_DIR)

__all__ = ["POST_DIR", "GHD_DIR", "SURFACE_SAVE_ROOT", "VOLUME_SAVE_ROOT", "PACKAGE_DIR"]
