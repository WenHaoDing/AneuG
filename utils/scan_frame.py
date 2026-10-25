"""Move geometry between the scan's true world frame and the "direction-dropped" frame.

AneuSeg's `forward_marching_cubes` (IAgents/tools/reconstruction.py) wrote surface vertices
as `origin + spacing * idx`, leaving out the scan's direction matrix D. Every mesh and
centerline built from those surfaces inherits that frame, including the ImperialNHS CFD
meshes in angioflowv2_merged. The true scanner frame is `origin + D @ (spacing * idx)`,
so a point in the dropped frame maps back with

    V_world = origin + D @ (V_dropped - origin)

This is the fix `fix_mesh_affine` in Aneu_GHD/create_cfd_mesh_reference.py applied to the
cfd_mesh_reference meshes. It is exact and per scan; no registration is involved. On a
scan whose direction is the identity, both frames coincide.

Nothing here writes files: every function returns new arrays or meshes.
"""
from __future__ import annotations

import numpy as np
import SimpleITK as sitk


def scan_affine(reference):
    """(origin [3], direction [3, 3]) of a scan, given a SimpleITK image or a path to one.

    For a path only the header is read, so this is cheap on large volumes.
    """
    if isinstance(reference, sitk.Image):
        origin, direction = reference.GetOrigin(), reference.GetDirection()
    else:
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(reference))
        reader.ReadImageInformation()
        origin, direction = reader.GetOrigin(), reader.GetDirection()
    return np.asarray(origin, dtype=np.float64), np.asarray(direction, dtype=np.float64).reshape(3, 3)


def restore_scan_direction(points, origin, direction):
    """Direction-dropped frame -> true scanner frame. points: [N, 3]."""
    points = np.asarray(points, dtype=np.float64)
    return origin + (points - origin) @ np.asarray(direction).T


def drop_scan_direction(points, origin, direction):
    """True scanner frame -> direction-dropped frame (inverse of restore_scan_direction)."""
    points = np.asarray(points, dtype=np.float64)
    return origin + (points - origin) @ np.linalg.inv(np.asarray(direction)).T


def restore_mesh_scan_direction(mesh, reference):
    """Copy of `mesh` (pyvista or trimesh) moved into the true scanner frame of `reference`
    (a SimpleITK image or a path to one). The input mesh is left unchanged."""
    origin, direction = scan_affine(reference)
    out = mesh.copy()
    if hasattr(out, "vertices"):     # trimesh
        out.vertices = restore_scan_direction(out.vertices, origin, direction)
    else:                            # pyvista
        out.points = restore_scan_direction(out.points, origin, direction)
    return out
