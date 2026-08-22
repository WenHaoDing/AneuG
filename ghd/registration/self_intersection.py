"""self_intersection.py -- genuine self-intersection detection for a
triangle mesh, as opposed to the WEAKER proxy checks used earlier in this
investigation (is_watertight / is_winding_consistent / degenerate-face
area): those are necessary but not sufficient -- a mesh can pass all
three and still have two non-adjacent regions of itself crossing through
each other.

Requires python-fcl (`pip install python-fcl`), which unlocks trimesh's
CollisionManager. Registers each FACE as its own named collision object
(not the whole mesh as one object -- that would only test the mesh
against something else, not against itself) and asks the manager which
pairs collide. Adjacent faces (sharing a vertex) always "collide" at
that shared point, which isn't a real self-intersection, so those pairs
are filtered out before counting.
"""

import numpy as np
import trimesh
from trimesh.collision import CollisionManager


def find_self_intersections(mesh):
    """mesh: trimesh.Trimesh. Returns a list of (face_i, face_j) index
    pairs whose triangles genuinely cross (share no vertex, but collide).
    Empty list = no self-intersection detected."""
    manager = CollisionManager()
    for i, f in enumerate(mesh.faces):
        tri = trimesh.Trimesh(vertices=mesh.vertices[f], faces=[[0, 1, 2]], process=False)
        manager.add_object(str(i), tri)

    in_collision, names = manager.in_collision_internal(return_names=True)
    if not in_collision:
        return []

    face_verts = mesh.faces
    real_pairs = []
    for a, b in names:
        i, j = int(a), int(b)
        if len(set(face_verts[i]) & set(face_verts[j])) == 0:
            real_pairs.append((i, j))
    return real_pairs


def report_self_intersections(mesh, label=""):
    """Prints a one-line summary and returns the pair list -- convenience
    wrapper for use inline in a fitting/registration script's sanity
    checks."""
    pairs = find_self_intersections(mesh)
    tag = f" ({label})" if label else ""
    if not pairs:
        print(f"  self-intersection check{tag}: none found "
              f"({len(mesh.faces)} faces)", flush=True)
    else:
        involved = sorted(set(i for p in pairs for i in p))
        centroids = mesh.triangles_center[involved]
        print(f"  self-intersection check{tag}: {len(pairs)} intersecting pair(s), "
              f"{len(involved)} face(s) involved, clustered around "
              f"{centroids.mean(axis=0).round(3).tolist()} "
              f"(bbox {centroids.min(axis=0).round(3).tolist()} .. "
              f"{centroids.max(axis=0).round(3).tolist()})", flush=True)
    return pairs
