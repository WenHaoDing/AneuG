"""
Pymeshlab functions for connection smoothing.
"""
import pymeshlab as pml
from pytorch3d.structures import Meshes
import torch


def pymeshlab_smoothing(mesh_p3d, adjacent_verts_idx, stepsmoothnum=25):
    mesh_pymesh = pml.Mesh(vertex_matrix=mesh_p3d.verts_packed().detach().cpu().numpy(), 
                           face_matrix=mesh_p3d.faces_packed().detach().cpu().numpy())
    ms = pml.MeshSet()
    ms.add_mesh(mesh_pymesh)
    condition_str = ' || '.join([f'(vi == {idx})' for idx in adjacent_verts_idx])
    ms.compute_selection_by_condition_per_vertex(condselect=condition_str)
    # ms.apply_coord_laplacian_smoothing(stepsmoothnum=stepsmoothnum, selected=True, cotangentweight=False)
    ms.apply_coord_laplacian_smoothing(stepsmoothnum=stepsmoothnum, selected=True)
    ms.apply_coord_laplacian_smoothing(stepsmoothnum=2, selected=False, cotangentweight=False)
    mesh_pymesh = ms.current_mesh()
    smoothed_mesh_p3d = Meshes(verts=[torch.Tensor(mesh_pymesh.vertex_matrix())],
                               faces=[torch.Tensor(mesh_pymesh.face_matrix())])
    return smoothed_mesh_p3d