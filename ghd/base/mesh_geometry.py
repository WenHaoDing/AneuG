import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter

from pytorch3d.structures import Meshes
from pytorch3d.ops import mesh_face_areas_normals
from pytorch3d.loss.point_mesh_distance import point_face_distance
from einops import rearrange, einsum, repeat

import numpy as np
from pytorch3d.ops import cot_laplacian
from pytorch3d import _C


class MeshThickness(nn.Module):
    def __init__(self, r=0.2, num_bundle_filtered=10, innerp_threshold=0.4, num_sel=10):
        """
        Args:
            r: the weight of the euclidean distance D_b = D_n + r*D_e
            num_bundle_filtered: the number of faces to be filtered by D_b
            innerp_threshold: remove faces of the normal with inner product < 0.7
            num_sel: the number of faces to be selected finally to calculate the min distance viewed as the thickness
        """

        super(MeshThickness, self).__init__()

        self.r = r  # the weight of the euclidean distance D_b = D_n + r*D_e

        self.num_bundle_filtered = num_bundle_filtered  # the number of faces to be filtered by D_b

        self.innerp_threshold = innerp_threshold  # remove faces of the normal with inner product < 0.7

        self.num_sel = num_sel  # the number of faces to be selected finally to calculate the min distance viewed as the thickness

    def forward(self, current_mesh: Meshes):
        pt, pt_normals = current_mesh.verts_padded(), current_mesh.verts_normals_padded()

        # Get the interior normal
        pt_normals_inv = -pt_normals

        # Get the faces normals and centeroid
        faces_normals = current_mesh.faces_normals_padded()
        faces_centeroid = current_mesh.verts_packed()[current_mesh.faces_packed(), :].mean(dim=-2)

        ## normal distance
        inner_value = pt_normals_inv.matmul(faces_normals.transpose(-2, -1))
        normal_dist = 1. - inner_value

        ## euclidean distance
        ec_dist = torch.cdist(pt[0], faces_centeroid)
        ec_dist = (ec_dist - ec_dist.mean(dim=-1, keepdim=True))
        ec_dist = ec_dist / ec_dist.std(dim=-1, keepdim=True)

        ## normal bundle distance
        distant_ed_with_normal = normal_dist + ec_dist * self.r

        ## filter the top closest faces
        face_index = distant_ed_with_normal.argsort(dim=-1, descending=False)[0, :, :self.num_bundle_filtered]
        selected_normals = faces_normals[:, face_index, :]

        ## remove the wrong directed faces (double check)
        inner_value_selected = (pt_normals_inv.unsqueeze(-2) * selected_normals).sum(dim=-1)
        inner_value_selected = torch.where(inner_value_selected > 0.5, inner_value_selected,
                                           -torch.ones_like(inner_value_selected))

        ### get the final face selected
        face_reletive_index_1 = inner_value_selected.argsort(dim=-1, descending=True)[0, :, :self.num_sel]
        face_reletive_index_0 = torch.arange(0, pt.shape[1]).unsqueeze(-1).repeat(1, face_reletive_index_1.shape[-1])
        face_index = face_index[face_reletive_index_0, face_reletive_index_1]

        ## Thickness
        face_vert_indx = current_mesh.faces_packed()[face_index, :]
        tri = current_mesh.verts_packed()[face_vert_indx, :]
        tri_normals = current_mesh.faces_normals_packed()[face_index, :]
        tri_normals = tri_normals.view(-1, 3)
        tri_centeroid = tri.mean(dim=-2)
        tri_centeroid = tri_centeroid.view(-1, 3)
        pt_first_idx = torch.arange(pt.shape[1]).view(-1).to(pt.device)
        tris_first_idx = torch.tensor([i * tri.shape[1] for i in range(tri.shape[0])]).to(tri.device)

        dist, indx = _C.point_face_dist_forward(pt.view(-1, 3), pt_first_idx, tri.view(-1, 3, 3), tris_first_idx, 1,
                                                1e-6)
        ## signed
        dist_v = (pt.view(-1, 3) - tri_centeroid[indx, :])
        sign = -(dist_v * tri_normals[indx, :]).sum(dim=-1)
        closed_indx = face_index.view(-1)[indx]
        return dist, dist_v.norm(dim=-1), closed_indx, sign
    

def faces_angle(meshs: Meshes)->torch.Tensor:
    """
    Compute the angle of each face in a mesh
    Args:
        meshs: Meshes object
    Returns:
        angles: Tensor of shape (N,3) where N is the number of faces
    """
    Face_coord = meshs.verts_packed()[meshs.faces_packed()]
    A = Face_coord[:,1,:] - Face_coord[:,0,:]
    B = Face_coord[:,2,:] - Face_coord[:,1,:]
    C = Face_coord[:,0,:] - Face_coord[:,2,:]
    angle_0 = torch.arccos(-torch.sum(A*C,dim=1)/(torch.norm(A,dim=1))/(torch.norm(C,dim=1)))
    angle_1 = torch.arccos(-torch.sum(A*B,dim=1)/(torch.norm(A,dim=1))/(torch.norm(B,dim=1)))
    angle_2 = torch.arccos(-torch.sum(B*C,dim=1)/(torch.norm(B,dim=1))/(torch.norm(C,dim=1)))
    angles = torch.stack([angle_0,angle_1,angle_2],dim=1)
    return angles

def dual_area_weights_faces(Surfaces: Meshes)->torch.Tensor:
    """
    Compute the dual area weights of 3 vertices of each triangles in a mesh
    Args:
        Surfaces: Meshes object
    Returns:
        dual_area_weight: Tensor of shape (N,3) where N is the number of triangles
        the dual area of a vertices in a triangles is defined as the area of the sub-quadrilateral divided by three perpendicular bisectors
    """
    angles = faces_angle(Surfaces)
    sin2angle = torch.sin(2*angles)
    dual_area_weight = sin2angle/torch.sum(sin2angle,dim=-1,keepdim=True)
    dual_area_weight = (dual_area_weight[:,[2,0,1]]+dual_area_weight[:,[1,2,0]])/2
    return dual_area_weight


def dual_area_vertex(Surfaces: Meshes)->torch.Tensor:
    """
    Compute the dual area of each vertices in a mesh
    Args:
        Surfaces: Meshes object
    Returns:
        dual_area_per_vertex: Tensor of shape (N,1) where N is the number of vertices
        the dual area of a vertices is defined as the sum of the dual area of the triangles that contains this vertices
    """
    dual_weights = dual_area_weights_faces(Surfaces)
    dual_areas = dual_weights*Surfaces.faces_areas_packed().view(-1,1)
    face2vertex_index = Surfaces.faces_packed().view(-1)
    dual_area_per_vertex = scatter(dual_areas.view(-1), face2vertex_index, reduce='sum')
    return dual_area_per_vertex.view(-1,1)


def gaussian_curvature(Surfaces: Meshes,return_topology=False)->torch.Tensor:
    """
    Compute the gaussian curvature of each vertices in a mesh by local Gauss-Bonnet theorem
    Args:
        Surfaces: Meshes object
        return_topology: bool, if True, return the Euler characteristic and genus of the mesh
    Returns:
        gaussian_curvature: Tensor of shape (N,1) where N is the number of vertices
        the gaussian curvature of a vertices is defined as the sum of the angles of the triangles that contains this vertices minus 2*pi and divided by the dual area of this vertices
    """
    face2vertex_index = Surfaces.faces_packed().view(-1)
    angle_face = faces_angle(Surfaces)
    dual_weights = dual_area_weights_faces(Surfaces)
    dual_areas = dual_weights*Surfaces.faces_areas_packed().view(-1,1)
    dual_area_per_vertex = scatter(dual_areas.view(-1), face2vertex_index, reduce='sum')
    angle_sum_per_vertex = scatter(angle_face.view(-1), face2vertex_index, reduce='sum')
    _, inverse_A = cot_laplacian(Surfaces.verts_packed(), Surfaces.faces_packed())
    curvature = (2*torch.pi - angle_sum_per_vertex) * inverse_A
    if return_topology:
        Euler_chara = (curvature*dual_area_per_vertex).sum()/2/torch.pi
        return curvature, Euler_chara
    return curvature