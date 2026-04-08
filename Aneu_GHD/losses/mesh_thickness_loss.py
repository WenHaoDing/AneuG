import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d import _C
from pytorch3d.structures import Meshes


class MeshThickness(nn.Module):
    def __init__(self, r=0.2, num_bundle_filtered=10, innerp_threshold=0.4, num_sel=10):
        """
        Args:
            r: the weight of the euclidean distance D_b = D_n + r*D_e
            num_bundle_filtered: the number of faces to be filtered by D_b
            innerp_threshold: remove faces of the normal with inner product < threshold
            num_sel: the number of faces to be selected finally to calculate the min distance viewed as the thickness
        """
        super(MeshThickness, self).__init__()

        self.r = r
        self.num_bundle_filtered = num_bundle_filtered
        self.innerp_threshold = innerp_threshold
        self.num_sel = num_sel

    def forward(self, current_mesh: Meshes):
        pt, pt_normals = current_mesh.verts_padded(), current_mesh.verts_normals_padded()

        # Get the interior normal
        pt_normals_inv = -pt_normals

        # Get the faces normals and centroid
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
