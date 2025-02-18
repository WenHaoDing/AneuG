import os
import sys
# sys.path.append(os.path.join(os.path.dirname(__file__),'.','..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pytorch3d.loss import chamfer_distance,mesh_laplacian_smoothing, mesh_normal_consistency, mesh_edge_loss
from pytorch3d.ops import sample_points_from_meshes, cot_laplacian, padded_to_packed
from .meshloss import Mesh_loss
from utils_oa.fn_opening_alignment import opening_alignment_pca
import torch
from pytorch3d.structures import Meshes
import numpy as np


class Mesh_loss_opa_dv(Mesh_loss):
    def __init__(self, args, oa_class_canonical: opening_alignment_pca, oa_class_target: opening_alignment_pca):
        self.device = torch.device(args.device)
        base_shape = getattr(oa_class_canonical, "mesh_target_p3d").to(self.device)  # transform o3d mesh to pytorch3d mesh
        self.target_mesh = getattr(oa_class_target, "mesh_target_p3d").to(self.device)
        # TODO: check if chamfer loss cares about normal direction
        # register static meshes for the target mesh
        self.target_openings = oa_class_target.return_opening_Meshes_static(register_normal=False)
        self.sample_num = args.sample_num
        self.op_sample_num = args.op_sample_num
        super(Mesh_loss_opa_dv, self).__init__(mesh_std=base_shape, sample_num=self.sample_num)

    def forward_opening_alignment(self, warped_mesh, warped_openings, loss_weighting: dict, B=1):
        # TODO: write a switch so children classes can skip opa losses
        loss_dict = self.forward(meshes_scr=warped_mesh, trg=self.target_mesh, loss_list=loss_weighting, B=B)
        loss_p_list = []
        loss_n_list = []
        if ('loss_openings_p' in loss_weighting) or ('loss_openings_n' in loss_weighting):
            for idx in range(len(self.target_openings)):
                pcd_wo, nor_wo = sample_points_from_meshes(warped_openings[idx], self.op_sample_num, return_normals=True)
                sample_points_from_meshes(warped_openings[idx], self.op_sample_num, return_normals=False)
                pcd_to, nor_to = sample_points_from_meshes(self.target_openings[idx].to(self.device), self.op_sample_num, return_normals=True)
                loss_p, loss_n = chamfer_distance(pcd_wo, pcd_to, x_normals=nor_wo, y_normals=nor_to)
                loss_p_list.append(loss_p)
                loss_n_list.append(loss_n)
            if 'loss_openings_p' in loss_weighting:
                loss_dict['loss_openings_p'] = loss_p_list
            if 'loss_openings_n' in loss_weighting:
                loss_dict['loss_openings_n'] = loss_n_list
        return loss_dict


class Mesh_loss_opa_do(Mesh_loss):
    """
    opening alignment and differentiable occupancy
    """
    def __init__(self, args, oa_class_canonical: opening_alignment_pca, oa_class_target: opening_alignment_pca):
        self.device = torch.device(args.device)
        base_shape = getattr(oa_class_canonical, "mesh_target_p3d").to(self.device)  # transform o3d mesh to pytorch3d mesh
        self.target_mesh = getattr(oa_class_target, "mesh_target_p3d").to(self.device)
        # TODO: check if chamfer loss cares about normal direction
        # register static meshes for the target mesh
        self.target_openings = oa_class_target.return_opening_Meshes_static(register_normal=False)
        self.sample_num = args.sample_num
        self.op_sample_num = args.op_sample_num
        super(Mesh_loss_opa_do, self).__init__(mesh_std=base_shape, sample_num=self.sample_num)

    def forward_opening_alignment(self, warped_mesh, warped_openings, loss_weighting: dict, B=1):
        # TODO: write a switch so children classes can skip opa losses
        loss_dict = self.forward(meshes_scr=warped_mesh, trg=self.target_mesh, loss_list=loss_weighting, B=B)
        loss_p_list = []
        loss_n_list = []
        if ('loss_openings_p' in loss_weighting) or ('loss_openings_n' in loss_weighting):
            for idx in range(len(self.target_openings)):
                pcd_wo, nor_wo = sample_points_from_meshes(warped_openings[idx], self.op_sample_num, return_normals=True)
                sample_points_from_meshes(warped_openings[idx], self.op_sample_num, return_normals=False)
                pcd_to, nor_to = sample_points_from_meshes(self.target_openings[idx].to(self.device), self.op_sample_num, return_normals=True)
                loss_p, loss_n = chamfer_distance(pcd_wo, pcd_to, x_normals=nor_wo, y_normals=nor_to)
                loss_p_list.append(loss_p)
                loss_n_list.append(loss_n)
            if 'loss_openings_p' in loss_weighting:
                loss_dict['loss_openings_p'] = loss_p_list
            if 'loss_openings_n' in loss_weighting:
                loss_dict['loss_openings_n'] = loss_n_list
        return loss_dict


def o3d_mesh_to_pytorch3d(o3d_mesh) -> Meshes:
    verts = torch.Tensor(np.asarray(o3d_mesh.vertices))
    faces = torch.Tensor(np.asarray(o3d_mesh.triangles))
    pytorch3d_mesh = Meshes(verts=[verts], faces=[faces])
    return pytorch3d_mesh
