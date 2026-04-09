import os.path as osp
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import ShapeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, PointNetConv, fps, global_max_pool, radius, knn_interpolate
from torch_geometric.typing import WITH_TORCH_CLUSTER
from torch_geometric.utils import scatter
from typing import List
from torch_geometric.data import Data, Batch
import numpy as np
from pytorch3d.structures import Meshes
from ..ghd.base.graph_harmonic_deformation import Graph_Harmonic_Deform
import torch.nn as nn
import matplotlib.pyplot as plt
import pickle
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle


class GHD_Reconstruct(object):
    """
    Class for GHD reconstruction
    * This class stores the normalized canonical instead of the original
    * For forward, box norm will normalize the reconstructed shape into a bounding box with size 1.0
    * For forward_as_Meshes, denormalize_shape will return shape as the state just after alignment
    """

    def __init__(self, canonical_Meshes: Meshes,
                 eigen_chk: str,
                 num_Basis=12**2,
                 device=torch.device('cpu')):
        self.canonical_Meshes = canonical_Meshes.to(device)
        # normalize
        self.norm_canonical, self.canonical_Meshes = self.normalize()
        self.forward_norm_canonical = self.canonical_Meshes.verts_packed().norm(dim=-1).max()
        # GHD object
        self.canonical_ghd = Graph_Harmonic_Deform(base_shape=self.canonical_Meshes, num_Basis=num_Basis, eigen_chk=eigen_chk)
        # deformed shape
        self.deformed_Meshes = None
        self.GHD_eigvec = self.canonical_ghd.GBH_eigvec.to(device)

    def normalize(self):
        """
        Normalize the vertices to the canonical mesh
        """
        norm_canonical = torch.max(torch.norm(self.canonical_Meshes.verts_packed(), dim=-1)).detach().item() * 1.10
        norm_canonical *= 2.50
        canonical_Meshes = self.canonical_Meshes.update_padded(self.canonical_Meshes.verts_padded() / norm_canonical)
        return norm_canonical, canonical_Meshes

    def reconstruct(self, ghd_checkpoint, return_ghd_chk=False):
        with open(ghd_checkpoint, 'rb') as f:
            ghd_chk = pickle.load(f)
        R, s, T = ghd_chk['R'], ghd_chk['s'], ghd_chk['T']
        print("R={}, s={}, T={}".format(R, s, T))
        ghd_coefficients = ghd_chk['GHD_coefficient']
        setattr(self.canonical_ghd, 'R', nn.Parameter(R))
        setattr(self.canonical_ghd, 's', nn.Parameter(s))
        setattr(self.canonical_ghd, 'T', nn.Parameter(T))
        self.deformed_Meshes = self.canonical_ghd.forward(ghd_coefficients)
        # denormalize as actual size
        self.deformed_Meshes = self.deformed_Meshes.update_padded(self.deformed_Meshes.verts_padded() * self.norm_canonical)
        if not return_ghd_chk:
            return self.deformed_Meshes
        else:
            return self.deformed_Meshes, ghd_chk
    
    def visualize(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        T = self.canonical_Meshes.faces_packed().detach().cpu().numpy()
        vertices = self.canonical_Meshes.verts_packed().detach().cpu().numpy()
        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=T, edgecolor=[[0, 0, 0]], linewidth=0.01,
                    alpha=0.2, color='dodgerblue')
        T_target = self.deformed_Meshes.faces_packed().detach().cpu().numpy()
        vertices_target = self.deformed_Meshes.verts_packed().detach().cpu().numpy()
        ax.plot_trisurf(vertices_target[:, 0], vertices_target[:, 1], vertices_target[:, 2], triangles=T_target,
                    edgecolor=[[0, 0, 0]], linewidth=0.01, alpha=0.05, color='gray')
        ax.view_init(elev=45, azim=0)
        plt.show()
        plt.close(fig)

    def forward(self, x, mean, std, return_norm=True, box_norm=False, batched=True, normalize=False):
        """
        x: [B, 3*M]
        mean, std [1, 3*M]
        warning: box_norm normalizes shapes into a bounding box with size 1.0
        """
        # generate Meshes
        device = x.device
        B = x.shape[0]
        Meshes_ = Meshes(verts=self.canonical_Meshes.verts_padded().repeat(B, 1, 1), 
                         faces=self.canonical_Meshes.faces_padded().repeat(B, 1, 1))

        # denormalize
        if mean is not None and std is not None:
            mean, std = mean.to(device), std.to(device)
            x = x * std + mean
        else:
            pass
        # reshape
        x = x.reshape(B, -1, 3)
        # reconstruct pcd
        offset = torch.einsum('nm,bmc->bnc', self.GHD_eigvec, x)  # [B, N, 3]
        verts_updated_padded = self.canonical_Meshes.verts_padded() + offset
        if box_norm:
            verts_updated_padded = verts_updated_padded / self.forward_norm_canonical
        # update Meshes
        Meshes_ = Meshes_.update_padded(verts_updated_padded)

        pos = Meshes_.verts_padded()
        if normalize:
            pos -= pos.mean(dim=1, keepdim=True)
            norm = pos.norm(dim=-1).max()
            pos /= norm
        else:
            pass

        # return
        if return_norm:
            pcd = torch.cat([pos, Meshes_.verts_normals_padded()], dim=-1)  # [B, N, 6]
        else:
            pcd = pos
        # 
        if not batched:
            data = tensor2data(pcd, None)
        else:
            data = tensor2batch(pcd)
        return data  # [B, N, 3] or [B, N, 6]
    
    def forward_pcd(self, ghd, mean, std, verts_mask=None):
        """
        forward pcd for classification
        """
        ghd = ghd * std.to(ghd.device) + mean.to(ghd.device)
        B = ghd.shape[0]
        ghd = ghd.reshape(B, -1, 3)
        Meshes_ = Meshes(verts=self.canonical_Meshes.verts_padded().repeat(B, 1, 1), 
                         faces=self.canonical_Meshes.faces_padded().repeat(B, 1, 1))
        # reconstruct pcd
        offset = torch.einsum('nm,bmc->bnc', self.GHD_eigvec, ghd)  # [B, N, 3]
        verts_updated_padded = self.canonical_Meshes.verts_padded() + offset
        Meshes_ = Meshes_.update_padded(verts_updated_padded)
        verts = Meshes_.verts_padded()
        data_list = []
        for b in range(B):
            data = Data()
            data.x = verts[b, verts_mask, :]
            data.pos = data.x[..., :3]
            data_list.append(data)
        batched_data = Batch.from_data_list(data_list)
        return batched_data
    
    def forward_as_Meshes(self, ghd_dataset, denormalize_shape=False, copy_alignment=False):
        """
        return as pytorch3d.Meshes for centerline applications, we directly operate on dataset
        set copy_alignment=True to copy the alignment from the dataset
        set denormalize_shape=True to return the shape as the size just after alignment
        note: for centerline fitting, set both to False
        """
        # generate Meshes
        ghd, alignment = ghd_dataset.ghd, ghd_dataset.alignment
        B = len(ghd)
        ghd = torch.stack(ghd, dim=0).reshape(B, -1, 3)
        device = ghd.device
        Meshes_ = Meshes(verts=self.canonical_Meshes.verts_padded().repeat(B, 1, 1), 
                         faces=self.canonical_Meshes.faces_padded().repeat(B, 1, 1))
        ghd = ghd.reshape(B, -1, 3)
        alignment = torch.stack(alignment, dim=0)
        R, s, T = alignment[:, :3], alignment[:, 3], alignment[:, 4:]
        R_matrix = axis_angle_to_matrix(R)
        offset = torch.einsum('nm,bmc->bnc', self.GHD_eigvec, ghd)  # [B, N, 3]
        verts_updated_padded = self.canonical_Meshes.verts_padded() + offset
        if copy_alignment:
            verts_updated_padded = verts_updated_padded @ R_matrix.transpose(-1,-2)
            verts_updated_padded = verts_updated_padded * s.unsqueeze(-1).unsqueeze(-1) + T.unsqueeze(-2)
        if denormalize_shape:
            verts_updated_padded = verts_updated_padded * self.norm_canonical
        # update Meshes
        Meshes_ = Meshes_.update_padded(verts_updated_padded)
        return Meshes_
    
    def ghd_forward_as_Meshes(self, ghd: torch.Tensor, denormalize_shape=False, trimmed_faces=None, mean=None, std=None, scale=None):
        """
        we train our model in fitting env, set denormalize_shape=True to 
        return the actual size
        """
        if mean is not None and std is not None:
            ghd = ghd * std.to(ghd.device) + mean.to(ghd.device)
        B = ghd.shape[0]
        ghd = ghd.reshape(B, -1, 3)
        Meshes_ = Meshes(verts=self.canonical_Meshes.verts_padded().repeat(B, 1, 1), 
                         faces=self.canonical_Meshes.faces_padded().repeat(B, 1, 1)).to(ghd.device)
        offset = torch.einsum('nm,bmc->bnc', self.GHD_eigvec, ghd)  # [B, N, 3]
        verts_updated_padded = self.canonical_Meshes.verts_padded() + offset
        if denormalize_shape:
            verts_updated_padded = verts_updated_padded * self.norm_canonical
        if scale is not None:
            assert scale.dim() == 2
            verts_updated_padded = verts_updated_padded * scale.unsqueeze(-1)
        # update Meshes
        Meshes_ = Meshes_.update_padded(verts_updated_padded)
        if trimmed_faces is not None:
            faces = trimmed_faces.unsqueeze(0).repeat(B, 1, 1).to(Meshes_.device)
            Meshes_ = Meshes(verts=Meshes_.verts_padded(), faces=faces)
        return Meshes_


def tensor2data(x, y: int):
    """
    x: [B, N, C]
    """
    B, N, C = x.shape
    data = Data()
    data.x = x.reshape(B*N, C)
    data.pos = data.x[:, :3]
    data.batch = torch.arange(B, device=x.device).repeat_interleave(N).reshape(-1)
    if y is not None:
        data.y = torch.ones(B*N, dtype=torch.long) * y  # [B*N] true of false
    return data

def tensor2batch(x):
    B, N, C = x.shape
    data_list = []
    for b in range(B):
        data = Data()
        data.x = x[b, ...]
        data.pos = data.x[:, :3]
        data_list.append(data)
    batched_data = Batch.from_data_list(data_list)
    return batched_data