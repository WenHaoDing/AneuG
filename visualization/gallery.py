import matplotlib.pyplot as plt
import torch
import os.path as osp
from models.ghd_reconstruct import GHD_Reconstruct
import wandb
from typing import List
from torch_geometric.data import Data, Batch
import random
import string
import math
from models.mesh_plugins import MeshPlugins
import numpy as np
from pytorch3d.structures import Meshes
from pytorch3d.io import save_obj
from random import randint
import os
import pyvista as pv
from models.mesh_plugins import MeshFusion


def generate_random_name():
    # Generate 5 random characters (letters)
    letters = ''.join(random.choices(string.ascii_letters, k=5))
    
    # Generate 5 random digits (numbers)
    digits = ''.join(random.choices(string.digits, k=5))
    return letters + digits


def get_fig(ghd_reconstruct: GHD_Reconstruct, data: Batch, column=4, Title="Meshes", scale=None):
    B = len(data)
    if isinstance(data, Batch):
        data = data.to_data_list()
    else:
        data_list = []
        for i in range(B):
            data_ = Data()
            data_.pos = data[i][:, :3]
            data_list.append(data_)
        data = data_list
    row = math.ceil(B/column)
    
    # Set up a grid of subplots: 2 rows (real and fake), each with 3 plots (for different colors)
    fig, axes = plt.subplots(row, column, subplot_kw={'projection': '3d'}, figsize=(column*5, row*5))
    triangles = ghd_reconstruct.canonical_Meshes.faces_packed().detach().cpu().numpy()

    color = 'dodgerblue'
    for i in range(row):
        for j in range(column):
            idx = i*column+j
            ax = axes[i, j]
            if idx < B:
                data_ = data[idx]
                verts = data_.pos.detach().cpu().numpy()
                if scale is not None:
                    verts = verts * scale[idx].detach().cpu().numpy()
                ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=triangles, edgecolor=[[0, 0, 0]], linewidth=0.01, color=color, alpha=0.2)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_title(f"{idx}", fontsize=16, fontweight='bold')
            ax.grid(False)
    fig.suptitle(Title, fontsize=32, fontweight='bold')
    return fig


def get_fig_advanced(ghd_reconstruct: GHD_Reconstruct, data: Batch, column=4, Title="Meshes",
                     mesh_plugin: MeshPlugins = None, plot_tangent=False,
                     plot_vessel=False, cpcd_glo=None,
                     sub_size=5, plot_neck=False,
                     scale=None,
                     mesh_color='dodgerblue',
                     quiver_color="blue"):
    """
    cpcd_glo: [B, num_branch, dpi, 3]
    """
    if isinstance(data, Batch):
        data = data.to_data_list()
    else:
        data_list = []
        for i in range(B):
            data_ = Data()
            data_.pos = data[i][:, :3]
            data_list.append(data_)
        data = data_list
    B = len(data)
    row = math.ceil(B/column)

    # create Mehses
    verts = torch.stack([data_.pos for data_ in data], dim=0)
    if scale is not None:
        verts = verts * scale.unsqueeze(-1)
    faces = ghd_reconstruct.canonical_Meshes.faces_packed()
    if plot_tangent:
        combined_cap_faces = [face for sublist in mesh_plugin.cap_faces for face in sublist]
        mask = torch.ones(faces.shape[0], dtype=bool)
        mask[combined_cap_faces] = False
        faces = faces[mask]
        if mesh_plugin.trimmed_mesh_path is not None:
            faces = mesh_plugin.trimmed_faces.to(faces.device)
            # faces = mesh_plugin.trimmed_faces_wave.to(faces.device)
    faces = faces.unsqueeze(0).expand(B, -1, -1)
    meshes = Meshes(verts=verts, faces=faces)

    # create tangent vectors
    cpcd_trinity, tangent_pca_trinity, tangent_diff_trinity = mesh_plugin.mesh_forward_tangents(meshes)
    tangent_start_point = [cpcd_[:, mesh_plugin.loop_start[i], :] for i, cpcd_ in enumerate(cpcd_trinity)]
    
    # Set up a grid of subplots: 2 rows (real and fake), each with 3 plots (for different colors)
    fig, axes = plt.subplots(row, column, subplot_kw={'projection': '3d'}, figsize=(column*sub_size, row*sub_size))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    cpcd_color = ['red', 'black', 'blue']
    # First row: Real data
    for i in range(row):
        for j in range(column):
            idx = i*column+j
            ax = axes[i, j]
            if idx < B:
                verts = meshes.verts_list()[idx].detach().cpu().numpy()
                triangles = meshes.faces_list()[idx].detach().cpu().numpy()
                mesh_center = verts.mean(axis=0)
                ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=triangles, edgecolor=[[0, 0, 0]], linewidth=0.01, color=mesh_color, alpha=0.2)
                box_size = np.max(verts - mesh_center, axis=0).max(axis=0)
                ax.set_xlim([mesh_center[0]-box_size, mesh_center[0]+box_size])
                ax.set_ylim([mesh_center[1]-box_size, mesh_center[1]+box_size])
                ax.set_zlim([mesh_center[2]-box_size, mesh_center[2]+box_size])
                if plot_tangent:
                    for k in range(3):
                        start_point = tangent_start_point[k][idx].detach().cpu().numpy()
                        tangent_diff = tangent_diff_trinity[idx, k].detach().cpu().numpy()
                        tangent_pca = tangent_pca_trinity[idx, k].detach().cpu().numpy()
                        ax.quiver(start_point[0], start_point[1], start_point[2], tangent_diff[0], tangent_diff[1], tangent_diff[2], color=quiver_color, length=0.1, normalize=True)
                        # ax.quiver(start_point[0], start_point[1], start_point[2], tangent_pca[0], tangent_pca[1], tangent_pca[2], color='red', length=0.5, normalize=True)
                if plot_vessel:
                    assert cpcd_glo is not None
                    for k in range(3):
                        cpcd = cpcd_glo[idx][k].detach().cpu().numpy()
                        ax.plot(cpcd[:, 0], cpcd[:, 1], cpcd[:, 2], color=cpcd_color[k], linewidth=1.0, zorder=1)
                        ax.scatter(cpcd[::20, 0], cpcd[::20, 1], cpcd[::20, 2], color='yellow', s=1.25, zorder=2)
                        box_size = np.max(cpcd - mesh_center, axis=0).max(axis=0)
                        ax.set_xlim([mesh_center[0]-box_size, mesh_center[0]+box_size])
                        ax.set_ylim([mesh_center[1]-box_size, mesh_center[1]+box_size])
                        ax.set_zlim([mesh_center[2]-box_size, mesh_center[2]+box_size])
                if plot_neck:
                    assert mesh_plugin.neck_idx is not None
                    neck_idx = mesh_plugin.neck_idx
                    neck_verts = verts[neck_idx.cpu().numpy()]
                    ax.scatter(neck_verts[:, 0], neck_verts[:, 1], neck_verts[:, 2], color='red', s=1.25)
            
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_zlabel('')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            # ax.set_title(f"{idx}", fontsize=16, fontweight='bold')
            ax.set_axis_off()
            ax.grid(False)
    if Title is not None:
        fig.suptitle(Title, fontsize=32, fontweight='bold')
    return fig


def get_fig_mesh_fusion(ghd, mean_ghd, std_ghd,
                        ghd_reconstruct, cvae,
                        meshfusion: MeshFusion, 
                        column=4, sub_size=5, connection_smoothing=False,
                        title_=None,
                        z=None, scale=None,
                        control=False,
                        spline=False,
                        return_meshes=False,
                        mesh_color='dodgerblue',
                        tangent_shift=[0, 0.075, 0],
                        extrusion=3,
                        c_transition=False):
    """
    use mesh fusion to create merged mesh
    """
    B = ghd.shape[0]
    data_fake = ghd_reconstruct.forward(ghd, mean_ghd, std_ghd, return_norm=True)
    ghd_normalized = ghd
    if mean_ghd is not None and std_ghd is not None:
        ghd = ghd * std_ghd.to(ghd.device) + mean_ghd.to(ghd.device)
    if control:
        z = torch.randn(1, cvae.latent_dim).expand(B, -1).to(ghd.device)
        cpcd_glo_gen, cpcd_tangent_glo_gen, tangent_from_ghd = cvae.generate_controlled(ghd, z=z, scale=scale, tangent_shift=tangent_shift)
    else:
        cpcd_glo_gen, cpcd_tangent_glo_gen, tangent_from_ghd = cvae.generate(ghd, z=z, scale=scale, tangent_shift=tangent_shift)
    merged_mesh_list = meshfusion.forward_mesh_fusion(ghd_normalized, cpcd_glo_gen, cpcd_tangent_glo_gen,
                                                      connection_smoothing=connection_smoothing,
                                                      scale=scale,
                                                      spline=spline,
                                                      extrusion=extrusion,
                                                      c_transition=c_transition)
    row = math.ceil(B/column)
    fig, axes = plt.subplots(row, column, subplot_kw={'projection': '3d'}, figsize=(column*sub_size, row*sub_size))
    plt.subplots_adjust(wspace=0, hspace=0)

    cpcd_color = ['red', 'black', 'blue']

    for i in range(row):
        for j in range(column):
            idx = i*column+j
            ax = axes[i, j]
            if idx < B:
                # plot merged mesh
                verts = merged_mesh_list[idx].verts_packed().detach().cpu().numpy()
                triangles = merged_mesh_list[idx].faces_packed().detach().cpu().numpy()
                ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=triangles, edgecolor=[[0, 0, 0]], linewidth=0.02, color=mesh_color, alpha=0.2)
                # plot cpcd
                cpcd = cpcd_glo_gen[idx]  # [num_branch, dpi, 3]
                tangent = cpcd_tangent_glo_gen[idx]  # [num_branch, dpi, 3]
                for branch in range(3):
                    cpcd_ = cpcd[branch].detach().cpu().numpy()
                    tangent_ = tangent[branch].detach().cpu().numpy()
                    # ax.plot(cpcd_[:, 0], cpcd_[:, 1], cpcd_[:, 2], color=cpcd_color[branch], linewidth=1.0, zorder=1)
                    # ax.scatter(cpcd_[::20, 0], cpcd_[::20, 1], cpcd_[::20, 2], color='yellow', s=1.25, zorder=2)
                    # plot tangent
                    start_point = cpcd_[0]
                    start_tangent = tangent_[0]
                    # ax.quiver(start_point[0], start_point[1], start_point[2], start_tangent[0], start_tangent[1], start_tangent[2], color='red', length=0.3, normalize=True)
                    # adjust box
                    mesh_center = verts.mean(axis=0)    
                    box_size = np.max(cpcd_ - mesh_center, axis=0).max(axis=0)
                    ax.set_xlim([mesh_center[0]-box_size, mesh_center[0]+box_size])
                    ax.set_ylim([mesh_center[1]-box_size, mesh_center[1]+box_size])
                    ax.set_zlim([mesh_center[2]-box_size, mesh_center[2]+box_size])
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_zlabel('')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_axis_off()
            # ax.set_title(f"{idx}", fontsize=16, fontweight='bold')
            ax.grid(False)
    fig.suptitle(title_, fontsize=32, fontweight='bold') if title_ is not None else None
    if not return_meshes:
        return fig
    else:
        return fig, merged_mesh_list


def get_mesh_fusion_dataset(ghd, mean_ghd, std_ghd,
                            cvae,
                            meshfusion: MeshFusion, 
                            z=None, scale=None,
                            spline=False,
                            tgt_dir=None,
                            control=True,
                            tangent_shift=[0, 0.05, 0],
                            extrusion=None,
                            c_transition=False):
    """
    use mesh fusion to create merged mesh
    """
    B = ghd.shape[0]
    ghd_normalized = ghd
    if mean_ghd is not None and std_ghd is not None:
        ghd = ghd * std_ghd.to(ghd.device) + mean_ghd.to(ghd.device)
    if z is None:
        z = 1.0 * torch.randn(B, cvae.latent_dim).to(ghd.device)
    if control:
        cpcd_glo_gen, cpcd_tangent_glo_gen, _ = cvae.generate_controlled(ghd, z=z, scale=scale, tangent_shift=tangent_shift)
    else:
        cpcd_glo_gen, cpcd_tangent_glo_gen, _ = cvae.generate(ghd, z=z, scale=scale, tangent_shift=tangent_shift)
    cpcd_glo_gen, cpcd_tangent_glo_gen, _ = cvae.generate(ghd, z=z, scale=scale, tangent_shift=tangent_shift)
    merged_mesh_list = meshfusion.forward_mesh_fusion(ghd_normalized, cpcd_glo_gen, cpcd_tangent_glo_gen,
                                                      connection_smoothing=True,
                                                      scale=scale,
                                                      spline=spline,
                                                      extrusion=extrusion,
                                                      c_transition=c_transition)
    for i, merged_mesh in enumerate(merged_mesh_list):
        checkpoint = {}
        checkpoint['ghd'] = ghd_normalized[i].detach().cpu().numpy()
        checkpoint['mean_ghd'] = mean_ghd.detach().cpu().numpy()
        checkpoint['std_ghd'] = std_ghd.detach().cpu().numpy()
        checkpoint['cpcd_z'] = z[i].detach().cpu().numpy()
        checkpoint['cpcd_scale'] = scale[i].detach().cpu().numpy()
        checkpoint['cpcd_glo_gen'] = cpcd_glo_gen[i].detach().cpu().numpy()
        checkpoint['norm_canonical'] = meshfusion.ghd_reconstruct.norm_canonical
        save_path = os.path.join(tgt_dir, generate_random_name())
        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, 'checkpoint.npy'), checkpoint)
        # de-normalize shape
        merged_mesh = merged_mesh.update_padded(merged_mesh.verts_padded() * meshfusion.ghd_reconstruct.norm_canonical)
        save_obj(os.path.join(save_path, 'shape.obj'), verts=merged_mesh.verts_packed(), faces=merged_mesh.faces_packed())
        pv_mesh = pv.read(os.path.join(save_path, 'shape.obj'))
        pv_mesh.save(os.path.join(save_path, 'shape.vtp'))
    return None