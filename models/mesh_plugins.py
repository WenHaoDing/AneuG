"""
2025.02.09
912
"""
import os
import torch
import numpy as np
import open3d as o3d
import pyvista as pv
from pytorch3d.structures import Meshes
import trimesh
import pickle
import pyvista as pv
from skeletor.utilities import make_trimesh
import igraph as ig
from tqdm import tqdm
from typing import List, Tuple, Union
from pytorch3d.ops import knn_points, laplacian
from pytorch3d.io import load_objs_as_meshes
from utils.utils import safe_load_mesh
import matplotlib.pyplot as plt
from models.ghd_reconstruct import GHD_Reconstruct
from models.pymeshlab_plugin import pymeshlab_smoothing
from pytorch3d.loss import mesh_laplacian_smoothing, mesh_normal_consistency
from scipy.stats import shapiro
from copy import deepcopy
import torch.nn.functional as F
from scipy.interpolate import CubicSpline
import math


class MeshPlugins(object):
    """
    This class takes in the canonical mesh and register necessary components, 
    primarily for centerline extraction and vessel tangent extraction.
    """
    def __init__(self, 
                 mesh: Meshes,              # Canonical mesh
                 cep_chk: str,              # Centerline End Point registration file path
                 max_loops=[15, 13, 12],    # Max loops for each branch (for wave casting-based centerline extraction)
                 loop_start=[11, 8, 8],     # Start index for loops (for tangent extraction)
                 loop_range=4,              # Loop range for loops (the number of rings used for tangent computation, 2 is enough)
                 trimmed_mesh_path=None,    # Trimmed canonical mesh path (used as mask for face trimming and opening detection)
                 wave_based_trimming=False, # Use wave casting-based face trimming if True, otherwise use trimmed_mesh_path
                 neck_chk_path=None):       # Neck nodes' indices (on the canonical mesh)
        # init canonical mesh
        self.mesh_p3d = mesh
        self.mesh_trimesh = p3d_to_trimesh(mesh)
        self.mesh_o3d= p3d_to_o3d(mesh)
        self.trimmed_mesh_path = trimmed_mesh_path
        # load cep registration
        self.cep_indices, self.num_branches = self.register_cep_points(cep_chk)
        # cast waves
        self.clp_loops, self.tripot_loops, self.cap_indices, self.cap_faces, self.trimmed_faces_wave = self.cast_waves(max_loops)
        # forward tangent extraction
        self.loop_start = loop_start
        self.loop_range = loop_range
        if trimmed_mesh_path is not None:
            self.trimmed_faces, self.trimmed_mesh_p3d = self.register_trimming(trimmed_mesh_path, wave_based_trimming)
            self.opening_idx_list = self.register_smooth_openings()  # List[List[int]]
        # neck registration
        self.neck_chk_path = neck_chk_path
        if self.neck_chk_path is not None:
            self.neck_idx, self.branch_faces, self.dome_faces, self.dome_verts_idx = self.register_neck()

    def register_cep_points(self, chk_path: str, visualize=False) -> Tuple[List[int], int]:
        """
        Load cep points' indices. User should find the cep points first and save their indices.
        """
        with open(chk_path, 'rb') as f:
            chk = pickle.load(f)
        cep_indices = chk['diff_cep_registration']
        num_branches = len(cep_indices)
        color = ['red', 'black', 'blue']
        if visualize:
            pv.set_jupyter_backend('html')
            pv.start_xvfb()
            p = pv.Plotter()
            p.add_mesh(self.mesh_trimesh, color='w', opacity=0.5)
            for cep_indice, color_ in zip(cep_indices, color):
                p.add_points(self.mesh_trimesh.vertices[cep_indice], color=color_, point_size=10)
            p.show()
        return cep_indices, num_branches
        
    def cast_waves(self, max_loops=[15, 9, 9], visualize=False, tripot_max_loops=[23, 18, 17]) -> Tuple[List[List[List[int]]], List[List[List[int]]], List[List[int]], List[List[int]], torch.Tensor]:
        """
        Cast waves from cep points and record loops.
        """
        wave_loops = _cast_waves(self.mesh_trimesh, self.cep_indices)
        points = []
        cap_indices = []
        clp_loops = [loop[:max_loops[i]] for i, loop in enumerate(wave_loops)]
        tripot_loops = [loop[tripot_max_loops[i]-4:tripot_max_loops[i]] for i, loop in enumerate(wave_loops)]
        for i in range(len(self.cep_indices)):
            wave_loop = wave_loops[i][:max_loops[i]]
            cap_indices_ = []
            for loop in wave_loop:
                points.append(self.mesh_trimesh.vertices[loop])
                cap_indices_ += loop.tolist()
            cap_indices.append(cap_indices_)
        points = np.concatenate(points)
        cap_faces = [find_faces(self.mesh_p3d.faces_packed(), cap) for cap in cap_indices]
        # find trimmed faces (wave-based)
        faces = self.mesh_p3d.faces_packed()
        combined_cap_faces = [face for sublist in cap_faces for face in sublist]
        mask = torch.ones(faces.shape[0], dtype=bool)
        mask[combined_cap_faces] = False
        trimmed_faces_wave = faces[mask]
        if visualize:
            pv.set_jupyter_backend('html')
            pv.start_xvfb()
            p = pv.Plotter()
            p.add_mesh(self.mesh_trimesh, color='w', opacity=0.5)
            p.add_points(points, color='r', point_size=10)
            p.show()
        return clp_loops, tripot_loops, cap_indices, cap_faces, trimmed_faces_wave

    def mesh_forward_tangents(self, meshes: Meshes, return_start_points=False):
        """
        Given a Meshes object, extract the following objects:
        centerline point cloud: List[Tensor[B, N, 3]]
        tangent pca: Tensor[B, num_branch, 3]
        tangent diff: Tensor[B, num_branch, 3]
        centerline start points: Tensor[B, num_branch, 3]
        """
        if return_start_points:
            cpcd_trinity, tangent_pca_trinity, tangent_diff_trinity, start_points = mesh_forward_tangents_(meshes, self.clp_loops, self.loop_start, self.loop_range, return_start_points)
            return cpcd_trinity, tangent_pca_trinity, tangent_diff_trinity, start_points
        else:
            cpcd_trinity, tangent_pca_trinity, tangent_diff_trinity = mesh_forward_tangents_(meshes, self.clp_loops, self.loop_start, self.loop_range, return_start_points)
            return cpcd_trinity, tangent_pca_trinity, tangent_diff_trinity

    def return_decapped_faces(self) -> torch.Tensor:
        faces = self.mesh_p3d.faces_packed()
        combined_cap_faces = [face for sublist in self.cap_faces for face in sublist]
        mask = torch.ones(faces.shape[0], dtype=bool)
        mask[combined_cap_faces] = False
        faces = faces[mask]
        return faces
    
    def register_trimming(self, trimmed_mesh_path: str=None, wave_based=False) -> Tuple[torch.Tensor, Meshes]:
        """
        Register masked canonical mesh's triangles ([N, 3]) using a manually trimmed canonical mesh.
        One could use wave-based trimming instead, which leads to uneven openings.
        """
        if not wave_based:
            trimmed_mesh = load_objs_as_meshes([trimmed_mesh_path])
            dists = knn_points(self.mesh_p3d.verts_padded(), trimmed_mesh.verts_padded(), K=1, return_nn=False)[0]
            verts2trim = torch.where(dists.view(-1) > 1e-5)[0]
            faces2trim = find_faces(self.mesh_p3d.faces_packed(), verts2trim)
            trimmed_faces = self.mesh_p3d.faces_packed()
            mask = torch.ones(trimmed_faces.shape[0], dtype=bool)
            mask[faces2trim] = False
            trimmed_faces = trimmed_faces[mask]
            mesh_p3d_trimed = Meshes(verts=[self.mesh_p3d.verts_packed()], faces=[trimmed_faces]).to(self.mesh_p3d.device)
        else:
            trimmed_faces = self.trimmed_faces_wave
            mesh_p3d_trimed = Meshes(verts=[self.mesh_p3d.verts_packed()], faces=[trimmed_faces]).to(self.mesh_p3d.device)
        return trimmed_faces, mesh_p3d_trimed

    def register_smooth_openings(self, threshold=3) -> List[torch.Tensor]:
        """
        Given the masked mesh, find the nodes' indices on its openings.
        threshold: the distance threshold for opening vertices and their corresponding centerline end points.
        """
        _, opening_vertex_indices = extract_opening_vertices(self.trimmed_mesh_p3d)
        opening_idx_list = []
        for cep_indice in self.cep_indices:
            cep = self.mesh_p3d.verts_packed()[cep_indice]
            dists = (self.mesh_p3d.verts_packed()[opening_vertex_indices] - cep.unsqueeze(0)).norm(dim=-1)
            opening_indices = torch.where(dists.view(-1) < threshold)[0]
            opening_idx_list.append(opening_vertex_indices[opening_indices])
        return opening_idx_list
    
    def register_neck(self):
        """
        We manually register the neck nodes' indices on the canonical mesh. User should find the neck nodes first and save their indices.
        """
        chk = torch.load(self.neck_chk_path)
        neck_idx = chk['neck_v_indices'][0]
        neck_idx = torch.Tensor(neck_idx).long()
        branch_faces, dome_faces = seperate_mesh(self.mesh_trimesh, neck_idx.cpu().numpy())
        dome_verts_idx = torch.tensor(dome_faces).view(-1).long().unique()
        return neck_idx, branch_faces, dome_faces, dome_verts_idx
    
    def visualize_wave_propagation(self, nt=10):
        p = pv.Plotter()
        pv.set_jupyter_backend('html')
        pv.start_xvfb()
        L = laplacian(verts=self.mesh_p3d.verts_packed(), edges=self.mesh_p3d.edges_packed())
        energy = torch.zeros(self.mesh_p3d.verts_packed().shape[0], 1, device=self.mesh_p3d.device)
        energy[self.cep_indices] = 1
        for i in range(nt):
            energy = L @ energy
            if i == nt - 2:
                energy_ = energy.clone()
        energy[energy != 0] = 1
        energy_[energy_ != 0] = 1
        energy = energy - energy_
        energy = energy.detach().cpu().numpy()
        # visualzation
        p = pv.Plotter()
        p.add_mesh(self.mesh_trimesh, scalars=energy, cmap="jet", show_edges=False, opacity=0.75)
        p.enable_eye_dome_lighting()  # Better depth perception
        p.add_light(pv.Light(position=(1, 1, 1), intensity=0.8, color='white'))
        p.show()
        return energy


class MeshFusion(object):
    """
    This class primarily takes in ghd and centerline point cloud (cpcd).
    self.forward_mesh_fusion reconstruct aneurysm complex meshes (masked), propagate the cross sections to form tubular vessel meshes, and merge them.
    One could use self.apply_spline_connect to perform spline connection, no visible improvement is observed for our case.
    One could use connection_smoothing to smooth the connection between tubular meshes, very small improvment is observed for our case.
    """
    def __init__(self, 
                 ghd_reconstruct: GHD_Reconstruct,
                 mean_ghd, std_ghd,
                 mesh_plugin: MeshPlugins,
                 device=torch.device("cuda:0")):
        self.ghd_reconstruct = ghd_reconstruct
        self.mean_ghd = mean_ghd
        self.std_ghd = std_ghd
        self.mesh_plugin = mesh_plugin
        self.device = device

    def apply_spline_connect(self, ghd, mean_ghd, std_ghd, scale=None, cpcd_glo=None, ds_dpi=20, num_anchor=4, forward_step=10):
        """
        Given cpcd, conduct spline to make sure smooth transition at vessel cross-sections of the aneurysm complex mesh.
        ds_dpi: downsampling DPI for spline.
        num_anchor: number of centerline points extracted from aneurysm mesh using the wave-based method, which ensures smooth direction change.
        forward_step: the index of the first point in the splined cpcd.
        """
        # Doesn't create visable improvement
        cpcd_glo_o = cpcd_glo
        cpcd_glo = cpcd_glo[..., ds_dpi::ds_dpi, :].detach().cpu()  # [B, num_branch, dpi, 3]
        if not torch.allclose(cpcd_glo_o[..., -1, :].detach().cpu(), cpcd_glo[..., -1, :]):
            cpcd_glo = torch.cat(
                [cpcd_glo, cpcd_glo_o[..., -1:, :].detach().cpu()],
                dim=-2)
        B, num_branch, dpi, C = cpcd_glo.shape
        # extract cpcd from aneurysm segments
        meshes = self.ghd_reconstruct.ghd_forward_as_Meshes(ghd, mean=mean_ghd, std=std_ghd, scale=scale)
        cpcd_trinity, _, _ = self.mesh_plugin.mesh_forward_tangents(meshes)  # List[Tensor[B, N, 3]]
        cpcd_trinity = [cpcd_trinity[i][:, self.mesh_plugin.loop_start[i]:self.mesh_plugin.loop_start[i]+num_anchor, :] for i in range(num_branch)]
        cpcd_trinity = torch.stack(cpcd_trinity, dim=1).detach().to(cpcd_glo.device)  # [B, num_branch, num_anchor, 3]
        cpcd_trinity = cpcd_trinity.flip(dims=[-2])  # flip the order
        cpcd_glo_fusion = torch.cat((cpcd_trinity, cpcd_glo), dim=2)  # [B, num_branch, dpi+num_anchor, 3]
        loc_x = compute_cumulative_distance(cpcd_glo_fusion).squeeze(-1)  # [B, num_branch, dpi+num_anchor]
        splined_cpcd_glo = np.zeros((B, num_branch, dpi * ds_dpi+forward_step, C))
        # condcut spline upon cpcd
        for i in tqdm(range(B), desc="Applying spline connecting"):
            for j in range(num_branch):
                cpcd_glo_fusion_ = cpcd_glo_fusion[i, j, :, :].numpy()
                x_original_ = loc_x[i, j, :].numpy()
                x_new_ = np.linspace(x_original_[num_anchor-1], x_original_[-1], ds_dpi*dpi+forward_step)
                for dim in range(C):
                    cs = CubicSpline(x_original_, cpcd_glo_fusion_[:, dim])
                    splined_cpcd_glo[i, j, :, dim] = cs(x_new_)
        splined_cpcd_glo = torch.tensor(splined_cpcd_glo, dtype=torch.float32).to(cpcd_glo.device)  # [B, num_branch, dpi*10, 3]
        splined_cpcd_glo_tangent = torch.cat([cpcd_trinity[..., -1:, :], splined_cpcd_glo], dim=2)
        splined_cpcd_glo_tangent = splined_cpcd_glo_tangent[:, :, 1:, :] - splined_cpcd_glo_tangent[:, :, :-1, :]
        splined_cpcd_glo_tangent = F.normalize(splined_cpcd_glo_tangent, dim=-1)
        splined_cpcd_glo = splined_cpcd_glo[:, :, forward_step:, :]
        splined_cpcd_glo_tangent = splined_cpcd_glo_tangent[:, :, forward_step:, :]
        return splined_cpcd_glo.to(cpcd_glo_o.device), splined_cpcd_glo_tangent.to(cpcd_glo_o.device)
        
    def forward_mesh_fusion(self, 
                            ghd: torch.Tensor,
                            cpcd_glo, cpcd_tangent_glo,
                            connection_smoothing=False,
                            scale=None,
                            spline=False,
                            extrusion=3,
                            c_transition=False
                            ) -> List[Meshes]:
        """
        Given cpcd, create tubular vessel meshes and merge them with aneurysm complex mesh reconstructed from ghd.
        cpcd_glo: Tensor [B, num_branch, dpi, 3]
        cpcd_tangent_glo: Tensor [B, num_branch, dpi, 3]
        cpcd_start_points: Tensor [B, num_branch, 3] (start points calcualted from rings (from aneurysm mesh))
        Vector tangent regulization ensures correct connection in most cases. To get absolute physiological connection, set spline_connect=True.
        """
        
        # reconstruct masked aneurysm complex meshes
        data = self.ghd_reconstruct.forward(ghd, self.mean_ghd, self.std_ghd, return_norm=False).to_data_list()
        verts = torch.stack([data_.pos for data_ in data], dim=0).to(self.device)
        B = verts.shape[0]
        if scale is not None:
            verts = verts * scale.unsqueeze(-1).to(verts.device)
        faces = self.mesh_plugin.trimmed_faces.unsqueeze(0).repeat(B, 1, 1).to(self.device)
        meshes = Meshes(verts=verts, faces=faces)

        # extract opening verts
        opening_verts_list = [meshes.verts_padded()[:, idx, :] for idx in self.mesh_plugin.opening_idx_list]  # List[Tensor[B, N, 3]]
        # apply spline connection if required
        if spline:
            cpcd_glo, cpcd_tangent_glo = self.apply_spline_connect(ghd, self.mean_ghd, self.std_ghd, scale, cpcd_glo)
        # add extrusion if required
        if extrusion is not None:
            print("applying extrusion on cpcd")
            extrusion = [e / self.ghd_reconstruct.norm_canonical for e in extrusion]
            cpcd_glo, cpcd_tangent_glo = extrude_cpcd_glo(cpcd_glo, cpcd_tangent_glo, extrusion=extrusion)
        # get l2w_trans (no transition included)
        l2w_trans = get_tubular_l2w_trans(cpcd_tangent_glo)  # [B, num_branch, dpi, 3, 3]
        
        # get tubular mesh verts List[Tensor[B, dpi, N, 3]]
        tube_verts_glo_list, sort_indices_list = get_tubular_mesh_verts(opening_verts_list, cpcd_glo, cpcd_tangent_glo, l2w_trans, c_transition=c_transition)
        opening_idx_list = [idx.to(self.device).unsqueeze(0).repeat(B, 1) for idx in self.mesh_plugin.opening_idx_list]
        opening_idx_sorted_list = [torch.gather(op_idx, dim=1, index=sort_indices) for op_idx, sort_indices in zip(opening_idx_list, sort_indices_list)]  # List[Tensor[B, N]]
        # get tubular mesh faces List[Tensor[2*N*(dpi-1), 3]]
        tube_faces_list, tube_verts_glo_list = get_tubular_mesh_faces(tube_verts_glo_list, ds_r=5)
        tube_faces_list = [faces.to(self.device) for faces in tube_faces_list]
        # merge meshes
        merged_mesh_list = merge_meshes(meshes, tube_faces_list, tube_verts_glo_list, opening_idx_sorted_list)
        # connection smoothing
        if connection_smoothing:
            merged_mesh_list = self.connection_smoothing(merged_mesh_list, opening_idx_sorted_list)
        return merged_mesh_list
    
    def connection_smoothing(self, merged_mesh_list: List, opening_idx_sorted_list, range_=4, stepsmoothnum=20):
        B = len(merged_mesh_list)
        smoothed_mesh_p3d_list = []
        for i, mesh_p3d in enumerate(merged_mesh_list):
            opening_idx = torch.cat([idx_[i] for idx_ in opening_idx_sorted_list])
            adjacent_verts_idx = get_adjacent_verts(mesh_p3d, opening_idx, range_=range_)
            adjacent_verts_idx = [idx.item() for idx in adjacent_verts_idx]
            smoothed_mesh_p3d = pymeshlab_smoothing(mesh_p3d, adjacent_verts_idx)
            smoothed_mesh_p3d_list.append(smoothed_mesh_p3d)
        return smoothed_mesh_p3d_list


class MeshRegulizer(object):  
    """
    This class provides regulization loss functions for generated meshes,
    primarily trumpet loss, which penalizes trumpet-like shapes at vessel cross-sections,
    and morphing energy alignment loss terms.
    """
    def __init__(self, 
                 mesh_plugin: MeshPlugins, 
                 loop_start: List=[11, 8, 8],
                 loop_range: int=4,
                 visualize=False,
                 device=torch.device("cuda:0"),
                 rigidloss=None):
        # init
        self.mesh_plugin = mesh_plugin
        self.device = device
        self.energy_source_idx = self.mesh_plugin.cep_indices
        self.loop_start = loop_start
        self.loop_range = loop_range
        # cast waves
        self.ring_lists = self.cast_waves_(visualize=visualize)
        # KL regulization
        self.rigidloss = rigidloss
        self.loss_rigid_real = None
        self.loss_laplacian_real = None
        self.loss_consistency_real = None

    def cast_waves_(self, visualize):
        ring_lists = cast_waves_laplacian(self.mesh_plugin.mesh_p3d, self.energy_source_idx, self.loop_start, self.loop_range, mesh_plugin=self.mesh_plugin, visualize=visualize)
        return ring_lists
        
    def trumpet_loss(self, meshes: Meshes):
        """
        Encourage same radius at regions close to vessel cross-sections on the aneurysm complex mesh.
        """
        loss_trumpet = torch.tensor(0.0).to(self.device)
        for i in range(len(self.ring_lists)):
            ring_list = self.ring_lists[i]
            radius_list = []
            for j in range(len(ring_list)):
                ring = ring_list[j]
                ring_verts = meshes.verts_padded()[:, ring, :]
                radius = torch.norm(ring_verts - ring_verts.mean(dim=1, keepdim=True), dim=-1).mean(dim=-1, keepdim=True)  # [B, 1]
                radius_list.append(radius)
            radius_list = torch.cat(radius_list, dim=-1)  # [B, num_ring]
            loss_trumpet += torch.nn.MSELoss()(radius_list, radius_list.mean(dim=-1, keepdim=True).repeat(1, radius_list.shape[1]))
        return loss_trumpet
    
    def KL_regulization(self, meshes: Meshes, real_meshes: Meshes, MEA=False, overreg=False):
        """
        Loss term for Morphing Energy Alignment (MEA).
        Loss will be directly applied on energies if set MEA=False or overreg=True, which lead to over-reg results with low diversity.
        """
        B_reg = meshes.verts_padded().shape[0]
        if self.loss_rigid_real is None or self.loss_laplacian_real is None or self.loss_consistency_real is None:
            print("Calculating regulization losses on real shapes")
            B = real_meshes.verts_padded().shape[0]
            loss_rigid_real = []
            loss_laplacian_real = []
            loss_consistency_real = []       
            for i in range(B):
                rigid_i = self.rigidloss.forward(real_meshes[i].verts_packed())
                loss_rigid_real.append(rigid_i) if not torch.isnan(rigid_i) else 0
                loss_laplacian_real.append(mesh_laplacian_smoothing(real_meshes[i], method="uniform"))
                loss_consistency_real.append(mesh_normal_consistency(real_meshes[i]))
            loss_rigid_real = torch.stack(loss_rigid_real)
            loss_laplacian_real = torch.stack(loss_laplacian_real)
            loss_consistency_real = torch.stack(loss_consistency_real)
            self.loss_rigid_real = loss_rigid_real
            self.loss_laplacian_real = loss_laplacian_real
            self.loss_consistency_real = loss_consistency_real
            _, sw_rigid = shapiro(loss_rigid_real.cpu().detach().numpy())
            _, sw_laplacian = shapiro(loss_laplacian_real.cpu().detach().numpy())
            _, sw_consistency = shapiro(loss_consistency_real.cpu().detach().numpy())
            print(f"Shapiro-Wilk test: [rigid: {sw_rigid}] [laplacian: {sw_laplacian}] [consistency: {sw_consistency}]")
        # regulization on generated shapes
        loss_rigid_fake = []
        loss_laplacian_fake = []
        loss_consistency_fake = [] 
        for i in range(B_reg):
            rigid_i = self.rigidloss.forward(meshes[i].verts_packed())
            loss_rigid_fake.append(rigid_i) if not torch.isnan(rigid_i) else 0
            loss_laplacian_fake.append(mesh_laplacian_smoothing(meshes[i], method="uniform"))
            loss_consistency_fake.append(mesh_normal_consistency(meshes[i]))
        if not overreg:
            loss_rigid = KL_regulization(self.loss_rigid_real, torch.stack(loss_rigid_fake))
        else:
            loss_rigid = 0.1 * (torch.stack(loss_rigid_fake).mean()).abs()
        if not MEA:
            if not overreg:
                loss_laplacian = mesh_laplacian_smoothing(meshes, method="uniform")
                loss_consistency = mesh_normal_consistency(meshes)
            else:
                loss_laplacian = 10 * mesh_laplacian_smoothing(meshes, method="uniform")
                loss_consistency = 10 * mesh_normal_consistency(meshes)
        else:
            loss_laplacian = KL_regulization(self.loss_laplacian_real, torch.stack(loss_laplacian_fake))
            loss_consistency = 2 * (torch.stack(loss_consistency_fake).mean() - self.loss_consistency_real.mean()).abs() \
                + 2 * (torch.stack(loss_consistency_fake).std() - self.loss_consistency_real.std()).clamp(min=0)
        # trumpet loss
        loss_trumpet = self.trumpet_loss(meshes)
        return loss_rigid, loss_laplacian, loss_consistency, loss_trumpet
    

def KL_regulization(loss_real, loss_fake):
    mu_fake, mu_real = loss_fake.mean(), loss_real.mean()
    var_fake, var_real = loss_fake.var(), loss_real.var()
    loss = 0.5 * torch.log(var_real/var_fake) + (var_fake + (mu_fake-mu_real)**2)/(2*var_real) - 0.5
    return loss

def get_adjacent_verts(mesh_p3d: Meshes, idx: torch.Tensor, range_: int) -> torch.Tensor:
    """
    Find adjacent vertices' indices of the given vertices indices idx.
    range_: searching range.
    """
    L = laplacian(verts=mesh_p3d.verts_packed(), edges=mesh_p3d.edges_packed())
    energy = torch.zeros(mesh_p3d.verts_packed().shape[0], 1, device=mesh_p3d.device)
    energy[idx.long()] = 1
    L_ = L
    for i in range(range_-1):
        L_ = L_ @ L
    propagated_energy = L_ @ energy
    adjacent_verts_idx = torch.where(propagated_energy > 0)[0]
    return adjacent_verts_idx

def cast_waves_laplacian(mesh_p3d: Meshes, idx: list, loop_start: List, loop_range: int, mesh_plugin=None, visualize=False):
    """
    A replacement for cast_waves function, same functionality.
    """
    if mesh_plugin is None:
        assert len(idx) == len(loop_start)
        L = laplacian(verts=mesh_p3d.verts_packed().detach(), edges=mesh_p3d.edges_packed().detach())
        ring_lists = []
        for i in range(len(idx)):
            energy = torch.zeros(mesh_p3d.verts_packed().shape[0], 1, device=mesh_p3d.device)
            energy[idx[i]] = 1
            energy_list = [energy]
            ring_list = []
            for j in range(loop_start[i]+loop_range):
                energy = L @ energy
                energy[energy != 0] = 1
                energy_list.append(energy)
                ring_list.append(torch.where(energy_list[j+1] - energy_list[j] !=0)[0])
            ring_lists.append(ring_list[loop_start[i]:])
    else:
        ring_lists = []
        for i in range(len(idx)):
            ring_lists.append(mesh_plugin.clp_loops[i][loop_start[i]:loop_start[i]+loop_range])
    if visualize:
        mesh = p3d_to_trimesh(mesh_p3d)
        trimmed_mesh_p3d = mesh_plugin.trimmed_mesh_p3d
        trimmed_mesh = p3d_to_trimesh(trimmed_mesh_p3d)
        pv.set_jupyter_backend('html')
        pv.start_xvfb()
        p = pv.Plotter()
        p.add_mesh(trimmed_mesh, color='b', show_edges=True)
        for i in range(len(ring_lists)):
            for j in range(len(ring_lists[i])):
                for k in range(len(ring_lists[i][j])):
                    p.add_points(mesh.vertices[ring_lists[i][j][k]], color='r', point_size=5)
        p.show()
    return ring_lists
    
def get_tubular_l2w_trans(cpcd_tangent_glo: torch.Tensor):
    """
    Given cpcd tangent vectors, return local to world transformation matrix.
    tangent: Tensor [B, num_branch, dpi, 3] -> l2w_trans: Tensor [B, num_branch, dpi, 3, 3]
    """
    # safe normalize the cpcd tangent
    cpcd_tangent_glo = cpcd_tangent_glo / cpcd_tangent_glo.norm(dim=-1, keepdim=True)
    num_branch, dpi = cpcd_tangent_glo.shape[1], cpcd_tangent_glo.shape[2]
    start_tangent =  cpcd_tangent_glo[..., 0, :]
    average_cross = torch.mean(torch.cross(start_tangent, torch.cat((start_tangent[:, 1:, :], start_tangent[:, :1, :]), dim=1), dim=-1), dim=1, keepdim=True)
    average_cross = average_cross / torch.norm(average_cross, dim=2, keepdim=True)
    average_cross = average_cross.unsqueeze(-2).repeat(1, num_branch, dpi, 1)   # [B, num_branch, dpi, 3]
    # get the loc2glo matrix loc_pcd @ l2w_trans = glo_pcd
    loc_sys_x = cpcd_tangent_glo  # [B, num_branch, dpi, 3]
    loc_sys_z = average_cross - loc_sys_x * torch.einsum('bvdc,bvdc->bvd', average_cross, loc_sys_x).unsqueeze(-1)
    loc_sys_y = torch.cross(loc_sys_z, loc_sys_x, dim=-1)  # [B, num_branch, dpi, 3]
    loc_sys_y = loc_sys_y / torch.norm(loc_sys_y, dim=-1, keepdim=True)
    l2w_trans = torch.stack((loc_sys_x, loc_sys_y, loc_sys_z), dim=-2)  # [B, num_branch, dpi, 3, 3]
    return l2w_trans

def get_tubular_mesh_verts(opening_pcd: List, cpcd_glo, cpcd_tangent_glo, l2w_trans, radius_map=True, c_transition=True):
    """
    Given point cloud on cross-section openings and cpcd, generate tubular mesh vertices.
    opening_pcd: List[Tensor[B, N, 3]]
    cpcd_glo: Tensor [B, num_branch, dpi, 3]
    TODO: break this function into smaller functions
    Journal: The trumpet problem has been fixed by using mapped radius for the cross-sections.
    """
    num_branch = len(opening_pcd)
    w2l_trans = l2w_trans.permute(0, 1, 2, 4, 3)  # [B, num_branch, dpi, 3, 3]
    B, num_branch, dpi, _ = cpcd_glo.shape
    pop_glo_list = []  # List[Tensor[B, N, 3]]  projected opening pcd global list
    tube_verts_glo_list = []
    sort_indices_list = []
    for i in range(num_branch):
        opening_pcd_ = opening_pcd[i]  # [B, N, 3]
        N = opening_pcd_.shape[1]
        centroid = cpcd_glo[:, i, :1, :]  # [B, 1, 3]
        tangent = cpcd_tangent_glo[:, i, :1, :]  # [B, 1, 3]
        D = -1 * (torch.sum(centroid * tangent, dim=-1))  # [B, 1]
        t = -1 * (torch.sum(tangent * opening_pcd_, dim=-1) + D) / torch.norm(tangent, dim=-1).pow(2)  # [B, N]
        pop_glo = opening_pcd_ + t.unsqueeze(-1) * tangent  # [B, N, 3]
        pop_glo_list.append(pop_glo)

        # get the radius using loc coordinates of the mapped 0-ring
        projected_pcd_loc = torch.einsum('bnc,bcl->bnl', pop_glo - centroid, w2l_trans[:, i, 0, :, :])  # [B, N, 3]
        # radius_set = torch.norm(projected_pcd_loc[..., 1:], dim=-1)  # [B, N]
        radius_set = torch.norm(opening_pcd_ - centroid, dim=-1)  # [B, N]
        if radius_map:
            rays = opening_pcd_ - centroid
            radius_set = torch.sqrt(radius_set.pow(2) - (tangent * rays).sum(dim=-1).pow(2))  # [B, N]

        # uncomment to use standard circle cross section
        # average_radius = radius_set.mean(dim=-1)  # [B]
        # radius_set = torch.ones_like(radius_set).to(radius_set.device) * average_radius.unsqueeze(-1)  # [B, N]

        # sort projected points, get indices and sorted angles and radius
        loc_angle = torch.atan2(projected_pcd_loc[..., -1], projected_pcd_loc[..., -2])  # [B, N]
        sort_indices = torch.argsort(loc_angle, dim=-1)  # [B, N]
        loc_angle_sorted = torch.gather(loc_angle, dim=-1, index=sort_indices)  # [B, N]
        radius_sorted = torch.gather(radius_set, dim=-1, index=sort_indices)  # [B, N]
        sort_indices_list.append(sort_indices)

        # generate tubluar verts in loc coordinate system
        tube_verts_loc_x = torch.zeros(B, dpi, N).to(opening_pcd_.device)
        if not c_transition:
            tube_verts_loc_y = (torch.cos(loc_angle_sorted) * radius_sorted).unsqueeze(1).repeat(1, dpi, 1)
            tube_verts_loc_z = (torch.sin(loc_angle_sorted) * radius_sorted).unsqueeze(1).repeat(1, dpi, 1)
        else:
            radius_average = torch.norm(radius_sorted, p=2, dim=-1, keepdim=True).unsqueeze(-1) /  math.sqrt(radius_sorted.shape[-1]) # [B, 1, 1]
            transit = torch.linspace(0, 1, dpi).unsqueeze(0).unsqueeze(0).to(radius_average.device)  # [1, 1, dpi]
            radius_sorted = radius_sorted.unsqueeze(-1).repeat(1, 1, dpi)  # [B, N, dpi]
            radius_transit = radius_sorted + (radius_average - radius_sorted) * transit
            tube_verts_loc_y = (torch.cos(loc_angle_sorted.unsqueeze(-1).repeat(1, 1, dpi)) * radius_transit).permute(0, 2, 1)
            tube_verts_loc_z = (torch.sin(loc_angle_sorted.unsqueeze(-1).repeat(1, 1, dpi)) * radius_transit).permute(0, 2, 1)

        tube_verts_loc = torch.stack((tube_verts_loc_x, tube_verts_loc_y, tube_verts_loc_z), dim=-1)  # [B, dpi, N, 3]
        tube_verts_glo = torch.einsum('bdnc,bdcl->bdnl', tube_verts_loc, l2w_trans[:, i, ...]) + cpcd_glo[:, i, :, :].unsqueeze(-2)  # [B, dpi, N, 3]
        tube_verts_glo_list.append(tube_verts_glo)
    return tube_verts_glo_list, sort_indices_list

def get_tubular_mesh_faces(tube_verts_glo_list: List[torch.Tensor], ds_r=2):
    """
    Given tubular mesh vertices, generate faces. -> faces_list: List[Tensor[2*N*(dpi-1), 3]]
    [B, dpi, N, 3]
    cylinder mesh structure:
    ._ ._ ._ . :0 0+num_vert_per_ring ...
    |/ |/ |/ | :1 1+num_vert_per_ring ...
    . _. _. _. :2 2+num_vert_per_ring ...
    """
    faces_list = []
    ds_tube_verts_glo_list = []
    for tube_verts_glo in tube_verts_glo_list:
        tube_verts_glo = tube_verts_glo[:, ::ds_r, ...]
        ds_tube_verts_glo_list.append(tube_verts_glo)
        B, dpi, N, _ = tube_verts_glo.shape

        faces_element = torch.arange((dpi - 1) * N).view(dpi - 1, N)  # [dpi-1, N]
        faces_1_1 = faces_element.view(-1, 1)  # first column of the cylinder mesh face part 1
        faces_1_2 = torch.cat((faces_element[:, 1:], faces_element[:, 0].unsqueeze(-1)), dim=-1).view(-1, 1)
        faces_1_3 = faces_1_1 + N
        faces_2_1 = faces_1_2  # first column of the cylinder mesh face part 2
        faces_2_2 = faces_2_1 + N
        faces_2_3 = faces_1_3
        faces = torch.cat((torch.cat((faces_1_1, faces_1_2, faces_1_3), dim=1), torch.cat((faces_2_1, faces_2_2, faces_2_3), dim=1)), dim=0)  # [2*N*(dpi-1), 3]
        faces_list.append(faces)
    return faces_list, ds_tube_verts_glo_list
        
def merge_meshes(meshes: Meshes, tube_faces_list, tube_verts_glo_list, opening_idx_sorted_list, as_list=True):
    """
    Merge aneurysm complex meshes with parent vessel meshesm, return meshes object if not as_list.
    v1.0
    opening_idx_sorted_list: List[Tensor[B, N]] sorted opening vert indices (for the original mesh)
    We want to do different dpi for different shapes, and therefore by defalut return as list.
    """
    num_branch = len(tube_faces_list)
    B = meshes.verts_padded().shape[0]
    if as_list:
        Meshes_list = []
        for nm in range(B):
            verts = meshes.verts_padded()[nm]
            faces = meshes.faces_padded()[nm]
            for i in range(num_branch):
                n_count = verts.shape[0]
                # cat
                tube_verts_glo = tube_verts_glo_list[i][nm]
                dpi, N, _ = tube_verts_glo.shape
                tube_verts_glo = tube_verts_glo.reshape(dpi*N, 3)
                verts = torch.cat((verts, tube_verts_glo), dim=0)
                faces = torch.cat((faces, tube_faces_list[i] + n_count), dim=0)
                # path faces
                opening_idx_sorted = opening_idx_sorted_list[i][nm]
                path_faces = get_face_patch(opening_idx_sorted, n_count)
                faces = torch.cat((faces, path_faces), dim=0)
            Meshes_list.append(Meshes(verts=[verts], faces=[faces]))
        return Meshes_list
    else:
        verts = meshes.verts_padded()
        faces = meshes.faces_padded()
        n_count = verts.shape[1]
        # for i in range(num_branch):
        return None

def get_face_patch(opening_idx_sorted, n_count):
    """
    Given sorted opening indices, return patch faces (patch triangles between generated tubular meshes and aneurysm complex meshes).
    We append these faces to the merged mesh and therefore merge them.
    opening_idx_sorted: Tensor [N]
    n_count: int, total number of aneurysm mesh nodes and tubular vessel mesh nodes, path meshes' indices start from n_count.
    """
    opening_idx_sorted = opening_idx_sorted.view(-1, 1)
    N = opening_idx_sorted.shape[0]
    faces_1_1 = opening_idx_sorted
    faces_1_2 = torch.cat((opening_idx_sorted[1:, :], opening_idx_sorted[:1, :]), dim=0).to(opening_idx_sorted.device)
    faces_1_3 = (torch.arange(N).view(-1, 1) + n_count).to(opening_idx_sorted.device)
    path_faces_1 = torch.cat((faces_1_1, faces_1_2, faces_1_3), dim=1)
    faces_2_1 = faces_1_2
    faces_2_2 = torch.cat((faces_1_3[1:, :], faces_1_3[:1, :]), dim=0)
    faces_2_3 = faces_1_3
    path_faces_2 = torch.cat((faces_2_1, faces_2_2, faces_2_3), dim=1)
    path_faces = torch.cat((path_faces_1, path_faces_2), dim=0)
    return path_faces

def extract_opening_vertices(mesh: Meshes):
    """
    Extracts the vertices associated with openings in the mesh.
    mesh (Meshes): A PyTorch3D Meshes object.
    torch.Tensor: A tensor of vertex coordinates that belong to openings.
    """
    if isinstance(mesh, trimesh.Trimesh):
        mesh = Meshes(verts=[torch.tensor(mesh.vertices, dtype=torch.float32)], 
                      faces=[torch.tensor(mesh.faces, dtype=torch.long)])
    # Get the faces and vertices of the mesh
    faces = mesh.faces_packed()  # (F, 3) face indices
    verts = mesh.verts_packed()  # (V, 3) vertex coordinates
    # Generate all edges for the faces
    edges = torch.cat([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]],
    ], dim=0)
    edges = torch.sort(edges, dim=1).values
    edges, counts = torch.unique(edges, return_counts=True, dim=0)
    boundary_edges = edges[counts == 1]
    opening_vertex_indices = torch.unique(boundary_edges.flatten())
    opening_verts = verts[opening_vertex_indices]
    return opening_verts, opening_vertex_indices
   
def p3d_to_trimesh(mesh: Meshes):
    vertices = mesh.verts_packed().detach().cpu().numpy()
    faces = mesh.faces_packed().detach().cpu().numpy()
    trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return trimesh_mesh

def p3d_to_o3d(mesh: Meshes) -> o3d.geometry.TriangleMesh:
    vertices = mesh.verts_packed().cpu().numpy()
    faces = mesh.faces_packed().cpu().numpy()
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
    return o3d_mesh

def find_faces(faces: torch.Tensor, cap_indices: List[int]):
    cap_tensor = torch.tensor(cap_indices, device=faces.device)
    # Check if any vertex in a face matches the cap indices
    mask = (faces.unsqueeze(-1) == cap_tensor).any(dim=-1).any(dim=-1)
    # Get the indices of the faces that satisfy the condition
    selected_face_indices = torch.nonzero(mask, as_tuple=True)[0].tolist()
    return selected_face_indices

def _cast_waves(mesh_trimesh, cep_indices, step_size=1, random_origin=False, progress=True):
    """
    Cast waves across mesh.
    -> 
    One can use this function or the laplacian version
    """
    if not random_origin:
        origins = cep_indices
    else:
        origins = None

    if not isinstance(origins, type(None)):
        if isinstance(origins, int):
            origins = [origins]
        elif not isinstance(origins, (set, list)):
            raise TypeError('`origins` must be vertex ID (int) or list '
                            f'thereof, got "{type(origins)}"')
        origins = np.asarray(origins).astype(int)
    else:
        origins = np.array([])

    # Wave must be a positive integer >= 1
    waves = int(len(origins)) if origins is not None else 1
    if waves < 1:
        raise ValueError('`waves` must be integer >= 1')

    # Same for step size
    step_size = int(step_size)
    if step_size < 1:
        raise ValueError('`step_size` must be integer >= 1')

    mesh = make_trimesh(mesh_trimesh, validate=False)
    # Generate Graph (must be undirected)
    G = ig.Graph(edges=mesh.edges_unique, directed=False)
    # G.es['weight'] = mesh.edges_unique_length

    # Prepare empty array to fill with centers
    centers = np.full((mesh.vertices.shape[0], 3, waves), fill_value=np.nan)
    radii = np.full((mesh.vertices.shape[0], waves), fill_value=np.nan)

    # Go over each connected component
    with tqdm(desc='Skeletonizing', total=len(G.vs), disable=not progress) as pbar:
        for cc in G.clusters():
            # Make a subgraph for this connected component
            SG = G.subgraph(cc)
            cc = np.array(cc)

            # Select seeds according to the number of waves
            n_waves = min(waves, len(cc))
            pot_seeds = np.arange(len(cc))
            np.random.seed(1984)  # make seeds predictable
            # See if we can use any origins
            if len(origins):
                # Get those origins in this cc
                in_cc = np.isin(origins, cc)
                if any(in_cc):
                    # Map origins into cc
                    cc_map = dict(zip(cc, np.arange(0, len(cc))))
                    seeds = np.array([cc_map[o] for o in origins[in_cc]])
                else:
                    seeds = np.array([])
                if len(seeds) < n_waves:
                    remaining_seeds = pot_seeds[~np.isin(pot_seeds, seeds)]
                    seeds = np.append(seeds,
                                        np.random.choice(remaining_seeds,
                                                        size=n_waves - len(seeds),
                                                        replace=False))
            else:
                seeds = np.random.choice(pot_seeds, size=n_waves, replace=False)
            seeds = seeds.astype(int)

            # Get the distance between the seeds and all other nodes
            dist = np.array(SG.shortest_paths(source=seeds, target=None, mode='all'))

            if step_size > 1:
                mx = dist.flatten()
                mx = mx[mx < float('inf')].max()
                dist = np.digitize(dist, bins=np.arange(0, mx, step_size))

            loops_list = []
            # Cast the desired number of waves
            for w in range(dist.shape[0]):
                loop_list = []
                this_wave = dist[w, :]
                # Collect groups
                mx = this_wave[this_wave < float('inf')].max()
                for i in range(0, int(mx) + 1):
                    this_dist = this_wave == i
                    ix = np.where(this_dist)[0]
                    SG2 = SG.subgraph(ix)
                    for cc2 in SG2.clusters():
                        this_verts = cc[ix[cc2]]
                        loop_list.append(this_verts)
                        # this_center = mesh.vertices[this_verts].mean(axis=0)
                        # this_radius = cdist(this_center.reshape(1, -1), mesh.vertices[this_verts])
                        # this_radius = rad_agg_func(this_radius)
                        # centers[this_verts, :, w] = this_center
                        # radii[this_verts, w] = this_radius
                loops_list.append(loop_list)
            pbar.update(len(cc))

    wave_loops = loops_list
    return wave_loops

def mesh_forward_tangents_(meshes: Meshes, wave_loops: list, loop_start=[8, 6, 6], loop_range=2, return_start_points=False, backward_steps=[3, 3, 3]):
    """
    backward_step: int, number of loops to go backward for getting the start points, set as small as possible.
    """
    cpcd_trinity = []  # centerline point cloud List[Tensor[B, N, 3]]
    tangent_pca_trinity = []  # tangent pca Tensor[B, num_branch, 3]
    tangent_diff_trinity = []  # tangent diff Tensor[B, num_branch, 3]
    start_points = []  # start points Tensor[B, num_branch, 3]

    verts = meshes.verts_padded()
    for wave_loop, loop_start_, backward_step in zip(wave_loops, loop_start, backward_steps):
        cpcd_branch = []
        for loop in wave_loop:
            indices = torch.tensor(loop).long()
            extracted_pcd = verts[:, indices, :]
            averaged_pcd = torch.mean(extracted_pcd, dim=1, keepdim=True)
            cpcd_branch.append(averaged_pcd)
        pcd = torch.cat(cpcd_branch, dim=1)
        cpcd_trinity.append(pcd)  # [B, N, 3]
        # get tangent vectors
        pcd_filtered = pcd[:, loop_start_:loop_start_+loop_range+1, :]
        start_points_ = pcd_filtered[:, 0, :].unsqueeze(1)  # [B, 1, 3]
        start_points_ = pcd[:, loop_start_-backward_step, :].unsqueeze(1)  # [B, 1, 3]
        start_points.append(start_points_)
        B = pcd_filtered.shape[0]
        tangent_diff_batch = []
        tangent_pca_batch = []
        for pcd_filtered_ in pcd_filtered:
            tangent_pca = torch.pca_lowrank(pcd_filtered_, q=1)[-1].permute(0, 1)
            tangent_diff = (pcd_filtered_[1:] - pcd_filtered_[:-1]).mean(dim=0, keepdim=True)
            tangent_diff = tangent_diff/tangent_diff.norm()
            tangent_pca = tangent_pca/tangent_pca.norm()
            tangent_pca = tangent_pca * torch.matmul(tangent_diff, tangent_pca).sign()
            tangent_diff_batch.append(tangent_diff)
            tangent_pca_batch.append(tangent_pca)
        tangent_diff_batch = torch.stack(tangent_diff_batch)
        tangent_pca_batch = torch.stack(tangent_pca_batch)
        tangent_diff_trinity.append(tangent_diff_batch)
        tangent_pca_trinity.append(tangent_pca_batch)
    tangent_diff_trinity = -1 * torch.cat(tangent_diff_trinity, dim=1)
    tangent_pca_trinity = -1 * torch.cat(tangent_pca_trinity, dim=1)
    start_points = torch.cat(start_points, dim=1)  # [B, num_branch, 3]
    if not return_start_points:
        return cpcd_trinity, tangent_pca_trinity, tangent_diff_trinity
    else:
        return cpcd_trinity, tangent_pca_trinity, tangent_diff_trinity, start_points

def tangent_forward_cartesian_axes(tangent_trinity):
    # tangent trinity: Tensor[B, num_branch, 3] -> Tensor[B, num_branch, 3(xyz), 3]
    tangent_trinity2 = torch.cat([tangent_trinity[:, 1:, :], tangent_trinity[:, 0, :].unsqueeze(-2)], dim=1)
    cartesian_cross = torch.cross(tangent_trinity, tangent_trinity2, dim=-1).mean(dim=-2, keepdim=True)  # [B, 1, 3]
    cartesian_z = map_vector_to_plane(cartesian_cross, tangent_trinity)  # [B, num_branch, 3]
    cartesian_x = tangent_trinity
    cartesian_y = torch.cross(cartesian_z, cartesian_x, dim=-1)  # [B, num_branch, 3]
    cartesian_axes = torch.stack([cartesian_x, cartesian_y, cartesian_z], dim=-2)
    return cartesian_axes

def map_vector_to_plane(vector: torch.Tensor, plane_vector: torch.Tensor):
    # vector: Tensor[B, 1, 3]; plane_vector: Tensor[B, num_branch, 3]
    # -> Tensor[B, num_branch, 3]
    # return mapped average cross vector for each branch
    vector = vector / vector.norm(dim=-1, keepdim=True)
    plane_vector = plane_vector / plane_vector.norm(dim=-1, keepdim=True)
    mapped_vector = vector - torch.matmul(vector, plane_vector.transpose(-1, -2)).transpose(-1, -2) * plane_vector
    return mapped_vector

def flood_fill(mesh: trimesh.Trimesh, seed_face, visited, neck_faces):
    """
    propagate mesh faces (for mesh seperation)
    """
    stack = [seed_face]
    region = set()
    while stack:
        current_face = stack.pop()
        if current_face in visited:
            continue
        visited.add(current_face)
        region.add(current_face)

        # Find neighboring faces
        neighbors = mesh.face_adjacency[mesh.face_adjacency[:, 0] == current_face, 1]
        neighbors = np.concatenate((neighbors, 
                                    mesh.face_adjacency[mesh.face_adjacency[:, 1] == current_face, 0]))
        for neighbor in neighbors:
            if neighbor not in visited and neighbor not in neck_faces:
                stack.append(neighbor)
    return region

def seperate_mesh(mesh: trimesh.Trimesh, neck_idx: np.ndarray):
    """
    This function splits one mesh into two parts given a loop of vertices
    Note: we can only seperate mesh for clean mesh (no redundant vertices)
    """
    neck_faces = []
    other_faces = set(range(len(mesh.faces)))
    # find neck faces
    for i, face in enumerate(mesh.faces):
        if any(vertex in neck_idx for vertex in face):
            neck_faces.append(i)
            other_faces.remove(i)
    visited = set()
    seed_face = next(iter(other_faces))
    region_1 = flood_fill(mesh, seed_face, visited, neck_faces)
    region_2 = other_faces - region_1
    # Assign neck faces to regions based on vertex membership
    region_1_verts = set(mesh.faces[list(region_1)].flatten())
    region_2_verts = set(mesh.faces[list(region_2)].flatten())
    for neck_face in neck_faces:
        neck_face_verts = set(mesh.faces[neck_face])
        if neck_face_verts & region_2_verts:  # If any vertex in region_2
            region_2.add(neck_face)
        else:  # Otherwise, assign to region_2
            region_1.add(neck_face)

    part_1_faces = mesh.faces[list(region_1)]
    part_2_faces = mesh.faces[list(region_2)]
    return part_1_faces, part_2_faces

class CondDistributionApprox(object):
    """
    This class approximates the distribution of aneurysm morphological markers.
    Note:
    It is difficult train decoder to interpret the conditions correctly on a small dataset.
    At the same time we do not wish to complicate things by training another boring VAE for conditions.
    Therefore, we simply expand the condition dataset using unconditional generation and MMC.
    synthetic conditions will be extracted from this dataset during conditional training.
    """
    def __init__(self,
                 ghd_dataset, 
                 ghd_reconstruct, dmm_calculator,
                 in_memory_size=1280,
                 batch_size=64,
                 uncond_net=None,
                 device=torch.device("cuda:0"),
                 apply_norm=True):
        # init
        self.ghd_dataset = ghd_dataset
        self.device = device
        self.apply_norm = apply_norm
        # unconditional net
        self.uncond_net = uncond_net
        # plugins
        self.ghd_reconstruct = ghd_reconstruct
        self.dmm_calculator = dmm_calculator
        self.cond_keys = dmm_calculator.cond_keys
        # get norm
        self.all_keys, self.mean_cond, self.std_cond = self.get_norm()
        # fill
        self.in_memory_size = in_memory_size
        self.batch_size = batch_size
        _ = self.fill()

    def fill(self):
        mean, std = self.ghd_dataset.get_mean_std()
        with torch.no_grad():
            z = torch.randn(self.in_memory_size, self.uncond_net.latent_dim).to(self.device)
            ghd_fake = self.uncond_net.decode(z)
            data_fake = self.ghd_reconstruct.forward(ghd_fake, mean.to(self.device), std.to(self.device))
            cond_fake = self.dmm_calculator.forward(data_fake)
        self.cond_fake = cond_fake.detach().cpu()
        self.ghd_fake = ghd_fake.detach().cpu()
        return None
    
    def get_norm(self):
        all_keys = ['AR', 'SR', 'LI', 'NW', 'V', 'Hmax', 'TA1', 'ImpD']
        dmm_calculator = deepcopy(self.dmm_calculator)
        dmm_calculator.cond_keys = all_keys
        with torch.no_grad():
            ghd_real = torch.stack(self.ghd_dataset.ghd).to(self.device)
            data_real = self.ghd_reconstruct.forward(ghd_real, mean=None, std=None)
            cond_real = dmm_calculator.forward(data_real)
            mean_cond, std_cond = cond_real.mean(dim=0, keepdim=True), cond_real.std(dim=0, keepdim=True)
        return all_keys, mean_cond, std_cond
    
    def normalize(self, cond, cond_keys=None, denormalize=False):
        if cond_keys is None:
            cond_keys = self.cond_keys
        idx = [self.all_keys.index(key) for key in cond_keys]
        mean_cond, std_cond = self.mean_cond[:, idx], self.std_cond[:, idx]
        mean_cond, std_cond = mean_cond.to(cond.device), std_cond.to(cond.device)

        if self.apply_norm:
            if not denormalize:
                return (cond - mean_cond) / std_cond
            else:
                return cond * std_cond + mean_cond
        else:
            return cond

    def get_batch(self, normalize=False):
        idx = torch.randint(0, self.in_memory_size, (self.batch_size,))
        cond_fake = self.cond_fake[idx].to(self.device)
        if normalize:
            cond_fake = self.normalize(cond_fake)
        else:
            pass
        return cond_fake
    
    def cond_loss_forward(self, cond_input, cond_recall, cond_keys=None, normalize_recall=True, normalize_input=False,
                          generator=None, drop_dmm=False):
        if cond_keys is None:
            cond_keys = self.cond_keys
        if normalize_recall:
            cond_recall = self.normalize(cond_recall, cond_keys)
        if normalize_input:
            cond_input = self.normalize(cond_input, cond_keys)
        if not drop_dmm:
            cond_loss = F.mse_loss(cond_input, cond_recall)
        else:
            mean, std = self.ghd_dataset.get_mean_std()
            idx = torch.randint(0, self.in_memory_size, (self.batch_size,))
            cond_fake = self.cond_fake[idx].to(self.device)
            ghd_fake = self.ghd_fake[idx].to(self.device)
            data_fake = self.ghd_reconstruct.forward(ghd_fake, mean=mean.to(self.device), std=std.to(self.device))
            ghd_recon, mu, logvar = generator(ghd_fake, self.normalize(cond_fake, cond_keys))
            data_recon = self.ghd_reconstruct.forward(ghd_recon, mean, std, return_norm=True)
            vert_loss = F.mse_loss(data_recon.pos, data_fake.pos)
            norm_loss = F.mse_loss(data_recon.x[:, 3:], data_fake.x[:, 3:])
            mse_loss = F.mse_loss(ghd_fake, ghd_recon)
            kl_loss = KL_divergence(mu, logvar)
            cond_loss = 1 * mse_loss + 100 * vert_loss + 10 * norm_loss + 0.0002 * kl_loss
            cond_recall = self.dmm_calculator.forward(data_recon)
            cond_loss_recall = F.mse_loss(cond_fake, cond_recall)
            print("recall loss: ", cond_loss_recall)
            # print(f"mse: {mse_loss} vert: {vert_loss} norm: {norm_loss} kl: {kl_loss}")
            cond_loss = 0.1 * cond_loss
        return cond_loss


def KL_divergence(mu, logvar):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD

def columnwise_relative_l2_loss(predictions, targets):
    l2_diff = torch.norm(predictions - targets, p=2, dim=0)
    l2_target = torch.norm(targets, p=2, dim=0) + 1e-8
    # Compute relative L2 loss for each column
    relative_loss_per_column = l2_diff / l2_target
    # Return the mean relative loss across all columns
    return relative_loss_per_column.mean()

def compute_cumulative_distance(cpcd):
    """
    Given cpcd, return cumulative distance for each point.
    input: [B, num_branch, num_points, 3]。
    return: [B, num_branch, num_points, 1]。
    """
    diff = cpcd[:, :, 1:, :] - cpcd[:, :, :-1, :]  # [B, num_branch, num_points-1, 3]
    distances = torch.norm(diff, dim=-1, keepdim=True)  # [B, num_branch, num_points-1, 1]
    cumulative_distances = torch.zeros(*cpcd.shape[:3], 1, device=cpcd.device)  # [B, num_branch, num_points, 1]
    cumulative_distances[:, :, 1:, :] = torch.cumsum(distances, dim=2)  # [B, num_branch, num_points, 1]
    return cumulative_distances


def extrude_cpcd_glo(cpcd_glo: torch.Tensor, cpcd_tangent_glo: torch.Tensor, extrusion: list):
    """
    cpcd_glo: Tensor [B, num_branch, dpi, 3]
    """
    steps = round(max(extrusion)) * 6
    tangent = cpcd_glo[:, :, -1, :] - cpcd_glo[:, :, -2, :]
    tangent = tangent.unsqueeze(-2)
    tangent = F.normalize(tangent, dim=-1)
    extrusion = torch.tensor(extrusion, device=cpcd_glo.device).view(1, -1, 1, 1)  # Shape: [1, 3, 1, 1]

    extruded_points = cpcd_glo[:, :, -1, :].unsqueeze(-2) + tangent * torch.linspace(0, 1, steps=steps, device=cpcd_glo.device).view(1, 1, -1, 1) * extrusion
    cpcd_glo = torch.cat((cpcd_glo, extruded_points), dim=2)
    cocd_tangent_glo = torch.cat((cpcd_tangent_glo, tangent.repeat(1, 1, steps, 1)), dim=2)
    return cpcd_glo, cocd_tangent_glo







    

    


    
        

        
    