"""
Differentiable Morphological Marker Calculator
"""
import torch
from torch_geometric.data import Data, Batch
from pytorch3d.structures import Meshes
from pytorch3d.ops import cot_laplacian
from ops.mesh_geometry2 import vert_feature_packed_padded, get_dual_area_vertex_packed, get_gaussian_curvature_vertices_from_face_packed, get_mean_curvature_vertices_packed


class DiffMorphoMarkerCalculator(object):
    def __init__(self, mesh_plugin, cond_keys=['AR', 'SR', 'LI'], device=torch.device("cuda:0")):
        self.mesh_plugin = mesh_plugin
        self.device = device

        self.mesh_p3d = self.mesh_plugin.mesh_p3d.to(device)
        self.branch_faces, self.dome_faces = self.mesh_plugin.branch_faces, self.mesh_plugin.dome_faces
        self.branch_vidx, self.dome_vidx = self.get_verts_idx_from_faces(self.branch_faces), self.get_verts_idx_from_faces(self.dome_faces)
        self.neck_vidx = self.mesh_plugin.neck_idx

        self.cond_keys = cond_keys
        self.cond_dim = len(self.cond_keys)

    def process_input(self, input):
        if isinstance(input, Batch):
            Meshes_ = self.Batch2Meshes(input)
        elif isinstance(input, Meshes):
            Meshes_ = input
        else:
            raise ValueError("input should be either Batch or Meshes")
        return Meshes_

    def forward_AR(self, input):
        Meshes_ = self.process_input(input)
        NW = self.get_NW(Meshes_)
        AD = self.get_AD(Meshes_)
        AR = AD / NW
        return AR
    
    def forward_Hmax(self, input):
        Meshes_ = self.process_input(input)
        Hmax = self.get_Hmax(Meshes_)
        return Hmax
    
    def forward_SR(self, input):
        Meshes_ = self.process_input(input)
        Hmax = self.get_Hmax(Meshes_)
        Dvessel = self.get_Dvessel(Meshes_)
        SR = Hmax / Dvessel
        return SR
    
    def forward_V(self, input):
        Meshes_ = self.process_input(input)
        V = self.get_V(Meshes_)
        return 15*V
    
    def forward_LI(self, input):
        Meshes_ = self.process_input(input)
        V = self.get_V(Meshes_)
        SA = self.get_SA(Meshes_)
        LI = 0.1 * SA / V
        return LI
    
    def forward_NGCI(self, input):
        """Negative Gaussian Curvature integration"""
        Meshes_ = self.process_input(input)
        dual_areas_padded = vert_feature_packed_padded(Meshes_, get_dual_area_vertex_packed(Meshes_).view(-1,1))
        gc_vertex_packed = get_gaussian_curvature_vertices_from_face_packed(Meshes_).view(-1,1)
        gc_integration = vert_feature_packed_padded(Meshes_, gc_vertex_packed.view(-1,1)) * dual_areas_padded
        gc_integration = gc_integration[:, self.mesh_plugin.dome_verts_idx]
        NGCI = torch.sum(torch.clamp(gc_integration, max=0), dim=1)
        NGCI = -0.01 * NGCI
        if torch.isnan(NGCI).any():
            NGCI[torch.isnan(NGCI)] = 0.0
        return NGCI
    
    def forward_MC(self, input):
        Meshes_ = self.process_input(input)
        MC = self.get_MC(Meshes_)
        return MC

    def forward_NW(self, input):
        Meshes_ = self.process_input(input)
        NW = self.get_NW(Meshes_)
        return NW
    
    def forward_TripotAngles(self, input):
        Meshes_ = self.process_input(input)
        TripotAngles = self.get_TripotAngles(Meshes_)
        for i, angle in enumerate(TripotAngles):
            if torch.isnan(angle).any():
                print(f"***********************Warning: NaN values in TripotAngles[{i}]***********************")
        TripotAngles = tuple(torch.where(torch.isnan(angle), torch.zeros_like(angle).to(Meshes_.device), angle) for angle in TripotAngles)
        return TripotAngles

    def forward(self, input):
        cond_dict = {}
        if 'AR' in self.cond_keys:
            cond_dict['AR'] = self.forward_AR(input)
        if 'SR' in self.cond_keys:
            cond_dict['SR'] = self.forward_SR(input)
        if 'LI' in self.cond_keys:
            cond_dict['LI'] = self.forward_LI(input)
        if 'MC' in self.cond_keys:
            cond_dict['MC'] = self.forward_MC(input)
        if 'NGCI' in self.cond_keys:
            cond_dict['NGCI'] = self.forward_NGCI(input)
        if 'NW' in self.cond_keys:
            cond_dict['NW'] = self.forward_NW(input)
        if 'V' in self.cond_keys:
            cond_dict['V'] = self.forward_V(input)
        if 'Hmax' in self.cond_keys:
            cond_dict['Hmax'] = self.forward_Hmax(input)
        if any(key in self.cond_keys for key in ['TA1', 'TA2', 'TA3', 'ImpD']):
            TA1, TA2, TA3, ImpD = self.forward_TripotAngles(input)
            pi = 3.1415926
            TA1, TA2, TA3 = TA1 / pi, TA2 / pi, TA3 / pi
            # ImpD = ImpD * 10
            # TA1 = TA1 * 5
            for key in ['TA1', 'TA2', 'TA3', 'ImpD']:
                if key in self.cond_keys:
                    cond_dict[key] = locals()[key]

        cond = torch.cat([cond_dict[key] for key in self.cond_keys], dim=-1)
        return cond

    def Batch2Meshes(self, batch: Batch):
        data_list = batch.to_data_list()
        B = len(data_list)
        verts = [data.pos.to(self.device) for data in data_list]
        faces = [self.mesh_p3d.faces_packed() for data in data_list]
        Meshes_ = Meshes(verts, faces)
        return Meshes_
    
    def get_verts_idx_from_faces(self, faces):
        faces = torch.Tensor(faces).long().to(self.device)
        verts_idx = faces.detach().flatten().unique().long().to(self.device)
        return verts_idx
    
    def get_NW(self, Meshes_: Meshes):
        """Neck Width"""
        neck_verts = Meshes_.verts_padded()[:, self.neck_vidx, :]  # [B, N, 3]
        centroid = neck_verts.mean(dim=1, keepdim=True)  # [B, 1, 3]
        NW = 2.0 * (neck_verts - centroid).norm(dim=-1).mean(dim=1, keepdim=True)  # [B, 1]
        return NW
    
    def get_AD(self, Meshes_: Meshes):
        """
        Aneurysm Depth:
        Maximum distance from dome to neck, projected to the normal direction of the neck plane.
        """
        neck_verts = Meshes_.verts_padded()[:, self.neck_vidx, :]  # [B, N, 3]
        centroid = neck_verts.mean(dim=1, keepdim=True)  # [B, 1, 3]
        rays = neck_verts - centroid  # [B, N, 3]
        N = rays.shape[1]
        shift = round(N / 4)
        normal = torch.cross(rays[:, :-shift, :], rays[:, shift:, :], dim=-1).mean(dim=1, keepdim=True)  # [B, 1, 3]
        normal = normal / normal.norm(dim=-1, keepdim=True)

        dome_verts = Meshes_.verts_padded()[:, self.dome_vidx, :]  # [B, M, 3]
        AD = ((dome_verts - centroid) * normal).sum(dim=-1).abs().max(dim=1, keepdim=True)[0] # [B]
        return AD

    def get_Hmax(self, Meshes_: Meshes):
        """
        Hmax: Maximum height of the dome
        Maximum absolute distance from the dome the centroid of the neck plane.
        """
        neck_verts = Meshes_.verts_padded()[:, self.neck_vidx, :]  # [B, N, 3]
        centroid = neck_verts.mean(dim=1, keepdim=True)  # [B, 1, 3]
        Hmax = (neck_verts - centroid).norm(dim=-1).max(dim=1, keepdim=True)[0]
        return Hmax
    
    def get_Dvessel(self, Meshes_: Meshes):
        clp_loops = self.mesh_plugin.clp_loops
        # loop_start = self.mesh_plugin.loop_start
        loop_start = [7, 5, 5]
        num_branch = len(clp_loops)
        ring_diameter_list = []
        for i in range(num_branch):
            ring_diameter = []
            for j in range(loop_start[i], loop_start[i]+4):
                ring_verts = Meshes_.verts_padded()[:, clp_loops[i][j], :]  # [B, N, 3]
                centroid = ring_verts.mean(dim=1, keepdim=True)
                diameter = (ring_verts - centroid).norm(dim=-1).max(dim=1, keepdim=True)[0] * 2  # [B, 1]
                ring_diameter.append(diameter)
            ring_diameter_list.append(torch.cat(ring_diameter, dim=-1))  # [B, 4]
        Dvessel = torch.stack(ring_diameter_list, dim=1)  # [B, num_branch, 4]
        Dvessel = Dvessel.mean(dim=-1).mean(dim=-1, keepdim=True)  # [B]
        return Dvessel
    
    def get_V(self, Meshes_: Meshes):
        volume = get_gaussian_volume(Meshes_)
        return volume
    
    def get_SA(self, Meshes_: Meshes):
        surface_areas = get_surface_areas(Meshes_)
        return surface_areas
    
    def get_MC(self, Meshes_: Meshes):
        B = Meshes_.verts_padded().shape[0]
        N = Meshes_.verts_padded().shape[1]
        mean_curvatures = []
        for i in range(B):
            verts = Meshes_.verts_padded()[i]  # [N, 3]
            faces = Meshes_.faces_padded()[i]  # [F, 3]
            laplacian, inverse_A = cot_laplacian(verts, faces)
            distances = verts.unsqueeze(1) - verts.unsqueeze(0)  # [N, N, 3]
            mean_curvature = (laplacian.to_dense().unsqueeze(-1) * distances).sum(dim=1).norm(dim=-1, keepdim=True) * 2 * inverse_A
            mean_curvatures.append(mean_curvature)
        mean_curvatures = torch.stack(mean_curvatures, dim=0)  # [B, N, 1]
        mask = torch.ones_like(mean_curvatures, dtype=torch.bool)
        mask[:, self.dome_vidx] = False
        mean_curvatures[mask] = 0.0  # [B, N, 1]
        mean_curvatures = mean_curvatures.mean(dim=1)  # [B, 1]
        return mean_curvatures
    
    def get_TripotAngles(self, Meshes_: Meshes):
        """
        ↑   ↑      <-- Tangent vectors (outlets)
        \   /
         \o/    <-- Aneurysm (bulge at bifurcation)
          | ↑    <-- Normal vector (at pouch neck)
          |      
          ↑      <-- Tangent vector (inlet)
        We calculate the angle between tangent vectors of aneurysm inlet, outlets (superior and inferior), and normal vector at the pouch neck:
        [\theta(inlet, n), \theta(inlet, ioutlet), \theta(n, soutlet)] (*pi)
        """
        # get neck normal
        neck_verts = Meshes_.verts_padded()[:, self.neck_vidx, :]  # [B, N, 3]
        centroid = neck_verts.mean(dim=1, keepdim=True)  # [B, 1, 3]
        rays = neck_verts - centroid  # [B, N, 3]
        N = rays.shape[1]
        shift = round(N / 4)
        normal = torch.cross(rays[:, :-shift, :], rays[:, shift:, :], dim=-1).mean(dim=1, keepdim=False)  # [B, 3]
        normal = normal / normal.norm(dim=-1, keepdim=True)
        # get tangent vectors
        tripot_loops = self.mesh_plugin.tripot_loops
        tripot_tangents = []
        tripot_centroids = []
        for i in range(len(tripot_loops)):
            tripot_cpcd = []
            for j in range(len(tripot_loops[i])):
                cpcd = Meshes_.verts_padded()[:, tripot_loops[i][j], :].mean(dim=1, keepdim=True)
                tripot_cpcd.append(cpcd)
            tripot_cpcd = torch.cat(tripot_cpcd, dim=1)  # [B, 4, 3]
            tripot_centroids.append(tripot_cpcd.mean(dim=1, keepdim=True))  # [B, 1, 3]
            tangent = (tripot_cpcd[:, 1:, :] - tripot_cpcd[:, :-1, :]).mean(dim=1, keepdim=False)  # [B, 3]
            tangent = tangent / tangent.norm(dim=-1, keepdim=True)
            tripot_tangents.append(tangent)

        # calculate angles
        angle_inlet_n = torch.acos(cosin_clamp((tripot_tangents[0] * normal).sum(dim=-1, keepdim=True).abs()))  # [B, 1]
        angle_inlet_ioutlet = torch.acos(cosin_clamp((tripot_tangents[0] * tripot_tangents[1]).sum(dim=-1, keepdim=True).abs()))  # [B, 1]
        angle_inlet_soutlet = torch.acos(cosin_clamp((tripot_tangents[0] * tripot_tangents[2]).sum(dim=-1, keepdim=True).abs()))  # [B, 1]

        # calculate flow impigenement distance
        imp_distance = torch.norm(tripot_centroids[0] - centroid, dim=-1, keepdim=False)  # [B, 1]
        inlet_neck_v = (tripot_centroids[0] - centroid).squeeze(1)  # [B, 3]
        inlet_neck_v = inlet_neck_v / inlet_neck_v.norm(dim=-1, keepdim=True)
        imp_theta = torch.acos(cosin_clamp((inlet_neck_v * tripot_tangents[0]).sum(dim=-1, keepdim=True).abs()))
        imp_distance = imp_distance * torch.sin(imp_theta)  # [B, 1]

        # TripotAngles = torch.cat([angle_inlet_n, angle_inlet_ioutlet, angle_inlet_soutlet], dim=-1)  # [B, 3] 
        return angle_inlet_n, angle_inlet_ioutlet, angle_inlet_soutlet, imp_distance
 
    def get_dome_meshes(self, Meshes_: Meshes):
        neck_vidx = self.neck_vidx.to(Meshes_.device)
        dome_vidx = self.dome_vidx
        dome_faces = torch.Tensor(self.dome_faces).long().to(Meshes_.device)

        neck_centroids = Meshes_.verts_padded()[:, neck_vidx, :].mean(dim=1, keepdim=True)  # [B, 1, 3]
        B, N, _ = Meshes_.verts_padded().shape
        dome_verts = torch.cat((Meshes_.verts_padded(), neck_centroids), dim=1)  # [B, M+1, 3]
        # get face patches
        patch_faces = torch.stack((neck_vidx[:-1], neck_vidx[1:], torch.Tensor([N]).to(Meshes_.device).repeat(len(neck_vidx)-1)), dim=1)
        patch_faces2 = torch.stack((neck_vidx[-1:], neck_vidx[:1], torch.Tensor([N]).to(Meshes_.device)), dim=1)
        dome_faces = torch.cat((dome_faces, patch_faces, patch_faces2), dim=0).unsqueeze(0).repeat(B, 1, 1)  # [B, F', 3]
        dome_Meshes = Meshes(verts=dome_verts, faces=dome_faces)
        return dome_Meshes


def get_gaussian_volume(meshes: Meshes):
    """
    Given Meshes object, calcualte the aneurysm dome volume using Gaussian Theorem.
    V = 1/3 * sum_{F} (n_f * c_f) * A_f
    volume = posision divergence integration over the 3D geometry = surface intergration of dot product of normal and centroid position
    """
    B, F = meshes.faces_padded().shape[0], meshes.faces_padded().shape[1]
    verts = meshes.verts_padded()  # [B, N, 3]
    faces = meshes.faces_padded()  # [B, F, 3]
    faces_areas = meshes.faces_areas_packed().view(B, -1)  # [B, F]
    faces_normals = meshes.faces_normals_padded()  # [B, F, 3]
    # faces_centroids = torch.gather(verts.unsqueeze(1).expand(-1, F, -1, -1), dim=2, 
    #                                index=faces.unsqueeze(-1).expand(-1, -1, -1, 3)).mean(dim=2)  # [B, F, 3, 3(xyz)] -> [B, F, 3]

    batch_indices = torch.arange(B, device=verts.device).view(-1, 1, 1)  # Shape [B, 1, 1]
    faces_centroids = verts[batch_indices, faces].mean(dim=2)

    # faces_centroids = verts[torch.arange(B).unsqueeze(-1), faces].mean(dim=2)
    volume = ((faces_normals * faces_centroids).sum(dim=-1) * faces_areas).sum(dim=-1, keepdim=True) / 3  # [B, 1]
    return volume
    
def get_surface_areas(meshes: Meshes):
    """
    Given Meshes object, calcualte the surface areas.
    """
    B = meshes.verts_padded().shape[0]
    surface_areas = meshes.faces_areas_packed().view(B, -1).sum(dim=-1, keepdim=True)  # [B, 1]
    return surface_areas

def cosin_clamp(x, disable=False):
    if disable:
        return x
    else:
        return torch.clamp(x, min=-0.999, max=0.999)