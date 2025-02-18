import sys
sys.path.append('/media/yaplab/"HDD Storage"/wehao/ghb_aneurysms')
import os
import torch
from torch.utils.data import Dataset

from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes, join_meshes_as_batch
from pytorch3d.io import load_objs_as_meshes
from new_version.ops.mesh_geometry import Winding_Occupancy
from tqdm import tqdm
from models.discriminators import GHD_Reconstruct
import pyvista as pv
import trimesh
from pytorch3d.ops import knn_points
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tqdm import tqdm
from torch.utils.data import DataLoader


class OccupancyDataset(Dataset):
    """
    We gather closed meshes (ground truth) and calculate occupancy field for each mesh.
    """
    def __init__(self, shape_dir: str, 
                 load_path: str, re_solve=False,
                 surface_sample: int=2048, occupancy_sample=12800,
                 ghd_reconstruct: GHD_Reconstruct=None,
                 device=torch.device("cuda:0"),
                 sdf=False,
                 return_normals=False
                 ):
        self.shape_dir = shape_dir
        self.load_path = load_path
        self.re_solve = re_solve
        self.surface_sample = surface_sample
        self.occupancy_sample = occupancy_sample
        self.sdf = sdf
        self.return_normals = return_normals

        # plugins 
        self.ghd_reconstruct = ghd_reconstruct

        self.device = device

        if not os.path.exists(self.load_path) or re_solve:
            self.data = self.solve_dataset()
        else:
            self.data = torch.load(self.load_path, map_location=torch.device('cpu'))

    def solve_dataset(self):
        with torch.no_grad():
            # process batch_size points at a time to save memory
            batch_size = 1200
            shape_list = [dir for dir in os.listdir(self.shape_dir) if os.path.isdir(os.path.join(self.shape_dir, dir)) and 'canonical' not in dir]
            mesh_path_list = [f'{self.shape_dir}/{shape}/part_aligned_updated.obj' for shape in shape_list]

            surface_points_list = []
            surface_normals_list = []
            occupancy_points_list = []
            occupancy_mask_list = []

            for mesh_path in tqdm(mesh_path_list):
                # load mesh & normalize (to keep same normalization with ghd)
                mesh = load_objs_as_meshes([mesh_path]).to(self.device)
                mesh = Meshes(verts=[mesh.verts_packed() / self.ghd_reconstruct.norm_canonical], faces=[mesh.faces_packed()])

                # surface points
                surface_points, surface_normals = sample_points_from_meshes(mesh, self.surface_sample, return_normals=True)
                # occupancy points
                min_bounds = mesh.verts_packed().min(dim=0)[0]
                max_bounds = mesh.verts_packed().max(dim=0)[0]
                expansion = 0.05 * (max_bounds - min_bounds)
                min_bounds -= expansion
                max_bounds += expansion
                occupancy_points = torch.rand((self.occupancy_sample, 3), device=self.device) * (max_bounds - min_bounds) + min_bounds
                loader = torch.utils.data.DataLoader(occupancy_points, batch_size=batch_size, shuffle=False)

                occupancy_mask_  = []
                for batch in loader:
                    # get mask
                    occupancy_mask = Winding_Occupancy(mesh, batch)
                    occupancy_mask = torch.sigmoid((occupancy_mask - 0.5) * 100)
                    occupancy_mask_.append(occupancy_mask)
                occupancy_mask = torch.cat(occupancy_mask_, dim=0)
                sdf, _, _ = knn_points(occupancy_points.unsqueeze(0), surface_points, K=1)
                sdf = sdf.squeeze(0).squeeze(-1).sqrt() * (occupancy_mask - 0.5).sign()
                sdf = 1 * sdf

                # append
                if not occupancy_mask.sum() < 1000:
                    surface_points_list.append(surface_points.cpu())
                    surface_normals_list.append(surface_normals.cpu())
                    occupancy_points_list.append(occupancy_points.cpu())
                    occupancy_mask_list.append(occupancy_mask.cpu()) if not self.sdf else occupancy_mask_list.append(sdf.cpu())
                    # occupancy_mask_list.append(sdf.cpu())
                else:
                    print(f"Skip {mesh_path} due to low occupancy points")

            data = {'surface_points': surface_points_list,
                'surface_normals': surface_normals_list,
                'occupancy_points': occupancy_points_list, 
                'occupancy_mask': occupancy_mask_list,
                'mesh_path': mesh_path_list}
            torch.save(data, self.load_path)
            print("Occupancy dataset saved at {}".format(self.load_path))
        return data
    
    def __len__(self):
        return len(self.data['surface_points'])
    
    def __getitem__(self, idx):
        surface_points = self.data['surface_points'][idx].squeeze(0)
        surface_normals = self.data['surface_normals'][idx].squeeze(0)
        occupancy_points = self.data['occupancy_points'][idx]
        occupancy_mask = self.data['occupancy_mask'][idx]
        if self.N_surf > surface_points.shape[0]:
            selected_surface_points = surface_points
            selected_surface_normals = surface_normals
        else:
            selected_surface_points = surface_points[torch.randperm(surface_points.shape[0])[:self.N_surf]]
            selected_surface_normals = surface_normals[torch.randperm(surface_normals.shape[0])[:self.N_surf]]

        selected_indices = torch.randperm(occupancy_points.shape[0])[:self.N_occupancy]
        selected_occupancy_points = occupancy_points[selected_indices]
        selected_occupancy_mask = occupancy_mask[selected_indices]
        if self.return_normals:
            return selected_surface_points, selected_surface_normals, selected_occupancy_points, selected_occupancy_mask
        else:
            return selected_surface_points, selected_occupancy_points, selected_occupancy_mask

    def set_N(self, N_surf, N_occupancy):
        self.N_surf = N_surf
        self.N_occupancy = N_occupancy
    
    def debug(self, idx):
        surface_points = self.data['surface_points'][idx].squeeze(0)
        occupancy_points = self.data['occupancy_points'][idx]
        occupancy_mask = self.data['occupancy_mask'][idx]
        mesh_path = self.data['mesh_path'][idx]

        pv.set_jupyter_backend('html')
        pv.start_xvfb()
        mesh = trimesh.load(mesh_path)
        p = pv.Plotter()
        # p.add_mesh(mesh, color='lightgrey')

        occupancy_points_0 = occupancy_points[occupancy_mask < 0.05]
        occupancy_points_1 = occupancy_points[occupancy_mask >= 0.95]

        occupancy_colors = occupancy_mask.cpu().numpy()
        # p.add_points(occupancy_points.cpu().numpy(), scalars=occupancy_colors, point_size=1, cmap='jet')
        p.add_points(surface_points.cpu().numpy(), color='black', point_size=2)
        p.add_points(occupancy_points_0.cpu().numpy(), color='red', point_size=5)
        p.add_points(occupancy_points_1.cpu().numpy(), color='blue', point_size=5)
        p.show()


            

        
class SymmetricOccupancyDataset(Dataset):
    """
    We gather closed meshes (ground truth) and calculate occupancy field for each mesh.
    """
    def __init__(self, shape_dir: str, 
                 load_path: str, re_solve=False,
                 surface_sample: int=2048, occupancy_sample=12800,
                 near_ratio:int=2,
                 ghd_reconstruct: GHD_Reconstruct=None,
                 device=torch.device("cuda:0"),
                 sdf=False,
                 return_normals=False,
                 boxsize=0.25,
                 return_gt_mesh=False
                 ):
        self.shape_dir = shape_dir
        self.load_path = load_path
        self.re_solve = re_solve
        self.near_ratio = near_ratio,
        self.surface_sample = surface_sample
        self.occupancy_sample = occupancy_sample
        self.sdf = sdf
        self.return_normals = return_normals
        self.return_gt_mesh = return_gt_mesh

        # boxsize
        self.boxsize = boxsize

        # plugins 
        self.ghd_reconstruct = ghd_reconstruct

        self.device = device

        if not os.path.exists(self.load_path) or re_solve:
            self.data = self.solve_dataset()
        else:
            self.data = torch.load(self.load_path, map_location=torch.device('cpu'))

        # calculate bounding box
        self.bound_box()
        

    def solve_dataset(self):
        N = self.occupancy_sample
        with torch.no_grad():
            # process batch_size points at a time to save memory
            batch_size = 5000
            shape_list = [dir for dir in os.listdir(self.shape_dir) if os.path.isdir(os.path.join(self.shape_dir, dir)) and 'canonical' not in dir]
            mesh_path_list = [f'{self.shape_dir}/{shape}/part_aligned_updated.obj' for shape in shape_list]

            surface_points_list = []
            surface_normals_list = []
            volume_points_list = []
            near_points_list = []

            for mesh_path in tqdm(mesh_path_list):
                # load mesh & normalize (to keep same normalization with ghd)
                mesh = load_objs_as_meshes([mesh_path]).to(self.device)
                mesh = Meshes(verts=[mesh.verts_packed() / self.ghd_reconstruct.norm_canonical], faces=[mesh.faces_packed()])

                # surface points
                surface_points, surface_normals = sample_points_from_meshes(mesh, self.surface_sample, return_normals=True)
                # occupancy points
                min_bounds = mesh.verts_packed().min(dim=0)[0]
                max_bounds = mesh.verts_packed().max(dim=0)[0]
                expansion = self.boxsize * (max_bounds - min_bounds)
                min_bounds -= expansion
                max_bounds += expansion

                count = 0
                volume_points = []
                near_points = []
                for batch in range(round(500*N/batch_size)):
                    occupancy_points = torch.rand((batch_size, 3), device=self.device) * (max_bounds - min_bounds) + min_bounds
                    occupancy_mask = Winding_Occupancy(mesh, occupancy_points)
                    occupancy_mask = torch.sigmoid((occupancy_mask - 0.5) * 100)
                    volume_points.append(occupancy_points[occupancy_mask > 0.95])
                    near_points.append(occupancy_points[occupancy_mask < 0.05])
                    count += len(occupancy_points[occupancy_mask > 0.95])
                    if count > N:
                        break
                if count < N:
                    raise ValueError(f"Insufficient points for {mesh_path}")
                volume_points = torch.cat(volume_points, dim=0)[:N, :]
                if self.near_ratio is None:
                    near_points = torch.cat(near_points, dim=0)[:N, :]
                    
                else:
                    near_points = torch.cat(near_points, dim=0)[:(N*2), :]
                    sdf, _, _ = knn_points(near_points.unsqueeze(0), surface_points, K=1)
                    sdf = sdf.squeeze(0).squeeze(-1).sqrt()
                    near_points = near_points[sdf.argsort()[:N], :]
                # append
                if not len(volume_points) < 1000:
                    surface_points_list.append(surface_points.cpu())
                    surface_normals_list.append(surface_normals.cpu())
                    volume_points_list.append(volume_points.cpu())
                    near_points_list.append(near_points.cpu())
                else:
                    print(f"Skip {mesh_path} due to low occupancy points")

            data = {'surface_points': surface_points_list,
                'surface_normals': surface_normals_list,
                'volume_points': volume_points_list, 
                'near_points': near_points_list,
                'mesh_path': mesh_path_list}
            torch.save(data, self.load_path)
            print("Occupancy dataset saved at {}".format(self.load_path))
        return data
    
    def __len__(self):
        return len(self.data['surface_points'])
    
    def __getitem__(self, idx):
        surface_points = self.data['surface_points'][idx].squeeze(0)
        surface_normals = self.data['surface_normals'][idx].squeeze(0)
        volume_points = self.data['volume_points'][idx]
        near_points = self.data['near_points'][idx]

        if self.N_surf > surface_points.shape[0]:
            selected_surface_points = surface_points
            selected_surface_normals = surface_normals
        else:
            selected_surface_points = surface_points[torch.randperm(surface_points.shape[0])[:self.N_surf]]
            selected_surface_normals = surface_normals[torch.randperm(surface_normals.shape[0])[:self.N_surf]]

        selected_indices = torch.randperm(volume_points.shape[0])[:self.N_occupancy]
        volume_queries = torch.cat([volume_points[selected_indices], near_points[selected_indices]], dim=0)
        labels = torch.cat([1*torch.ones(self.N_occupancy), 0 * torch.ones(self.N_occupancy)])

        if self.return_normals:
            if not self.return_gt_mesh:
                return selected_surface_points, selected_surface_normals, volume_queries, labels
            else:
                return selected_surface_points, selected_surface_normals, volume_queries, labels, idx
        else:
            return selected_surface_points, volume_queries, labels

    def set_N(self, N_surf, N_occupancy):
        self.N_surf = N_surf
        self.N_occupancy = N_occupancy
    
    def debug(self, idx):
        surface_points = self.data['surface_points'][idx].squeeze(0)
        occupancy_points = self.data['occupancy_points'][idx]
        occupancy_mask = self.data['occupancy_mask'][idx]
        mesh_path = self.data['mesh_path'][idx]

        pv.set_jupyter_backend('html')
        pv.start_xvfb()
        mesh = trimesh.load(mesh_path)
        p = pv.Plotter()
        # p.add_mesh(mesh, color='lightgrey')

        occupancy_points_0 = occupancy_points[occupancy_mask < 0.05]
        occupancy_points_1 = occupancy_points[occupancy_mask >= 0.95]

        occupancy_colors = occupancy_mask.cpu().numpy()
        # p.add_points(occupancy_points.cpu().numpy(), scalars=occupancy_colors, point_size=1, cmap='jet')
        p.add_points(surface_points.cpu().numpy(), color='black', point_size=2)
        p.add_points(occupancy_points_0.cpu().numpy(), color='red', point_size=5)
        p.add_points(occupancy_points_1.cpu().numpy(), color='blue', point_size=5)
        p.show()

    def bound_box(self):
        min_bounds = []
        max_bounds = []
        for i in range(len(self)):
            surface_points = self.data['surface_points'][i].squeeze(0)
            min_bounds.append(surface_points.min(dim=0)[0])
            max_bounds.append(surface_points.max(dim=0)[0])
        min_bounds = torch.stack(min_bounds, dim=0)
        max_bounds = torch.stack(max_bounds, dim=0)
        min_bounds = min_bounds.min(dim=0)[0]
        max_bounds = max_bounds.max(dim=0)[0]
        self.min_bounds = min_bounds
        self.max_bounds = max_bounds

    def generate_queries(self, bsz, query_size):
        min_bounds = self.min_bounds.to(self.device)
        max_bounds = self.max_bounds.to(self.device)
        volume_queries = torch.rand((bsz, query_size, 3), device=self.device) * (max_bounds - min_bounds) + min_bounds
        return volume_queries
    
    def generate_queries_voxel(self, voxel_size=64):
        min_bounds = self.min_bounds.to(self.device)
        max_bounds = self.max_bounds.to(self.device)
        grid = torch.meshgrid(
            torch.linspace(min_bounds[0], max_bounds[0], voxel_size),
            torch.linspace(min_bounds[1], max_bounds[1], voxel_size),
            torch.linspace(min_bounds[2], max_bounds[2], voxel_size)
        )
        grid = torch.stack(grid, dim=-1).reshape(-1, 3).to(self.device)
        spacing = (max_bounds - min_bounds) / voxel_size
        return grid, spacing, min_bounds

    @torch.no_grad()
    def marching_cube(self, bsz, voxel_size, Occupancy_VAE, CUS_Diffuser, batch_size=2, smooth=True):
        surfaces = []
        queries, spacing, min_bounds = self.generate_queries_voxel(voxel_size)
        queries = queries.unsqueeze(0).repeat(batch_size, 1, 1)
        for ii in tqdm(range(round(bsz/batch_size)+1)):
            latents = CUS_Diffuser.sample(batch_size, steps=50, eta=0.0)
            occupancies = Occupancy_VAE.query_geometry(queries, latents[0])
            if smooth:
                occupancies = torch.tanh(occupancies)
            for i in range(batch_size):
                occupancy = occupancies[i].view(voxel_size, voxel_size, voxel_size)
                # Extract surface for the current instance
                verts, faces, normals, values = measure.marching_cubes(occupancy.detach().cpu().numpy(), level=0.0, spacing=spacing.cpu().numpy())
                verts = torch.tensor(verts, device=self.device) + min_bounds.view(1, -1)
                faces = torch.tensor(faces.copy(), device=self.device)
                meshes = Meshes(verts=[verts], faces=[faces])
                surfaces.append(meshes)
            if len(surfaces)>=bsz:
                break
        return surfaces
    
    @torch.no_grad()
    def marching_cube_true(self, latents, voxel_size, Occupancy_VAE, CUS_Diffuser, batch_size=2, smooth=True):
        surfaces = []
        queries, spacing, min_bounds = self.generate_queries_voxel(voxel_size)
        queries = queries.unsqueeze(0).repeat(batch_size, 1, 1)
        bsz = latents.shape[0]
        class custom_dataset(torch.utils.data.Dataset):
            def __init__(self, data):
                self.data = data
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]

        loader = DataLoader(custom_dataset(latents), batch_size=batch_size, shuffle=False)
        for latent in loader:
            latent = Occupancy_VAE.decode(latent)
            occupancies = Occupancy_VAE.query_geometry(queries, latent)
            if smooth:
                occupancies = torch.tanh(occupancies)
            for i in range(batch_size):
                occupancy = occupancies[i].view(voxel_size, voxel_size, voxel_size)
                # Extract surface for the current instance
                verts, faces, normals, values = measure.marching_cubes(occupancy.detach().cpu().numpy(), level=0.0, spacing=spacing.cpu().numpy())
                verts = torch.tensor(verts, device=self.device) + min_bounds.view(1, -1)
                faces = torch.tensor(faces.copy(), device=self.device)
                meshes = Meshes(verts=[verts], faces=[faces])
                surfaces.append(meshes)
            if len(surfaces)>=bsz:
                break
        surfaces = join_meshes_as_batch(surfaces)
        return surfaces
        
    def return_gt_meshes(self, idx):
        mesh_path = [self.data['mesh_path'][idx_] for idx_ in idx.tolist()]
        Meshes_ = load_objs_as_meshes(mesh_path)
        Meshes_ = Meshes_.update_padded(Meshes_.verts_padded() / self.ghd_reconstruct.norm_canonical)
        return Meshes_






