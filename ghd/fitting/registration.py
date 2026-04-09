"""
Resigration classes for ghd fitting.
Truth mesh is registered to enable opening alignment & differentiable centreline losses during ghd fitting.
"""
import open3d as o3d
import numpy as np
import numpy
import logging
import os
import trimesh
import shapely
from ...utils import utils_registration as u_register
import pickle
from pytorch3d.structures import Meshes
import torch
import sys
from ...utils.utils import o3d_mesh_to_pytorch3d
import vtk
import pytorch3d as p3d
import igraph as ig
from tqdm import tqdm
from skeletor.utilities import make_trimesh
import pyvista as pv
from pytorch3d.io import save_obj, load_objs_as_meshes
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle


class RegistrationwOpeningAlignment(object):
    def __init__(self, args, root, target, num_op=3, suffix=None):
        self.device = torch.device(args.device)
        self.root = root
        self.target = target
        self.suffix = suffix if suffix is not None else '.obj'
        assert self.suffix == '.obj', 'Not implemented for mesh file other than .obj'
        # mesh objects of true complexes
        self.mesh_target = o3d.io.read_triangle_mesh(os.path.join(self.root, self.target + self.suffix))
        self.mesh_target_trimesh = trimesh.load(os.path.join(self.root, self.target + self.suffix))
        self.mesh_target_p3d = o3d_mesh_to_pytorch3d(self.mesh_target)
        self.num_op = num_op  # number of openings
        # assembly of opening v indices (num_op, N), v coordinates (num_op, N, 3), n (num_op, N, 3)
        self.op_v_indices, self.op_v_coords, self.op_v_normal, self.op_n_mean = [], [], [], []
        # assembly of newly reconstructed plane meshes
        self.op_rec_v, self.op_rec_f = [], []
        # assembly of mapped reconstructed plane meshes
        self.op_rec_v_indices_map, self.op_rec_f_map = [], []

    def register_openings(self):
        """
        Select nodes on an opening.
        No need to register in a sequence, code will sort it clock-wise.
        No need to register all nodes on an opening, as long as the reconstructed mesh captures the majortiy of the area.
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(self.mesh_target.vertices))
        pcd.estimate_normals()
        for idx in range(self.num_op):
            print('Select nodes on opening No. {}'.format(idx))
            v_indices = u_register.pick_points(pcd)
            self.op_v_indices.append(v_indices)
            self.op_v_coords.append(np.asarray(pcd.points)[v_indices])
            self.op_v_normal.append(np.asarray(pcd.normals)[v_indices])
            self.op_n_mean.append(u_register.get_average_cross_product_from_center(np.asarray(pcd.points)[v_indices]))

    def create_opening_meshes(self, viz=False):
        """
        Create opening meshes from registered nodes.
        """
        for idx in range(self.num_op):
            points_projected = u_register.pcd_to_approx_plane(self.op_v_coords[idx], self.op_n_mean[idx])
            point_sequence = np.concatenate((np.arange(points_projected.shape[0]), np.array([0])))
            points_projected = np.asarray( list(np.concatenate((points_projected, points_projected[0, :].reshape(-1, 3)))))
            path_topo = trimesh.path.entities.Entity(point_sequence)
            path3D = trimesh.path.path.Path3D([path_topo], vertices=points_projected)
            path2D, to_3D = path3D.to_planar()
            sorted_2D_vertices = u_register.clock_sort_2D_points(np.asarray(path2D.vertices))
            # check points sequence is clock wise
            # warning: do not close the polygon which leads to error
            polygon = shapely.geometry.Polygon(sorted_2D_vertices)
            vertices, faces = trimesh.creation.triangulate_polygon(polygon, triangle_args='p')
            # some boring processing
            faces -= 1  # shapely index vertices from 1 instead of 0. :(
            vertices = vertices[:-1, :]  # trimesh will add one last vertices
            points_projected = points_projected[:-1, :]  # skim off the overlapped vert (the 1st one)
            vertices_3d = u_register.trimesh_points_2d_to_3d(vertices, to_3D)
            # return point sequence in original system
            op_rec_v_indices_map, op_rec_f_map = u_register.get_mapped_sequence_and_faces(self.op_v_indices[idx], points_projected, vertices_3d, faces)
            # trimesh's 2d to 3d is done by sequence, however when feeding vertices to construct path3D or path2D,
            if viz:
                u_register.lazy_viz_mesh(vertices_3d, faces)
            self.op_rec_v.append(vertices_3d)
            self.op_rec_f.append(faces)
            self.op_rec_v_indices_map.append(op_rec_v_indices_map)
            self.op_rec_f_map.append(op_rec_f_map)

    def save_checkpoint_opa(self, chk_path: str):
        # automatic saving
        chk_path = os.path.join(self.root, self.target, 'opa_checkpoint') if chk_path is None else chk_path
        if not os.path.exists(os.path.dirname(chk_path)):
            os.makedirs(os.path.dirname(chk_path))
        chk = {'op_v_indices': self.op_v_indices, 'op_v_coords': self.op_v_coords, 'op_v_normal': self.op_v_normal,
               'op_n_mean': self.op_n_mean,
               'op_rec_v': self.op_rec_v, 'op_rec_f': self.op_rec_f,
               'op_rec_v_indices_map': self.op_rec_v_indices_map, 'op_rec_f_map': self.op_rec_f_map}
        # use self.op_rec_f to offset opening meshes, use self.op_rec_f_map if creating opening meshes from mother mesh
        if not chk_path.endswith('.pkl'):
            chk_path += '.pkl'
        with open(chk_path, 'wb') as f:
            pickle.dump(chk, f)

    def load_checkpoint_opa(self, chk_path: str, redo=False):
        # automatic loading
        chk_path = os.path.join(self.root, self.target, 'opa_checkpoint') if chk_path is None else \
            chk_path
        if not chk_path.endswith('.pkl'):
            chk_path += '.pkl'
        # register openings, create meshes and save chk if chk doesn't exist
        if not os.path.exists(chk_path) or redo:
            logging.warning('checkpoint does not exist, redo registration.')
            self.register_openings()
            self.create_opening_meshes()
            self.save_checkpoint_opa(None)
        with open(chk_path, 'rb') as f:
            chk = pickle.load(f)
        for key in chk.keys():
            setattr(self, key, chk[key])
        logging.info('checkpoint has been loaded {}'.format(chk))
        self.log_register = 'Yes'
        self.log_reconstruct = 'Yes'
        return None

    def return_opening_Meshes_static(self, register_normal=True) -> list:
        # return opening meshes for non-canonical shapes (static)
        opening_Meshes = []
        for idx in range(self.num_op):
            verts = torch.tensor(self.op_rec_v[idx]).unsqueeze(0).float()
            faces = torch.tensor(self.op_rec_f[idx]).unsqueeze(0).float()
            normals = torch.tensor(np.repeat(self.op_n_mean[idx].reshape(-1, 3), verts.shape[1], axis=0)).unsqueeze(0).float()
            opening_Meshes.append(Meshes(verts=verts, faces=faces, verts_normals=normals if register_normal else None))
        return opening_Meshes


class RegistrationwOpeningAlignmentwCentreline(RegistrationwOpeningAlignment):
    def __init__(self, args, root, target, num_op=3, suffix=None, num_cep=4, c_suffix="Centerline model.vtk"):
        self.num_cep = num_cep  # number of centreline end points
        self.c_suffix = c_suffix  # suffix of true complex mesh object (only support obj for now)
        self.centreline_pcd = None  # pcd of cecntreline object
        self.cep_registration = None  # registration of centreline end points
        super(RegistrationwOpeningAlignmentwCentreline, self).__init__(args, root, target, num_op, suffix)

    def register_centreline(self):
        """
        Need to pick cep points in a sequence.
        """
        centreline_file = os.path.join(self.root, self.target, self.c_suffix)
        centreline_pcd = load_point_cloud_vtk(centreline_file).cpu()
        self.centreline_pcd = centreline_pcd
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(centreline_pcd.cpu()))
        for idx in range(self.num_cep):
            print("Pick centreline end points for each parent vessel branch.")
            self.cep_registration = torch.Tensor(u_register.pick_points(pcd))

    def save_checkpoint_centreline(self, chk_path: str):
        chk_path = os.path.join(self.root, self.target, 'centreline_checkpoint') if chk_path is None else chk_path
        if not os.path.exists(os.path.dirname(chk_path)):
            os.makedirs(os.path.dirname(chk_path))
        chk = {'centreline_pcd': self.centreline_pcd,
               'cep_registration': self.cep_registration}
        if not chk_path.endswith('.pkl'):
            chk_path += '.pkl'
        with open(chk_path, 'wb') as f:
            pickle.dump(chk, f)

    def load_checkpoint_centreline(self, chk_path: str, redo=False):
        chk_path = os.path.join(self.root, self.target, 'centreline_checkpoint') if chk_path is None else chk_path
        if not chk_path.endswith('.pkl'):
            chk_path += '.pkl'
        if not os.path.exists(chk_path) or redo:
            print('centreline checkpoints does not exist, redo registration.')
            self.register_centreline()
            self.save_checkpoint_centreline(None)
        with open(chk_path, 'rb') as f:
            chk = pickle.load(f)
        for key in chk.keys():
            setattr(self, key, chk[key])
        logging.info('checkpoint of centreline has been loaded {}'.format(chk))


class RegistrationwOpeningAlignmentwDifferentiableCentreline(RegistrationwOpeningAlignment):
    def __init__(self, args, root, target, num_op=3, num_cep=3, num_waves=5, step_size=2):
        self.num_cep = num_cep  # number of centreline end points
        self.centreline_pcd = None  # pcd of cecntreline object
        self.cep_registration = None  # registration of centreline end points indices
        self.num_waves = num_waves  # number of waves to cast for each cep point
        self.step_size = step_size  
        self.wave_loops = None  # registration of loops (List[List[]])
        super(RegistrationwOpeningAlignmentwDifferentiableCentreline, self).__init__(args, root, target, num_op, suffix=None)

    def register_centreline_end_points(self):
        """
        No need to pick cep points in a sequence.
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(self.mesh_target.vertices))
        pcd.estimate_normals()
        self.cep_registration = u_register.pick_points(pcd)

    def save_checkpoint_centreline(self, chk_path: str):
        chk_path = os.path.join(self.root, self.target, 'diff_centreline_checkpoint') if chk_path is None else chk_path
        if not os.path.exists(os.path.dirname(chk_path)):
            os.makedirs(os.path.dirname(chk_path))
        chk = {'diff_cep_registration': self.cep_registration,
               'wave_loops': self.wave_loops}
        if not chk_path.endswith('.pkl'):
            chk_path += '.pkl'
        with open(chk_path, 'wb') as f:
            pickle.dump(chk, f)

    def load_checkpoint_centreline(self, chk_path: str, redo=False):
        chk_path = os.path.join(self.root, self.target, 'diff_centreline_checkpoint') if chk_path is None else chk_path
        if not chk_path.endswith('.pkl'):
            chk_path += '.pkl'
        if not os.path.exists(chk_path) or redo:
            logging.warning('checkpoint does not exist, redo registration.')
            self.register_centreline_end_points()
            self._cast_waves()
            self.save_checkpoint_centreline(None)
        with open(chk_path, 'rb') as f:
            chk = pickle.load(f)
        for key in chk.keys():
            setattr(self, key, chk[key])
        print('Differentiable centreline checkpoint has been loaded {}'.format(chk))

    def _cast_waves(self, random_origin=False, progress=True):
        """
        Cast waves across mesh, refer to:
        https://github.com/navis-org/skeletor.git
        """
        if not random_origin:
            origins = self.cep_registration
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
        waves = int(len(origins)) if origins is not None else self.num_waves
        if waves < 1:
            raise ValueError('`waves` must be integer >= 1')
        # Same for step size
        step_size = int(self.step_size)
        if step_size < 1:
            raise ValueError('`step_size` must be integer >= 1')
        mesh = make_trimesh(self.mesh_target_trimesh, validate=False)
        G = ig.Graph(edges=mesh.edges_unique, directed=False)
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
                np.random.seed(1985)  # make seeds predictable
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
                    loops_list.append(loop_list)
                pbar.update(len(cc))
        self.wave_loops = loops_list
        return None


def load_point_cloud_vtk(vtk_file_path):
    # load pcd from vtk file
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtk_file_path)
    reader.Update()
    point_cloud = reader.GetOutput()
    points = point_cloud.GetPoints()
    num_points = points.GetNumberOfPoints()
    point_cloud_array = np.zeros((num_points, 3))  # Initialize array
    for i in range(num_points):
        point = points.GetPoint(i)
        point_cloud_array[i] = point
    point_cloud_array = torch.Tensor(point_cloud_array)
    return point_cloud_array

def p3d_to_pv(Meshes: Meshes):
    verts = Meshes.verts_packed()
    faces = Meshes.faces_packed()
    verts = verts.detach().cpu().numpy()
    faces = faces.detach().cpu().numpy()
    poly_data = pv.PolyData(verts)
    poly_data.faces = faces
    return poly_data


