import open3d as o3d
import numpy as np
import vtk
import pyvista as pv
from typing import Tuple
import os
import pickle
import logging
import json
import itertools
from scipy.interpolate import splprep, splev
import pytorch3d as p3d
from pytorch3d.structures import Meshes, join_meshes_as_batch
from pytorch3d.io import save_obj, load_objs_as_meshes
import trimesh
from trimesh.exchange.obj import export_obj
import torch
from pytorch3d.transforms import axis_angle_to_matrix
from utils.utils import safe_load_mesh


class ExcisionRegistrationFromVMTKBranches:
    def __init__(self, surface_mesh_file: str, centreline_dir: str, label, save_dir):
        """
        we document centerline information from vmtk, and perform registration for excision
        """
        self.surface_mesh_file = surface_mesh_file
        self.centreline_dir = centreline_dir
        self.label = label
        self.unit = None
        self.control_points_positions = []
        self.control_points_radius = []
        self.control_points_positions_sorted = []
        self.control_points_radius_sorted = []
        self.control_points_positions_interpolated = []
        self.control_points_radius_interpolated = []
        self.control_points_radius_interpolated_processed = []  # updated sequence
        self.control_points_positions_interpolated_procesed = []  # updated sequence
        self.num_branches = None
        self.branch_length = []  # total length of each branch
        self.branch_length_record = []  # length record for each interpolated point in each branch
        self.branch_register_sequence = []  # the sequence of branches, determined through human registration
        self.save_dir = save_dir

    def register_branches_automatic(self, visual_inspect=True):
        # extract data
        self.num_branches = len(os.listdir(self.centreline_dir))  # number of branches
        for branch_file in os.listdir(self.centreline_dir):
            cur_file = os.path.join(self.centreline_dir, branch_file)
            position, unit, radius = read_mrk_json_file_vmtk(cur_file)
            self.control_points_positions.append(position)
            self.unit = unit
            self.control_points_radius.append(radius)

        # automatically sort the orientation: branch -->
        logging.warning('Assuming vmtk already has sorted the control points for the branch!!!')
        index = [0, -1]
        index_product = list(itertools.product(index, repeat=self.num_branches))
        perimeters = []
        for idx in range(len(index_product)):
            vertices = []
            for id_b in range(self.num_branches):
                vertices.append(self.control_points_positions[id_b][index_product[idx][id_b], :])
            vertices = np.array(vertices)
            perimeters.append(get_perimeter(vertices))
        sorted_product = index_product[np.argmin(np.array(perimeters))]  # get the correct index product
        for id_v in range(len(sorted_product)):
            if sorted_product[id_v] == 0:
                self.control_points_positions_sorted.append(self.control_points_positions[id_v])
                self.control_points_radius_sorted.append(self.control_points_radius[id_v])
            elif sorted_product[id_v] == -1:
                self.control_points_positions_sorted.append(self.control_points_positions[id_v][::-1, :])
                self.control_points_radius_sorted.append(self.control_points_radius[id_v][::-1])
                logging.warning('flipping branch no.{}'.format(id_v))

    def comprehend_branches_automatic(self, human_inspect=False):
        # needs branch registration done first
        for id_branch in range(len(self.control_points_positions_sorted)):
            interpolated_points, interpolated_radius = interpolate_control_points(
                self.control_points_positions_sorted[id_branch],
                self.control_points_radius_sorted[id_branch])
            self.control_points_positions_interpolated.append(interpolated_points)
            self.control_points_radius_interpolated.append(interpolated_radius)
        logging.warning('finish interpolating branches')
        # human inspect
        if human_inspect:
            # add surface mesh
            if self.surface_mesh_file.endswith('.obj'):
                surface_mesh = pv.read(self.surface_mesh_file)
            else:
                reader = vtk.vtkXMLPolyDataReader()
                reader.SetFileName(self.surface_mesh_file)
                reader.Update()
                surface_mesh = reader.GetOutput()
            p = pv.Plotter()
            p.add_mesh(surface_mesh, color='black', opacity=0.025, pickable=False)
            # add pcd for control points
            color_list = ['blue', 'red', 'green', 'yellow', 'black', 'darkblue', 'pink', 'gray']
            for idx in range(self.num_branches):
                color = color_list.pop() if len(color_list) != 0 else np.random.rand(3)
                p.add_points(self.control_points_positions_sorted[idx], render_points_as_spheres=True, color=color,
                             point_size=8)
                p.add_points(self.control_points_positions_interpolated[idx], render_points_as_spheres=False,
                             color=color, point_size=8)
            p.show()
        # calculate total length for every branches
        for i in range(self.num_branches):
            total_length, length_record = get_total_length_of_a_branch(self.control_points_positions_interpolated[i])
            self.branch_length.append(total_length)
            self.branch_length_record.append(length_record)
            print('record branch no {} has length = {}'.format(i, total_length))

    def register_branches_human(self):
        # add surface mesh
        assert self.surface_mesh_file.endswith('.vtp'), 'surface mesh file must end with .vtp'
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(self.surface_mesh_file)
        reader.Update()
        surface_mesh = reader.GetOutput()
        p = pv.Plotter()
        p.add_mesh(surface_mesh, color='black', opacity=0.025, pickable=False)
        # add pcd for control points
        color_list = ['blue', 'red', 'green', 'yellow', 'black', 'darkblue', 'pink', 'gray']
        for idx in range(self.num_branches):
            color = color_list.pop() if len(color_list) != 0 else np.random.rand(3)
            p.add_points(self.control_points_positions_sorted[idx], render_points_as_spheres=True, color=color,
                         point_size=8)
        picked_points_list = []
        logging.warning(
            '***************************\nNow pick ONE point from each branch IN SEQUENCE!!!\n***************************************')

        def register_points_callback(picked_point):
            picked_points_list.append(picked_point)
            print('new point selected: {}'.format(picked_point))

        p.enable_point_picking(callback=register_points_callback)
        p.show()
        for i in range(len(picked_points_list)):
            point = picked_points_list[i]
            distances = []
            for j in range(self.num_branches):
                distances.append(
                    np.min(np.linalg.norm(point - self.control_points_positions_sorted[j], axis=-1), axis=0))
            self.branch_register_sequence.append(np.argmin(np.array(distances)))
        assert len(list(
            set(self.branch_register_sequence))) == self.num_branches, 'You picked multiple points on the same branch'

    def create_markers_for_geomagic(self, branch_retain_length=None, root_trim_length=None):
        max_radius = np.max([np.max(it) for it in self.control_points_radius_sorted])
        grid_points = max_radius * np.array([[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]])
        faces = np.array([[0, 1, 2], [2, 3, 0]])
        assert branch_retain_length is not None, 'must specify branch_retain_length'
        Meshes_list = []
        Cut_points_list = []
        alignment_control_points = []  # control points at every branch for alignment
        alignment_centreline_list = []  # centreline pcd of every branch for alignment
        # perform cutting off for the root
        root_trim_length = 2.5 if root_trim_length is None else root_trim_length
        root_cutoff_id = np.argmin(np.abs(self.branch_length_record[0][-1] - root_trim_length -
                                          self.branch_length_record[0]))
        eps = 0.25
        for i in range(self.num_branches):
            branch_id = self.branch_register_sequence[i]
            self.control_points_radius_interpolated_processed.append(self.control_points_radius_interpolated[branch_id])
            self.control_points_positions_interpolated_procesed.append(self.control_points_positions_interpolated[branch_id])
            length_record = self.branch_length_record[branch_id]
            cut_point_id = np.argmin(np.abs(branch_retain_length[i] - length_record))
            if cut_point_id == 0:
                cut_point_id += 1
            if i == 0:
                if cut_point_id >= root_cutoff_id:
                    cut_point_id = root_cutoff_id
            cut_point_pos = self.control_points_positions_interpolated[branch_id][cut_point_id]
            to_normal = tuple(
                self.control_points_positions_interpolated[branch_id][cut_point_id - 1] -
                self.control_points_positions_interpolated[branch_id][cut_point_id])
            from_normal = (0, 0, 1)
            rotation_matrix = calculate_rotation_matrix(from_normal, to_normal)
            rotated_points = np.dot(grid_points, rotation_matrix.T) + cut_point_pos
            Meshes_list.append(Meshes([torch.Tensor(rotated_points)], [torch.Tensor(faces)]))
            Cut_points_list.append(rotated_points)
            # add another planes at the "correct" location if the branch is too short
            extrusion_length = np.abs(length_record[cut_point_id] - branch_retain_length[i])
            if extrusion_length >= eps:
                print('detect branch being too short for cutting, adding additional extrusion plane')
                to_normal = np.array(to_normal)
                to_normal /= np.linalg.norm(to_normal)
                trans = extrusion_length * to_normal
                Meshes_list.append(Meshes([torch.Tensor(rotated_points - trans)], [torch.Tensor(faces)]))
                Cut_points_list.append(rotated_points - trans)
                alignment_centreline = self.control_points_positions_interpolated[branch_id][0: cut_point_id]
                num_interpolate = 500
                interpolated_values = np.zeros((num_interpolate, alignment_centreline.shape[1]))
                for ii in range(alignment_centreline.shape[1]):
                    interpolated_values[:, ii] = cut_point_pos[ii] - trans[ii] * np.linspace(0, 1, num_interpolate)
                alignment_centreline = np.concatenate((alignment_centreline, interpolated_values[1:, :])).transpose().tolist()
                tck, u = splprep(alignment_centreline, s=0, k=3)
                new_data = splev(np.linspace(0, 1, num_interpolate), tck)
                alignment_centreline = np.stack((new_data[0], new_data[1], new_data[2]), axis=1)
                print('interpolation has been done for one of the branch')
            else:
                trans = np.zeros((1, 3))

                alignment_centreline = self.control_points_positions_interpolated[branch_id][:cut_point_id]

            # add registration point coordinates for alignment
            control_point = cut_point_pos - trans
            control_point = control_point.reshape(3)
            alignment_control_points.append(control_point)
            # save centreline branch pcd
            alignment_centreline_list.append(alignment_centreline)

        # save cutting planes
        joined_Meshes = join_meshes_as_batch(Meshes_list)
        filename_obj = os.path.join(self.save_dir, "cutting_planes.obj")
        if os.path.isfile(filename_obj):
            os.remove(filename_obj)
        save_obj(filename_obj, verts=joined_Meshes.verts_packed(), faces=joined_Meshes.faces_packed())

        # save alignment control points & centreline branch pcd
        chk_path = os.path.join(self.save_dir, "centreline4alignment")
        alignment_control_points = torch.Tensor(np.array(alignment_control_points))
        chk = dict()
        chk = {'alignment_control_points': alignment_control_points,
               'alignment_centreline_list': alignment_centreline_list}
        # use self.op_rec_f to offset opening meshes, use self.op_rec_f_map if creating opening meshes from mother mesh
        if not chk_path.endswith('.pkl'):
            chk_path += '.pkl'
        with open(chk_path, 'wb') as f:
            pickle.dump(chk, f)

        # reader = vtk.vtkXMLPolyDataReader()
        # reader.SetFileName(self.surface_mesh_file)
        # reader.Update()
        # surface_mesh = reader.GetOutput()
        # p = pv.Plotter()
        # p.add_mesh(surface_mesh, color='black', opacity=0.025, pickable=False)
        # color = ["red", "blue", "yellow"]
        # for point, color in zip([Cut_points_list], color):
        #     p.add_points(point, render_points_as_spheres=True, point_size=8, color=color)
        # p.show()

    def visualization(self, surface_mesh, pcds: list):
        # if surface_mesh_path.endswith('.vtp'):
        #     reader = vtk.vtkXMLPolyDataReader()
        #     reader.SetFileName(self.surface_mesh_file)
        #     reader.Update()
        #     surface_mesh = reader.GetOutput()
        # elif surface_mesh_path.endswith('.obj'):
        #     surface_mesh = trimesh.load(surface_mesh_path)
        # else:
        #     raise Exception("only vtp and obj files allowed.")

        p = pv.Plotter()
        p.add_mesh(surface_mesh, color='black', opacity=0.025, pickable=False)
        # add pcd for control points
        color_list = ['blue', 'red', 'green']
        for idx in range(len(pcds)):
            color = color_list[idx]
            p.add_points(pcds[idx], render_points_as_spheres=False, color=color, point_size=8)  # middle points
            p.add_points(pcds[idx][0, :], render_points_as_spheres=True, color="black", point_size=8)  # start point
            p.add_points(pcds[idx][-1, :], render_points_as_spheres=True, color="yellow", point_size=8)  # end point
        p.show()

    def save_checkpoint(self, chk_path: str):
        chk_path = os.path.join(self.save_dir, 'checkpoint') if chk_path is None else \
            chk_path
        if not os.path.exists(os.path.dirname(chk_path)):
            os.makedirs(os.path.dirname(chk_path))
        chk = dict()
        chk = {'label': self.label, 'unit': self.unit,
               'control_points_positions': self.control_points_positions, 'control_points_radius': self.control_points_radius,
               'control_points_positions_sorted': self.control_points_positions_sorted,
               'control_points_radius_sorted': self.control_points_radius_sorted,
               'control_points_positions_interpolated': self.control_points_positions_interpolated,
               'control_points_radius_interpolated': self.control_points_radius_interpolated,
               'num_branches': self.num_branches,
               'branch_length': self.branch_length,
               'branch_length_record': self.branch_length_record,
               'branch_register_sequence': self.branch_register_sequence,
               'save_dir': self.save_dir}
        # use self.op_rec_f to offset opening meshes, use self.op_rec_f_map if creating opening meshes from mother mesh
        if not chk_path.endswith('.pkl'):
            chk_path += '.pkl'
        with open(chk_path, 'wb') as f:
            pickle.dump(chk, f)

    def load_checkpoint(self, chk_path: str, redo=False, branch_retain_length=None, root_trim_length=None):
        # automatic loading
        chk_path = os.path.join(self.save_dir, 'checkpoint') if chk_path is None else \
            chk_path
        if not chk_path.endswith('.pkl'):
            chk_path += '.pkl'
        # register openings, create meshes and save chk if chk doesn't exist
        if not os.path.exists(chk_path) or redo:
            logging.warning('checkpoint does not exist, redo registration.')
            self.register_branches_automatic()
            self.comprehend_branches_automatic(human_inspect=True)
            self.register_branches_human()
        if os.path.exists(chk_path) and not redo:
            with open(chk_path, 'rb') as f:
                chk = pickle.load(f)
            shadow_save_dir = self.save_dir
            for key in chk.keys():
                setattr(self, key, chk[key])
            self.save_dir = shadow_save_dir
        self.create_markers_for_geomagic(branch_retain_length=branch_retain_length, root_trim_length=root_trim_length)
        self.save_checkpoint(chk_path)
        logging.warning('checkpoint has been loaded {}'.format(chk_path))
        return None


def find_interpolated_point_matching_retain_length(interpolated_points: np.ndarray, retain_length):
    num_points = interpolated_points.shape[0]
    cur_length = 0
    length_record = []
    for i in range(0, num_points):
        length_record.append(cur_length)
        cur_length += np.sqrt()


def read_mrk_json_file_vmtk(json_file):
    # load mrk.json file
    with open(json_file, 'r') as f:
        markups_data = json.load(f)
    controlPoints = markups_data['markups'][0]['controlPoints']
    measurements = markups_data['markups'][0]['measurements']
    # fill control points and orientation array
    position = []
    radius = []
    unit = None
    # orientation = []  # orientation doesn't seem to contain tangent info, discard
    for i in range(len(controlPoints)):
        position.append(np.array(controlPoints[i]['position']))
    position = np.array(position)  # extract control points' coordinates~
    unit = measurements[0]['units']
    radius = measurements[5]['controlPointValues']
    print('successfully extracted info from {} control points'.format(len(controlPoints)))
    # if control points equals 3, add one more point
    assert len(position) >= 3, 'one of the branches have less than 3 control points, which is unacceptable'
    if len(position) == 3:
        print('one of the branches have only 3 control points, will do linear interpolation')
        li_position = 0.5 * (position[0, :] + position[1, :])
        li_radius = 0.5 * (radius[0] + radius[1])
        position = np.concatenate((position[0, :].reshape((-1, 3)), li_position.reshape((-1, 3)), position[1:, :]), axis=0)
        radius.insert(1, li_radius)
    return position, unit, radius


def get_perimeter(vertices: np.ndarray):
    num_vertices = vertices.shape[0]
    indices = range(num_vertices)
    combinations_list = list(itertools.combinations(indices, 2))
    combinations_list = [sorted(set(comb)) for comb in combinations_list]
    perimeter = 0
    for id_line in range(len(combinations_list)):
        it_combination = combinations_list[id_line]
        assert len(it_combination) == 2, 'invalid combination'
        va = vertices[it_combination[0], :]
        vb = vertices[it_combination[1], :]
        perimeter += np.sqrt(np.sum((va - vb) ** 2))
    return perimeter


def interpolate_control_points(control_point_positions, control_point_radius, num_interpolate=500):
    assert control_point_positions.shape[1] == 3, 'invalid control point dimension'
    data = []
    for i in range(3):
        data.append(control_point_positions[:, i])
    data.append(control_point_radius)
    # interpolate
    try:
        tck, u = splprep(data, s=0, k=3)
    except TypeError:
        print('One of the branches have control points less than 4, which is a must for k=3 spline function')
    new_data = splev(np.linspace(0, 1, num_interpolate), tck)
    interpolated_points = np.stack((new_data[0], new_data[1], new_data[2]), axis=1)
    interpolated_radius = np.array(new_data[3])
    return interpolated_points, interpolated_radius


def get_total_length_of_a_branch(control_point_positions: np.ndarray):
    assert control_point_positions.shape[1] == 3, 'invalid control point dimension'
    num_points = control_point_positions.shape[0]
    total_length = 0
    length_record = [0]
    for i in range(num_points - 1):
        total_length += np.sqrt(np.sum((control_point_positions[i + 1] - control_point_positions[i]) ** 2))
        length_record.append(total_length)
    length_record = np.array(length_record)
    return total_length, length_record


def calculate_rotation_matrix(from_normal, to_normal):
    """
    get rotational matrix from one normal vector to another

    Parameters:
        from_normal (tuple): original normal vector
        to_normal (tuple): targeted normal vector

    Returns:
        ndarray: Rotational matrix
    """
    # normalize normal vector
    from_normal = np.array(from_normal) / np.linalg.norm(from_normal)
    to_normal = np.array(to_normal) / np.linalg.norm(to_normal)

    # calcualte rotational axis and angle
    axis = np.cross(from_normal, to_normal)
    angle = np.arccos(np.dot(from_normal, to_normal))

    # R matrix
    rotation_matrix = rotation_matrix_from_axis_angle(axis, angle)

    return rotation_matrix


def rotation_matrix_from_axis_angle(axis, angle):
    """
    rotational matrix from axis and angle

    Parameters:
        axis (ndarray): 
        angle (float): 

    Returns:
        ndarray: 
    """
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c
    x, y, z = axis

    rotation_matrix = np.array([[t * x * x + c, t * x * y - z * s, t * x * z + y * s],
                                [t * x * y + z * s, t * y * y + c, t * y * z - x * s],
                                [t * x * z - y * s, t * y * z + x * s, t * z * z + c]])

    return rotation_matrix


def rotate_to_plane(points, to_normal):
    """
    将 XY 平面上的点旋转到另一个平面。

    Parameters:
        points (ndarray): 形状为 (N, 3) 的点数组，表示 XY 平面上的点。
        to_normal (tuple): 给定平面的法向量，形如 (nx, ny, nz)。

    Returns:
        ndarray: 形状为 (N, 3) 的点数组，表示旋转后的点。
    """
    # 将 XY 平面上的点投影到给定平面上
    projected_points = project_to_plane(points, to_normal)

    # 计算旋转矩阵
    rotation_matrix = calculate_rotation_matrix((0, 0, 1), to_normal)

    # 将投影点通过旋转矩阵旋转到给定平面
    rotated_points = np.dot(projected_points, rotation_matrix.T)

    return rotated_points