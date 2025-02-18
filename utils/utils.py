import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import point_cloud_utils as pcu
import open3d as o3d
import torch
import os
from pytorch3d.structures import Meshes
import pytorch3d
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from tqdm import trange


def o3d_mesh_to_pytorch3d(o3d_mesh) -> Meshes:
    verts = torch.Tensor(np.asarray(o3d_mesh.vertices))
    faces = torch.Tensor(np.asarray(o3d_mesh.triangles))
    pytorch3d_mesh = Meshes(verts=[verts], faces=[faces])
    return pytorch3d_mesh

def safe_load_mesh(path):
    mesh = o3d.io.read_triangle_mesh(path)
    mesh = o3d_mesh_to_pytorch3d(mesh)
    return mesh
