import os
import sys
# sys.path.append(os.path.join(os.path.dirname(__file__),'.','..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pytorch3d.loss import chamfer_distance, mesh_laplacian_smoothing, mesh_normal_consistency, mesh_edge_loss
from pytorch3d.ops import sample_points_from_meshes, cot_laplacian, padded_to_packed
from .meshloss import Mesh_loss
from ghd.fitting.registration import RegistrationwOpeningAlignmentwDifferentiableCentreline
import torch
from pytorch3d.structures import Meshes
import numpy as np
from pytorch3d.loss import chamfer_distance
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
from .diceloss import BinaryDiceLoss
from .meshloss_do import Mesh_loss_differentiable_occupancy
from torch import Tensor


class Mesh_loss_do_differentiable_centreline(Mesh_loss_differentiable_occupancy):
    def __init__(self, args, class_canonical: RegistrationwOpeningAlignmentwDifferentiableCentreline,
                 class_target: RegistrationwOpeningAlignmentwDifferentiableCentreline):
        # save computation, store target centreline pcd during init
        self.cpcd_target = extract_centreline_pcd(getattr(class_target, "mesh_target_p3d").verts_packed(),
                                                  getattr(class_target, "wave_loops")).to(args.device)
        self.wave_loops_canonical = getattr(class_canonical, "wave_loops")
        super(Mesh_loss_do_differentiable_centreline, self).__init__(args, class_canonical, class_target)

    def forward_do_dcforward_opa_do(self, warped_mesh, warped_openings, loss_weighting: dict, query_points,
                                    do_gt, B=1):
        loss_dict = self.forward_opa_do(warped_mesh, warped_openings, loss_weighting, query_points, do_gt, B)
        if 'loss_diff_centreline' in loss_weighting:
            cpcd_canonical = extract_centreline_pcd(warped_mesh.verts_packed(), self.wave_loops_canonical)
            loss_diff_centreline, _ = chamfer_distance(cpcd_canonical.unsqueeze(0), self.cpcd_target.unsqueeze(0))
            loss_dict['loss_diff_centreline'] = loss_diff_centreline
        return loss_dict


def extract_centreline_pcd(verts: Tensor, wave_loops: list):
    cpcd = []
    for wave_loop in wave_loops:
        for loop in wave_loop:
            indices = torch.tensor(loop).long()
            extracted_pcd = verts[indices]
            averaged_pcd = torch.mean(extracted_pcd, dim=0, keepdim=True)
            cpcd.append(averaged_pcd)
    return torch.cat(cpcd, dim=0)  # centreline pcd


def o3d_mesh_to_pytorch3d(o3d_mesh) -> Meshes:
    verts = torch.Tensor(np.asarray(o3d_mesh.vertices))
    faces = torch.Tensor(np.asarray(o3d_mesh.triangles))
    pytorch3d_mesh = Meshes(verts=[verts], faces=[faces])
    return pytorch3d_mesh
