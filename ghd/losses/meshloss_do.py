import os
import sys
# sys.path.append(os.path.join(os.path.dirname(__file__),'.','..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pytorch3d.loss import chamfer_distance,mesh_laplacian_smoothing, mesh_normal_consistency, mesh_edge_loss
from pytorch3d.ops import sample_points_from_meshes, cot_laplacian, padded_to_packed
from .meshloss import Mesh_loss
from ghd.fitting.registration import RegistrationwOpeningAlignment
import torch
from pytorch3d.structures import Meshes
import numpy as np
from pytorch3d.loss import chamfer_distance
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
from ghd.base.mesh_geometry3 import Winding_Occupancy
from .diceloss import BinaryDiceLoss, BinaryDiceLoss_Weighted
import math
import pyvista as pv
import trimesh


class Mesh_loss_differentiable_occupancy(Mesh_loss):
    def __init__(self, args, oa_class_canonical: RegistrationwOpeningAlignment, oa_class_target: RegistrationwOpeningAlignment):
        self.device = torch.device(args.device)
        base_shape = getattr(oa_class_canonical, "mesh_target_p3d").to(self.device)  # transform o3d mesh to pytorch3d mesh
        self.target_mesh = getattr(oa_class_target, "mesh_target_p3d").to(self.device)
        # TODO: check if chamfer loss cares about normal direction
        # register static meshes for the target mesh
        self.target_openings = oa_class_target.return_opening_Meshes_static(register_normal=False)
        self.sample_num = args.sample_num
        self.op_sample_num = args.op_sample_num
        self.do_dpi = getattr(args, "do_dpi", 32)
        self.do_style = None
        self.do_module = None  # upsampling module if using uniform
        self.do_loss_type = getattr(args, "do_loss_type", "dice_loss")
        self.do_number = getattr(args, "do_number", 10000)
        self.root_target = args.root_target
        self.name_target = args.name_target
        self.weights_attention = None
        super(Mesh_loss_differentiable_occupancy, self).__init__(mesh_std=base_shape, sample_num=self.sample_num)

    def get_static_mask_probabilistic(self):
        device = torch.device('cpu')
        bound_box1 = self.mesh_std.get_bounding_boxes().squeeze(0).to(device)
        bound_box2 = self.target_mesh.get_bounding_boxes().squeeze(0).to(device)
        box_min = torch.min(torch.cat((bound_box1, bound_box2), dim=1), dim=-1)[0]
        box_max = torch.max(torch.cat((bound_box1, bound_box2), dim=1), dim=-1)[0]
        bound_box = torch.cat((box_min.unsqueeze(-1), box_max.unsqueeze(-1)), dim=-1)
        radius = bound_box[:, 1] - bound_box[:, 0]
        bound_box[:, 0] -= 0.125 * radius
        bound_box[:, 1] += 0.125 * radius
        dpi = torch.max((bound_box[:, 1]-bound_box[:, 0]) / torch.min((bound_box[:, 1]-bound_box[:, 0])/self.do_dpi)).int().item()
        scale = []
        for axis in range(3):
            scale.append(torch.linspace(bound_box[axis, 0], bound_box[axis, 1], dpi))
        x_grid, y_grid, z_grid = torch.meshgrid(scale)
        voxel = torch.stack([x_grid, y_grid, z_grid], dim=-1)
        voxel = voxel.view(-1, 3).to(device)
        batch_size = round(voxel.shape[0]/10)
        voxel_set = TensorDataset(voxel)
        loader = DataLoader(voxel_set, batch_size=batch_size, shuffle=False)
        distance_list = []
        for batch in iter(loader):
            reference = self.mesh_std.verts_packed().unsqueeze(0).expand(batch[0].shape[0], -1, -1).to(device)
            distances, _ = chamfer_distance(batch[0].unsqueeze(1).to(device), reference, batch_reduction=None)
            distance_list.append(distances)
        distances = torch.cat(distance_list, dim=0)  # zero dim tensor indicating distance to the mesh
        sorted_indices = torch.argsort(distances).to(device)
        min_prob = 0.1
        max_prob = 0.9
        prob_step = (max_prob - min_prob) / (distances.size(0) - 1)
        probabilities = torch.linspace(min_prob, max_prob, steps=distances.size(0)).to(device)
        sorted_probabilities = probabilities[sorted_indices]
        mask = torch.rand_like(distances) > sorted_probabilities
        masked_tensor = torch.cat((voxel, distances.unsqueeze(-1)), dim=-1)
        masked_tensor = masked_tensor[mask, :3]
        print('static query points registered.')
        return masked_tensor  # [N, 3]

    def get_static_mask_uniform(self):
        device = torch.device('cpu')
        bound_box1 = self.mesh_std.get_bounding_boxes().squeeze(0).to(device)
        bound_box2 = self.target_mesh.get_bounding_boxes().squeeze(0).to(device)
        box_min = torch.min(torch.cat((bound_box1, bound_box2), dim=1), dim=-1)[0]
        box_max = torch.max(torch.cat((bound_box1, bound_box2), dim=1), dim=-1)[0]
        bound_box = torch.cat((box_min.unsqueeze(-1), box_max.unsqueeze(-1)), dim=-1)
        radius = bound_box[:, 1] - bound_box[:, 0]
        bound_box[:, 0] -= 0.125 * radius
        bound_box[:, 1] += 0.125 * radius
        dpi = torch.max((bound_box[:, 1] - bound_box[:, 0]) / torch.min(
            (bound_box[:, 1] - bound_box[:, 0]) / self.do_dpi)).int().item()
        scale = []
        for axis in range(3):
            scale.append(torch.linspace(bound_box[axis, 0], bound_box[axis, 1], dpi))
        x_grid, y_grid, z_grid = torch.meshgrid(scale)
        voxel = torch.stack([x_grid, y_grid, z_grid], dim=-1).to(device)  # [N,3]
        voxel = voxel.view(-1, 3).to(device)
        # mod = torch.nn.Upsample(scale_factor=2, mode='trilinear')
        # voxel = mod(voxel)
        # voxel = voxel.view(-1, 3)  #
        return voxel

    def get_static_mask_number_control(self, num_in=10000, num_out=10000):
        device = torch.device('cpu')
        batch_size = num_in + num_out
        # get bounding box
        bound_box1 = self.mesh_std.get_bounding_boxes().squeeze(0).to(device)
        bound_box2 = self.target_mesh.get_bounding_boxes().squeeze(0).to(device)
        box_min = torch.min(torch.cat((bound_box1, bound_box2), dim=1), dim=-1)[0]
        box_max = torch.max(torch.cat((bound_box1, bound_box2), dim=1), dim=-1)[0]
        bound_box = torch.cat((box_min.unsqueeze(-1), box_max.unsqueeze(-1)), dim=-1)
        radius = bound_box[:, 1] - bound_box[:, 0]
        bound_box[:, 0] -= 0.125 * radius
        bound_box[:, 1] += 0.125 * radius
        # iteratively sample until enough
        count_num_in = 0
        count_num_out = 0
        query_points_in = []
        query_points_out = []
        for batch in range(20):
            x = np.random.uniform(bound_box[0, 0], bound_box[0, 1], size=batch_size)
            y = np.random.uniform(bound_box[1, 0], bound_box[1, 1], size=batch_size)
            z = np.random.uniform(bound_box[2, 0], bound_box[2, 1], size=batch_size)
            query_points = torch.tensor(np.stack([x, y, z], axis=1)).float()
            do_gt = torch.sigmoid((Winding_Occupancy(self.target_mesh.to(device), query_points) - 0.5) * 100)
            indices_in = torch.where(do_gt > 0.95)[0]
            indices_out = torch.where(do_gt < 0.05)[0]
            count_num_in += indices_in.shape[0]
            count_num_out += indices_out.shape[0]
            query_points_in.append(query_points[indices_in, :].numpy())
            query_points_out.append(query_points[indices_out, :].numpy())
            print("{} and {} points have been founded for static dc registration.\n".format(count_num_in, count_num_out))
            if count_num_in >= num_in and count_num_out >= num_out:
                break
        query_points_in = torch.Tensor(np.concatenate(query_points_in, axis=0))[:num_in, :]
        query_points_out = torch.Tensor(np.concatenate(query_points_out, axis=0))[:num_out, :]
        query_points = torch.cat((query_points_in, query_points_out), dim=0)
        do_gt = torch.cat((torch.ones(num_in), torch.zeros(num_out)), dim=0)
        return query_points, do_gt

    def get_static_mask_number_control_v2(self, num_in=10000, num_out=10000, expand_ratio=5, smooth=0.02, inspect=False, redo=True):
        """
        :param num_in: point number inside the shape
        :param num_out:
        :param expand_ratio: 1 / expand_ratio of the points will be retained
        :return:
        """
        device = torch.device('cpu')
        batch_size = round((num_in + num_out) / 4)
        # get bounding box
        bound_box1 = self.mesh_std.get_bounding_boxes().squeeze(0).to(device)
        bound_box2 = self.target_mesh.get_bounding_boxes().squeeze(0).to(device)
        box_min = torch.min(torch.cat((bound_box1, bound_box2), dim=1), dim=-1)[0]
        box_max = torch.max(torch.cat((bound_box1, bound_box2), dim=1), dim=-1)[0]
        bound_box = torch.cat((box_min.unsqueeze(-1), box_max.unsqueeze(-1)), dim=-1)
        radius = bound_box[:, 1] - bound_box[:, 0]
        bound_box[:, 0] -= 0.125 * radius
        bound_box[:, 1] += 0.125 * radius
        # iteratively sample until enough
        count_num_in = 0
        count_num_out = 0
        query_points_in = []
        query_points_out = []
        points_path = os.path.join(self.root_target, self.name_target, "do_points.pt")
        if not os.path.exists(points_path) or redo:
            print("do_points no found, redoing points searching")
            for batch in range(1000):
                x = np.random.uniform(bound_box[0, 0], bound_box[0, 1], size=batch_size)
                y = np.random.uniform(bound_box[1, 0], bound_box[1, 1], size=batch_size)
                z = np.random.uniform(bound_box[2, 0], bound_box[2, 1], size=batch_size)
                query_points = torch.tensor(np.stack([x, y, z], axis=1)).float()
                do_gt = torch.sigmoid((Winding_Occupancy(self.target_mesh.to(device), query_points) - 0.5) * 100)
                indices_in = torch.where(do_gt > 0.95)[0]
                indices_out = torch.where(do_gt < 0.05)[0]
                if count_num_in <= expand_ratio * num_in:
                    count_num_in += indices_in.shape[0]
                    if batch == 0:
                        query_points_in = query_points[indices_in, :]
                    else:
                        query_points_in = torch.cat((query_points_in, query_points[indices_in, :]), dim=0)
                if count_num_out <= expand_ratio * num_out:
                    count_num_out += indices_out.shape[0]
                    if batch == 0:
                        query_points_out = query_points[indices_out, :]
                    else:
                        query_points_out = torch.cat((query_points_out, query_points[indices_out, :]), dim=0)
                print("{} and {} points have been founded for static dc registration.\n".format(count_num_in, count_num_out))
                if count_num_in >= expand_ratio * num_in and count_num_out >= expand_ratio * num_out:
                    break
            # calculate field strength
            query_points_in = torch.Tensor(query_points_in)[: expand_ratio * num_in, :]
            query_points_out = torch.Tensor(query_points_out)[: expand_ratio * num_out, :]
            torch.save({'query_points_in': query_points_in, 'query_points_out': query_points_out}, points_path)
        else:
            print("do points successfully loaded!")
            dict_pt = torch.load(points_path)
            query_points_in, query_points_out = dict_pt['query_points_in'], dict_pt['query_points_out']
        # dict_pt = torch.load('points.pt')
        # query_points_in, query_points_out = dict_pt['query_points_in'], dict_pt['query_points_out0']
        source_verts = torch.cat((self.mesh_std.verts_packed().detach().cpu().float(),
                                          self.target_mesh.verts_packed().detach().cpu().float()), dim=0)
        distance_in = torch.norm(source_verts.unsqueeze(1) - query_points_in.unsqueeze(0), dim=2) + smooth
        distance_out = torch.norm(source_verts.unsqueeze(1) - query_points_out.unsqueeze(0), dim=2) + smooth
        strength_in = torch.sum((1 / torch.pow(distance_in, 2)), dim=0)
        strength_out = torch.sum((1 / torch.pow(distance_out, 2)), dim=0)
        strength = torch.cat((strength_in, strength_out), dim=0)
        _, topk_indices_in = torch.topk(strength_in, num_in)
        _, topk_indices_out = torch.topk(strength_out, num_out)
        # query_points_in_filtered = query_points_in[topk_indices_in, :]
        query_points_in_filtered = query_points_in[:num_in, :]  # thin vessels don't need control for inside
        query_points_out_filtered = query_points_out[topk_indices_out, :]
        query_points = torch.cat((query_points_in_filtered, query_points_out_filtered), dim=0)
        # # visualize
        if inspect:
            p = pv.Plotter()
            trimesh_canonical = trimesh.Trimesh(self.mesh_std.verts_packed().detach().cpu().numpy(),
                                                self.mesh_std.faces_packed().detach().cpu().numpy())
            trimesh_target = trimesh.Trimesh(self.target_mesh.verts_packed().detach().cpu().numpy(),
                                             self.target_mesh.faces_packed().detach().cpu().numpy())
            p.add_mesh(trimesh_canonical, color='blue', opacity=0.5, pickable=False)
            p.add_mesh(trimesh_target, color='red', opacity=0.5, pickable=False)
            p.add_points(query_points_out_filtered.numpy(), color='black')
            # p.add_points(query_points_out.numpy(), color='grey')
            p.show()
        # return points and ground truth of differentiable voxel masks
        do_gt = torch.cat((torch.ones(num_in), torch.zeros(num_out)), dim=0)
        return query_points, do_gt

    def get_static_mask_and_gt(self, style='number_control_v2'):
        device = torch.device('cpu')
        if style == 'uniform':
            query_points = self.get_static_mask_uniform()
            # query_points_upsample = query_points.permute((3, 0, 1, 2)).unsqueeze(0)
            # self.do_module = torch.nn.Upsample(scale_factor=2, mode='trilinear')
            # query_points_upsample = self.do_module(query_points_upsample)
            do_gt = torch.sigmoid((Winding_Occupancy(self.target_mesh.to(device), query_points) - 0.5) * 100)
            query_points = query_points.view(-1, 3)
        elif style == 'number_control':
            query_points, do_gt = self.get_static_mask_number_control(num_in=self.do_number, num_out=self.do_number)
        elif style == 'number_control_v2':
            query_points, do_gt = self.get_static_mask_number_control_v2(num_in=self.do_number, num_out=self.do_number,
                                                                         expand_ratio=2, smooth=0.02, redo=True)
        else:
            query_points = self.get_static_mask_probabilistic()
            do_gt = torch.sigmoid((Winding_Occupancy(self.target_mesh.to(device), query_points) - 0.5) * 100)

        self.do_style = style

        return query_points, do_gt

    def get_weights_attention(self, query_points, min_w=1.0, max_w=3.0, smooth=0.01, inspect=False):
        # calculate field strength
        batch_size = int(1000)
        batch_num = math.ceil(query_points.shape[0] / batch_size)
        source_verts = self.target_mesh.verts_packed().detach().cpu().float()
        strength = []
        for i in range(batch_num):
            query_points_batch = query_points[i*batch_size: (i+1)*batch_size].cpu().float()
            distance_batch = torch.norm(source_verts.unsqueeze(1) - query_points_batch.unsqueeze(0), dim=2) + smooth
            strength_batch = torch.sum((1 / torch.pow(distance_batch, 2)), dim=0)
            strength.append(strength_batch)
            print('field strength has been caluclated for batch {}'.format(i))
        strength = torch.cat(strength, dim=0)
        # map to certain range
        ratio = (max_w - min_w) / (torch.max(strength) - torch.min(strength))
        strength = (strength - torch.min(strength)) * ratio + min_w
        # visualize
        if inspect:
            index = np.argsort(strength.detach().cpu().numpy())[::-1]
            ratio = 1.0  # for inspecting dropping threshold
            index = index[:round(ratio * len(index))]
            cloud = pv.PolyData(query_points.detach().cpu().numpy()[index, :])
            cloud['attention'] = strength.cpu().numpy()[index]  # just use z coordinate
            pv.plot(cloud, scalars='attention', cmap='jet', show_bounds=True, cpos='yz')
        # p = pv.Plotter()
        # p.add_points(query_points.detach().cpu().numpy(), render_points_as_spheres=False, color='red', point_size=0.1)
        # p.add_points(source_verts.numpy(), render_points_as_spheres=True, color='blue', point_size=1)
        # p.show()
        self.weights_attention = strength.unsqueeze(0).detach().to(self.device)
        print('field strength assigned')

    def forward_opa_do(self, warped_mesh, warped_openings, loss_weighting: dict, query_points, do_gt, do_index, B=1):
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
                loss_p_list.append(loss_p if not torch.isnan(loss_p) else torch.Tensor([0.0]).to(self.device))
                loss_n_list.append(loss_n if not torch.isnan(loss_n) else torch.Tensor([0.0]).to(self.device))
            if 'loss_openings_p' in loss_weighting:
                loss_dict['loss_openings_p'] = loss_p_list
            if 'loss_openings_n' in loss_weighting:
                loss_dict['loss_openings_n'] = loss_n_list
        if 'loss_do' in loss_weighting:
            if self.do_style == 'uniform_upsample':
                winding_field = torch.sigmoid((Winding_Occupancy(warped_mesh, query_points) - 0.5) * 100)
                resolution = int(np.cbrt(query_points.shape[0]))
                winding_field = winding_field.view((resolution, resolution, resolution)).unsqueeze(0).unsqueeze(0)
                winding_field = self.do_module(winding_field)
                winding_field = winding_field.squeeze(0).squeeze(0).view(-1)
            else:
                winding_field = torch.sigmoid((Winding_Occupancy(warped_mesh, query_points) - 0.5) * 100)
                # winding_field = Winding_Occupancy(warped_mesh, query_points)
            # loss_do = dice_loss.forward(winding_field.unsqueeze(0), do_gt.unsqueeze(0))
            if self.do_loss_type == "mse_loss":
                loss_do = self.mse_loss(winding_field, do_gt)
            elif self.do_loss_type == "dice_loss":
                loss_do = self.dice_loss.forward(winding_field.unsqueeze(0), do_gt.unsqueeze(0))
            elif self.do_loss_type == "dice_loss_attention":
                loss_do = self.dice_loss_attention.forward(winding_field.unsqueeze(0), do_gt.unsqueeze(0), self.weights_attention[:, do_index])
            else:
                raise NotImplementedError("do loss type not implemented")
            loss_dict['loss_do'] = loss_do if not torch.isnan(loss_do) else torch.Tensor([0.0]).to(self.device)
        return loss_dict



def o3d_mesh_to_pytorch3d(o3d_mesh) -> Meshes:
    verts = torch.Tensor(np.asarray(o3d_mesh.vertices))
    faces = torch.Tensor(np.asarray(o3d_mesh.triangles))
    pytorch3d_mesh = Meshes(verts=[verts], faces=[faces])
    return pytorch3d_mesh
