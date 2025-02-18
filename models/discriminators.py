"""
Discriminator components for VAE training.
"""

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


class SAModule(torch.nn.Module):
    def __init__(self, ratio, r_list: List, K_list: List, nn_list: List):
        super().__init__()
        self.ratio = ratio
        self.r_list = r_list
        self.K_list = K_list
        self.conv_list = torch.nn.ModuleList([PointNetConv(nn_, add_self_loops=False) for nn_ in nn_list])

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        x_dst = None if x is None else x[idx]
        x_list = []
        for r, K, conv in zip(self.r_list, self.K_list, self.conv_list):
            row, col = radius(pos, pos[idx], r, batch, batch[idx], max_num_neighbors=K)
            edge_index = torch.stack([col, row], dim=0)
            x_list.append(conv((x, x_dst), (pos, pos[idx]), edge_index))
        x = torch.cat(x_list, dim=1)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class PointNet2(torch.nn.Module):
    def __init__(self, use_norm=True):
        super().__init__()

        # SA layers
        in_channels = 6 if use_norm else 3
        self.sa1_module = SAModule(0.25, [0.1], [16], [MLP([in_channels+3, 8, 8], norm='layer_norm')]*1)
        self.sa2_module = SAModule(0.25, [0.2], [32], [MLP([8*1 + 3, 12, 16], norm='layer_norm')]*1)
        self.sa3_module = SAModule(0.25, [0.4], [64], [MLP([16*1 + 3, 24, 32], norm='layer_norm')]*1)
        self.sa4_module = GlobalSAModule(MLP([32*1 + 3, 48, 64], norm='layer_norm'))

        # scoring layer
        self.mlp = torch.nn.ModuleList([MLP([64, 64], dropout=0.1, norm=None),
                                        MLP([64, 1], act=None, norm=None)])

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        sa4_out = self.sa4_module(*sa3_out)
        x, pos, batch = sa4_out
        for mlp in self.mlp:
            x = mlp(x)  # [B, 1]
        return x


class PointNet2_Classifier(torch.nn.Module):
    def __init__(self, use_norm=False, num_classes=5, extract_features=False):
        super().__init__()
        """
        Pretrained PoinetNet2 for calculating FPD & KPD
        """
        self.latent_dim = 2048
        in_channels = 6 if use_norm else 3
        self.sa1_module = SAModule(0.1, [0.05, 0.1], [16, 16], [MLP([in_channels+3, 16, 32])]*2)
        self.sa2_module = SAModule(0.2, [0.2, 0.4], [32, 32], [MLP([32*2 + 3, 64, 128])]*2)
        self.sa3_module = SAModule(0.5, [0.5, 1.0], [64, 64], [MLP([128*2 + 3, 256, 512])]*2)
        self.sa4_module = GlobalSAModule(MLP([512*2 + 3, 1024, self.latent_dim]))

        # regression layer
        self.mlp = torch.nn.ModuleList([MLP([2048, 512]),
                                        MLP([512, 64]),
                                        MLP([64, num_classes], act=None, norm=None)])
        
        self.extract_features = extract_features

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        sa4_out = self.sa4_module(*sa3_out)
        x, pos, batch = sa4_out
        if not self.extract_features:
            for mlp in self.mlp:
                x = mlp(x)  # [B, num_classes]
        else:
            pass
        return x

    
        
