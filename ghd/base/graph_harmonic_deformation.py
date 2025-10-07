"""Base classes of Graph Harmonic Deformation fitting."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.structures import Meshes
from pytorch3d.io import load_objs_as_meshes, save_obj
from pytorch3d.ops import cubify, cot_laplacian, sample_points_from_meshes, knn_points, knn_gather, norm_laplacian
from torch_geometric.utils import degree, to_undirected, to_dense_adj, get_laplacian, add_self_loops
from torch_geometric.data import Data
from torch_scatter import scatter
import pickle
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import coo_matrix
# from ghd.base.graph_operators import NativeFeaturePropagation, LaplacianSmoothing, Laplacain, Normal_consistence
from pytorch3d.loss import chamfer_distance,mesh_laplacian_smoothing, mesh_normal_consistency, mesh_edge_loss
from pytorch3d.ops import sample_points_from_meshes, cot_laplacian, padded_to_packed
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.structures import Meshes
from torch_geometric.utils import to_undirected
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from ..fitting.registration import RegistrationwOpeningAlignment


def mix_laplacian(base_shape: Meshes):
    device = base_shape.device
    cotweight, _ = cot_laplacian(base_shape.verts_packed(), base_shape.faces_packed())
    connection = cotweight.coalesce().indices()
    cotweight_value = cotweight.coalesce().values()
    connection, cotweight_value = to_undirected(connection, edge_attr=cotweight_value)
    normweight = norm_laplacian(base_shape.verts_packed(), base_shape.edges_packed())
    connection_1 = normweight.coalesce().indices()
    normweight_value = normweight.coalesce().values()
    _, normweight = to_undirected(connection_1, edge_attr=normweight_value)
    normweight_mean = scatter(normweight, connection[0], dim=0, reduce='add')
    cotweight_mean = scatter(cotweight_value, connection[0], dim=0, reduce='add')
    normweight_value = normweight_value / normweight_mean[connection[0]]
    cotweight_value = cotweight_value / cotweight_mean[connection[0]]
    norlap =  add_self_loops(connection,edge_attr=normweight_value,fill_value='add', num_nodes=base_shape.verts_packed().shape[0])
    norlap_adj = add_self_loops(connection,edge_attr=normweight_value,fill_value=0, num_nodes=base_shape.verts_packed().shape[0])
    norlap = torch.sparse_coo_tensor(indices=norlap[0], values=norlap[1], size=(base_shape.verts_packed().shape[0], base_shape.verts_packed().shape[0]), dtype=torch.float32, device=device)
    norlap_adj = torch.sparse_coo_tensor(indices=norlap_adj[0], values=norlap_adj[1], size=(base_shape.verts_packed().shape[0], base_shape.verts_packed().shape[0]), dtype=torch.float32, device=device)
    norlap = norlap - 2*norlap_adj
    norlap_np = coo_matrix((norlap.coalesce().values().cpu().numpy(), (norlap.coalesce().indices()[0].cpu().numpy(), norlap.coalesce().indices()[1].cpu().numpy())))
    cotlap =  add_self_loops(connection,edge_attr=-cotweight_value,fill_value=1+1e-6, num_nodes=base_shape.verts_packed().shape[0])
    cotlap = torch.sparse_coo_tensor(indices=cotlap[0], values=cotlap[1], size=(base_shape.verts_packed().shape[0], base_shape.verts_packed().shape[0]), dtype=torch.float32, device=device)
    cotlap_np = coo_matrix((cotlap.coalesce().values().cpu().numpy(), (cotlap.coalesce().indices()[0].cpu().numpy(), cotlap.coalesce().indices()[1].cpu().numpy())))
    standerlap = get_laplacian(connection, num_nodes = base_shape.verts_packed().shape[0], dtype=torch.float32, normalization='sym')
    standerlap = torch.sparse_coo_tensor(indices=standerlap[0], values=standerlap[1], size=(base_shape.verts_packed().shape[0], base_shape.verts_packed().shape[0]), dtype=torch.float32, device=device)
    standerlap_np = coo_matrix((standerlap.coalesce().values().cpu().numpy(), (standerlap.coalesce().indices()[0].cpu().numpy(), standerlap.coalesce().indices()[1].cpu().numpy())))
    return cotlap_np, norlap_np, standerlap_np


class Graph_Harmonic_Deform(nn.Module):
    def __init__(self, base_shape: Meshes, num_Basis = 6*6+1, mix_lap_weight= [1,0.1,0.1], eigen_chk=None):
        super(Graph_Harmonic_Deform, self).__init__()
        self.device = base_shape.device
        self.base_shape = base_shape
        self.cotlap_np, self.norlap_np, self.standerlap_np = mix_laplacian(base_shape)
        self.mix_lap = mix_lap_weight[0]*self.cotlap_np + mix_lap_weight[1]*self.norlap_np + mix_lap_weight[2]*self.standerlap_np
        if eigen_chk is not None:
            with open(eigen_chk, 'rb') as f:
                chk = pickle.load(f)
            for key_ in chk.keys():
                setattr(self, key_, chk[key_].to(self.device))
        else:
            self.GBH_eigval, self.GBH_eigvec = eigsh(self.mix_lap, k=num_Basis, which='SM')
            self.GBH_eigvec = torch.from_numpy(self.GBH_eigvec).to(base_shape.device).float()
            self.GBH_eigval = torch.from_numpy(self.GBH_eigval).to(base_shape.device).float().unsqueeze(0)
        self.deformation_param = nn.Parameter(torch.zeros((num_Basis, 3), dtype=torch.float32, device=base_shape.device), requires_grad=True)
        self.reset_affine_param()
    
    def to(self, device):
        self.device = device
        self.base_shape = self.base_shape.to(device)
        self.GBH_eigvec = self.GBH_eigvec.to(device)
        self.GBH_eigval = self.GBH_eigval.to(device)
        self.deformation_param = nn.Parameter(self.deformation_param.to(device))
        self.R = nn.Parameter(self.R.to(device))
        self.s = nn.Parameter(self.s.to(device))
        self.T = nn.Parameter(self.T.to(device))
        return self

    def reset_affine_param(self):
        self.R = nn.Parameter(torch.zeros(1,3, device=self.device))
        self.s = nn.Parameter(torch.tensor([1.], device=self.device).unsqueeze(0))
        self.T = nn.Parameter(torch.zeros(1,3, device=self.device))

    def ghb_coefficient_recover(self, GHB_coefficient):
        assert GHB_coefficient.shape[0] == self.GBH_eigvec.shape[-1]
        try:
            n = GHB_coefficient.shape[1]
        except:
            n = 1
            GHB_coefficient = GHB_coefficient.unsqueeze(-1)
        return self.GBH_eigvec.matmul(GHB_coefficient) 
    
    def project_to_ghb_eig(self, input_shape):
        assert input_shape.shape[0] == self.GBH_eigvec.shape[0]
        d = input_shape.shape[-1]
        return self.GBH_eigvec.transpose(-1,-2).matmul(input_shape)
    
    def forward(self, GHB_coefficient=None):
        if GHB_coefficient is None:
            GHB_coefficient = self.deformation_param
        deformation = self.ghb_coefficient_recover(GHB_coefficient)  # n: recover laplacian
        output_shape = self.base_shape.offset_verts(deformation)  # n: x, y, and z coefficients are seperated. so we have (basis, 3)
        R_matrix = axis_angle_to_matrix(self.R)
        output_shape = output_shape.update_padded((output_shape.verts_padded() @ R_matrix.transpose(-1,-2)*self.s + self.T).float())  # n: not solved: rigid scaling, rot, and trans
        return output_shape


class Graph_Harmonic_Deform_opening_alignment_dynamic(Graph_Harmonic_Deform):
    def __init__(self, args, oa_class: RegistrationwOpeningAlignment):
        self.device = torch.device(args.device)
        base_shape = getattr(oa_class, "mesh_target_p3d").to(self.device)
        for attr_name in ["op_rec_v_indices_map", "op_rec_f"]:
            if not attr_name.startswith("__"):  # make sure won't copy private attributes
                setattr(self, attr_name, getattr(oa_class, attr_name))
        self.num_op = args.num_op
        self.open_Meshes = []
        # create opening Meshes
        for idx in range(self.num_op):
            self.open_Meshes.append(Meshes(verts=[base_shape.verts_packed()[self.op_rec_v_indices_map[idx], :]],
                                           faces=[torch.Tensor(self.op_rec_f[idx]).to(self.device)]))  # use non-mapped face indices
        super(Graph_Harmonic_Deform_opening_alignment_dynamic, self).__init__(base_shape=base_shape,
                                                                              num_Basis=args.num_Basis,
                                                                              mix_lap_weight=args.mix_lap_weights)

    def forward_with_opening_alignment(self, GHB_coefficient=None):
        if GHB_coefficient is None:
            GHB_coefficient = self.deformation_param
        deformation = self.ghb_coefficient_recover(GHB_coefficient)  # [nv, 3]
        output_shape = self.base_shape.offset_verts(deformation)  # x, y, and z coefficients. so we have (basis, 3)
        output_openings = []
        for idx in range(self.num_op):
            output_openings.append(self.open_Meshes[idx].offset_verts(deformation[self.op_rec_v_indices_map[idx], :]))
        R_matrix = axis_angle_to_matrix(self.R)
        output_shape = output_shape.update_padded((output_shape.verts_padded() @ R_matrix.transpose(-1,-2)*self.s.abs() + self.T).float())
        return output_shape, output_openings