from torch import LongTensor
from torch_geometric.typing import Adj, Size
from torch_geometric.utils import to_undirected
from torch_scatter import scatter
from pytorch3d.structures import Meshes
from pytorch3d.ops import cot_laplacian
import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


class NativeFeaturePropagation(MessagePassing):
    def __init__(self, W:float=0.8, num_iterations: int=10):
        super().__init__(aggr="mean", flow="source_to_target")
        self.W = W
        self.num_iterations = num_iterations

    def message(self, x_i, x_j):
        return self.W*x_j + (1-self.W)*x_i

    def forward(self, x, edge_index, org_mask):
        edge_index = to_undirected(edge_index)
        org_mask = org_mask.view(-1,1)
        for _ in range(self.num_iterations):
            y = self.propagate( edge_index,x=x,org_mask=org_mask)
            x = (1-org_mask)*y + org_mask*x
        return x


class Laplacain(MessagePassing):
    def __init__(self):
        super().__init__(aggr="mean", flow="target_to_source")

    def message(self, x_i, x_j, w):
        return (x_i - x_j)*w
    
    def forward(self, x, edge_index, w):
        return self.propagate(edge_index, x=x, w=w.view(-1,1))


class Normal_consistence(MessagePassing):
    def __init__(self):
        super().__init__(aggr="add", flow="target_to_source")

    def message(self, x_i, x_j, w):
        return (1.0-F.cosine_similarity(x_i, x_j, dim=-1, eps=1e-6))*w
    
    def forward(self, vert_normal, edge_index, w):
        return self.propagate(edge_index, x=vert_normal, w=w.view(-1,1))


class Average_Smooth(MessagePassing):
    def __init__(self, ratio=1/8):
        super().__init__(aggr="mean", flow="target_to_source")
        self.ratio = ratio

    def message(self, x_i, x_j):
        return (1-self.ratio)*x_i+self.ratio*x_j
    
    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)
    

class LaplacianSmoothing(MessagePassing):
    def __init__(self):
        super(LaplacianSmoothing, self).__init__(aggr='add', flow='target_to_source')

    def mesh_smooth(self, meshes: Meshes, num_iterations: int=5):
        meshes_out = meshes.clone()
        for _ in range(num_iterations):
            cotweight, _ = cot_laplacian(meshes_out.verts_packed(), meshes_out.faces_packed())
            connection = cotweight.coalesce().indices()
            cotweight = cotweight.coalesce().values()
            connection,cotweight = to_undirected(connection, edge_attr=cotweight)
            new_verts = self.forward(x=meshes_out.verts_packed(),edge_index = connection, cot_weights=cotweight.view(-1,1))
            meshes_out = meshes_out.offset_verts_(new_verts - meshes_out.verts_packed())
        return meshes_out

    def forward(self, x, edge_index, cot_weights):
        cot_weights_mean = scatter(cot_weights, edge_index[0], dim=0, reduce='add')
        cot_weights = cot_weights / (cot_weights_mean.index_select(0,edge_index[0])+1e-6)
        x = self.propagate(edge_index, x=x, cot_weights=cot_weights)
        return x

    def message(self, x_i, x_j, cot_weights):
        return cot_weights.view(-1, 1) * x_j
        


    

