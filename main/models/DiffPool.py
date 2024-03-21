from torch_geometric.nn import dense_diff_pool, dense_mincut_pool
import torch
import torch.nn

from math import ceil

import torch
from ogb.graphproppred.mol_encoder import AtomEncoder
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch_geometric.nn import DenseGraphConv
from torch_geometric.nn import MessagePassing
# from utils import fetch_assign_matrix, GCNConv

NUM_SAGE_LAYERS = 3

def fetch_assign_matrix(random, dim1, dim2, normalize=False):
    if random == 'uniform':
        m = torch.rand(dim1, dim2)
    elif random == 'normal':
        m = torch.randn(dim1, dim2)
    elif random == 'bernoulli':
        m = torch.bernoulli(0.3*torch.ones(dim1, dim2))
    elif random == 'categorical':
        idxs = torch.multinomial((1.0/dim2)*torch.ones((dim1, dim2)), 1)
        m = torch.zeros(dim1, dim2)
        m[torch.arange(dim1), idxs.view(-1)] = 1.0

    if normalize:
        m = m / (m.sum(dim=1, keepdim=True) + EPS)
    return m

# class GCNConv(MessagePassing):
#     def __init__(self, emb_dim, aggr):
#         super(GCNConv, self).__init__(aggr=aggr)

#         self.linear = torch.nn.Linear(emb_dim, emb_dim)
#         self.root_emb = torch.nn.Embedding(1, emb_dim)
#         self.bond_encoder = BondEncoder(emb_dim = emb_dim)

#     def forward(self, x, edge_index, edge_attr):
#         x = self.linear(x)
#         edge_embedding = self.bond_encoder(edge_attr)
#         return self.propagate(edge_index, x=x, edge_attr=edge_embedding) + F.relu(x + self.root_emb.weight)

#     def message(self, x_j, edge_attr):
#         return F.relu(x_j + edge_attr)

#     def update(self, aggr_out):
#         return aggr_out

class SAGEConvolutions(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 lin=True):
        super().__init__()

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.conv3 = DenseSAGEConv(hidden_channels, out_channels)

        if lin is True:
            self.lin = nn.Linear((NUM_SAGE_LAYERS - 1) * hidden_channels + out_channels, out_channels)
        else:
            # GNN's intermediate representation is given by the concatenation of SAGE layers
            self.lin = None

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, 'bn{}'.format(i))(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        print(f"x shape {x.shape}")
        batch_size, num_nodes, in_channels = x.size()

        x0 = x
        x1 = self.bn(1, F.relu(self.conv1(x0, adj, mask)))
        x2 = self.bn(2, F.relu(self.conv2(x1, adj, mask)))
        x3 = self.conv3(x2, adj, mask)

        x = torch.cat([x1, x2, x3], dim=-1)

        # This is used by GNN_pool
        if self.lin is not None:
            x = self.lin(x)

        return x


class DiffPoolLayer(nn.Module):

    def __init__(self, dim_input, dim_hidden, dim_embedding, current_num_clusters,
                 no_new_clusters, pooling_type, invariant):

        super().__init__()
        self.pooling_type = pooling_type
        self.invariant = invariant
        if pooling_type != 'gnn':
            if self.invariant:
                self.rm = fetch_assign_matrix(pooling_type, dim_input, no_new_clusters)
            else:
                self.rm = fetch_assign_matrix(pooling_type, current_num_clusters, no_new_clusters)
            self.rm.requires_grad = False
        self.gnn_pool = SAGEConvolutions(dim_input, dim_hidden, no_new_clusters)
        self.gnn_embed = SAGEConvolutions(dim_input, dim_hidden, dim_embedding, lin=False)

    def forward(self, x, adj, mask=None):

        if self.pooling_type == 'gnn':
            s = self.gnn_pool(x, adj, mask)
        else:
            if self.invariant:
                s = self.rm.unsqueeze(dim=0).expand(x.size(0), -1, -1)
                s = s.to(x.device)
                s = x.detach().matmul(s)
            else:
                s = self.rm[:x.size(1), :].unsqueeze(dim=0)
                s = s.expand(x.size(0), -1, -1)
                s = s.to(x.device)
        x = self.gnn_embed(x, adj, mask)

        x, adj, l, e = dense_diff_pool(x, adj, s, mask)
        return x, adj, l, e

if __name__ == "__main__":
    pass