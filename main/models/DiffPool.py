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
from utils import fetch_assign_matrix, GCNConv

NUM_SAGE_LAYERS = 3


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
        batch_size, num_nodes, in_channels = x.size()

        x0 = x
        x1 = self.bn(1, F.relu(self.conv1(x0, adj, mask, add_loop=False)))
        x2 = self.bn(2, F.relu(self.conv2(x1, adj, mask, add_loop=False)))
        x3 = self.conv3(x2, adj, mask, add_loop=False)

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
