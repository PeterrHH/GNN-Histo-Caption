import os
import math
import numpy as np
import torch.nn.functional as F
#   Graph Message Passing
from torch_geometric.nn import GCNConv, GATConv, CuGraphSAGEConv, GINConv, PNAConv
#   Graph Pooling Opeartion
from torch_geometric.nn import dense_diff_pool, dense_mincut_pool




# pytorch
import torch
import torch.nn as nn

class GNNEncoder(nn.Module):
    def __init__(self, input_feat, cell_conv_method, tissue_conv_method, pool_method, num_layers, outpu_size = 256):
        super().__init__()
        self.conv_map = {
            "GCN": GCNConv,
            "GAT": GATConv,
            "GraphSage": CuGraphSAGEConv,
            "GIN": GINConv,
            "PNA":PNAConv
        }
        self.pool_map = {
            "Diff_Pool":dense_diff_pool,
            "MinCut":dense_mincut_pool,
        }
        
        self.num_layers = num_layers
        self.output_size = output_size
        self.selected_conv_method = None
        self.selected_pool_method = None
        self.layers = nn.ModuleList()
        self.graph_norms = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.pooling = nn.ModuleList()
        self.cg_to_tg_aggregate = None
        self.tg_readout_aggregate = None

        if cell_conv_method in self.conv_map:
            self.selected_cell_conv_method = conv_map[conv_method]
        if tissue_conv_method in self.conv_map:
            self.selected_tissue_conv_method = conv_map[conv_method]
   
        else:
            raise ValueError("No matching Graph Convolution methods")
        
        if pool_method in self.pool_map:
            self.selected_pool_method = pool_map[pool_method]
        '''
        Each Layer contains:
        - Message Passing
        - Graph Norm
        - Batch Norm
        - Pooling (optional)
        - Activation (We use RELU here)
        '''
        for _ in self.num_layers:
            self.layes.append(
                selected_conv_method(input_feat,dim)
            )
            self.graph_norms.append(GrpahNorm(dim))
            self.batch_norms.append(BatchNorm(dim))
            self.pooling.append(self.selected_pool_method(dim))

    def compute_assigned_feats(self, cg_feat,assignment_mat,tg_feat):
        """
        Use the assignment matrix to agg the feats
        Retur the tissue graph feature
        """
        #   Find what cells belong to each tissue

        #   Operationt to concet all cells belong to same tissue with some aggregation function
        
        #   Concat aggregation result with tissue feature
        column_numbers = np.argmax(assignment_mat, axis=1)
        summed_features = torch.zeros(tg_feat.shape)  #    summed_feature should have shape (num_tissue, feature size)
        #   Use torch.scatter_add to accumulate features based on class labels
        summed_features = torch.scatter_add(summed_features, dim=0, index=torch.from_numpy(column_numbers).unsqueeze(1), src=cg_feat)
        #   Add feature to cell_feature
        sum_tissue_feat = summed_features + tg_feat
        return sum_tissue_feat

    def readout(self,tg):
        '''
        Aggregate tissue graph feature to obtain a embedding that can be apssed into LSTM
        '''
        pass
    
    def forward(self,cell_graph,tissue_graph,assignment_mat):
        #   Cell Aggregation, gets all the features in a cell
        #   Original paper of HACT_NET use lstm connection
        cell_feat = cell_graph['feat']
        for i in range(self.num_layers):
            cell_feat = self.layers[i](cell_feat, edge_index)
            cell_feat = self.graph_norms[i](cell_feat)
            cell_feat = self.batch_norms[i](cell_feat)
            if selected_pooling_method:
                x = self.pooling[i](cell_feat)
            cell_feat = F.relu(cell_feat)
        
        #   Concat features of cell and tissue graph
        '''
        assignment_mat have shape (num_graph, num_cell_node_each_graph, num_tissue_node_each_graph)
        '''
        sum_tissue_feat = self.compute_assigned_feats(cg_feat,assignment_mat,tg_feat)

        #   Propogate Tissue Graph
        for i in range(self.num_layers):
            sum_tissue_feat = self.layers[i](sum_tissue_feat, edge_index)
            sum_tissue_feat = self.graph_norms[i](sum_tissue_feat)
            sum_tissue_feat = self.batch_norms[i](sum_tissue_feat)
            if selected_pooling_method:
                sum_tissue_feat = self.pooling[i](sum_tissue_feat)
            sum_tissue_feat = F.relu(sum_tissue_feat)

        #  readout a featurize 
        x = 1
        return x
