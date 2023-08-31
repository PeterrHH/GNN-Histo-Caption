import os
import math
import numpy as np
import torch.nn.functional as F
#   Graph Message Passing
from torch_geometric.nn import GCNConv, GATConv, CuGraphSAGEConv, GINConv, PNAConv
#   Graph Pooling Opeartion
from torch_geometric.nn import dense_diff_pool, dense_mincut_pool
from torch_geometric.nn.norm import GraphNorm, BatchNorm
import dgl
import h5py



# pytorch
import torch
import torch.nn as nn
'''
Output: torch.Tensor([batch_size,feature_bedding_size])
'''
class GNNEncoder(nn.Module):
    def __init__(self, cell_conv_method, tissue_conv_method, pool_method, num_layers, aggregate_method, input_feat = 514,output_size = 128):
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
        
        self.input_feat = input_feat
        self.aggregate_method = aggregate_method
        self.num_layers = num_layers
        self.output_size = output_size
        self.selected_cell_conv_method = None
        self.selected_tissue_conv_method = None
        self.selected_pool_method = None
        self.cell_layers = nn.ModuleList()
        self.tissue_layers = nn.ModuleList()
        self.cell_batch_norms = nn.ModuleList()
        self.tissue_batch_norms = nn.ModuleList()
        self.cell_graph_norms = nn.ModuleList()
        self.tissue_graph_norms = nn.ModuleList()
        self.pooling = nn.ModuleList()
        self.cg_to_tg_aggregate = None
        self.tg_readout_aggregate = None

        if cell_conv_method in self.conv_map:
            self.selected_cell_conv_method = self.conv_map[cell_conv_method]
        if tissue_conv_method in self.conv_map:
            self.selected_tissue_conv_method = self.conv_map[cell_conv_method]
        if tissue_conv_method in self.conv_map:
            self.selected_tissue_conv_method = self.conv_map[tissue_conv_method]
   
        else:
            raise ValueError("No matching Graph Convolution methods")
        
        if pool_method in self.pool_map:
            self.selected_pool_method = self.pool_map[pool_method]
        '''
        Each Layer contains:
        - Message Passing
        - Graph Norm
        - Batch Norm
        - Pooling (optional)
        - Activation (We use RELU here)
        '''

        for _ in range(self.num_layers):
            self.cell_layers.append(
                self.selected_cell_conv_method(self.input_feat,self.input_feat)
            )
            self.cell_graph_norms.append(GraphNorm(self.input_feat))
            self.cell_batch_norms.append(BatchNorm(self.input_feat))
            if pool_method:
                self.cell_layers.append(self.selected_pool_method(self.input_feat))

        for _ in range(self.num_layers):
            self.tissue_layers.append(
                self.selected_tissue_conv_method(self.input_feat*2,self.input_feat*2)
            )
            self.tissue_graph_norms.append(GraphNorm(self.input_feat*2))
            self.tissue_batch_norms.append(BatchNorm(self.input_feat*2))
        self.linear = nn.Linear(self.input_feat*2, self.output_size)

    def compute_assigned_feats(self, cg_feat,cell_graph,assignment_mat,tg_feat):
        """
        Use the assignment matrix to agg the feats
        Retur the tissue graph feature
        """   
        num_nodes_per_graph = cell_graph.batch_num_nodes().tolist()
        num_nodes_per_graph.insert(0, 0)
        intervals = [sum(num_nodes_per_graph[:i + 1])
                     for i in range(len(num_nodes_per_graph))]

        ll_h_concat = []
        for i in range(1, len(intervals)):
            h_agg = torch.matmul(
                assignment_mat[i - 1].t(), cg_feat[intervals[i - 1]:intervals[i], :]
            )
            ll_h_concat.append(h_agg)
       # print(f"ll_h_contact size is {len(ll_h_concat)}")
       # print(f"shape within 0: {ll_h_concat[0].shape} and 1 {ll_h_concat[1].shape} and tissue_feat here is {tg_feat.shape}")
        cat = torch.cat(ll_h_concat, dim=0)
       # print(f"cat size {cat.shape}")
       # print(f"tg_feat size {tg_feat.shape}")
        return torch.cat((cat, tg_feat), dim=1)


    def readout(self,tissue_graph):
        '''
        Aggregate tissue graph feature to obtain a embedding that can be apssed into LSTM
        '''
        # tissue_feat = tg['feat']
        if self.aggregate_method == "mean":
            aggregated_features = dgl.mean_nodes(tissue_graph,'h')
        else:
            #   sum
            aggregated_features = dgl.sum_nodes(tissue_graph,'h')
        return aggregated_features
    
    def forward(self,cell_graph,tissue_graph,assignment_mat):
        #   Cell Aggregation, gets all the features in a cell
        #   Original paper of HACT_NET use lstm connection
        cell_feat = cell_graph.ndata['feat']
        cell_edge = torch.stack(cell_graph.edges())
        tissue_edge = torch.stack(tissue_graph.edges())
        tissue_feat = tissue_graph.ndata['feat']
        # print(f"Cell feaeture {cell_feat.shape}")
        # print(f"Tissue feaeture {tissue_feat.shape}")
        for i in range(self.num_layers):
            cell_feat = self.cell_layers[i](cell_feat,cell_edge)
            cell_feat = self.cell_graph_norms[i](cell_feat)
            cell_feat = self.cell_batch_norms[i](cell_feat)
            if self.selected_pool_method:
                cell_feat = self.pooling[i](cell_feat)
            cell_feat = F.relu(cell_feat)
            cell_graph.ndata['h'] = cell_feat
        # print(f"Cell Feature after propagation {cell_feat.shape}")
        #   Concat features of cell and tissue graph
        '''
        assignment_mat have shape (num_graph, num_cell_node_each_graph, num_tissue_node_each_graph)
        '''
        sum_tissue_feat = self.compute_assigned_feats(cell_feat,cell_graph,assignment_mat,tissue_feat)
        # print(f"--------Sum Tissue Feat-----------")
        # print(type(sum_tissue_feat))
        # print(f"Cell Feat: {cell_feat.shape}")
        # print(f"Tissue Feat: {tissue_feat.shape}")
        # print(f"Sum tissue feature {sum_tissue_feat.shape}")
        # print(f"--------Sum Tissue Feat-----------")
        #   Propogate Tissue Graph
        for i in range(self.num_layers):
            sum_tissue_feat = self.tissue_layers[i](sum_tissue_feat, tissue_edge)
            sum_tissue_feat = self.tissue_graph_norms[i](sum_tissue_feat)
            sum_tissue_feat = self.tissue_batch_norms[i](sum_tissue_feat)
            if self.selected_pool_method:
                sum_tissue_feat = self.pooling[i](sum_tissue_feat)
            sum_tissue_feat = F.relu(sum_tissue_feat)
        sum_tissue_feat = self.linear(sum_tissue_feat)
        #  readout a featurize 
        #print(f"Before readout shape {sum_tissue_feat.shape}")
        tissue_graph.ndata['h'] = sum_tissue_feat
        aggregate_feat = self.readout(tissue_graph)
        return aggregate_feat.unsqueeze(1)


if __name__ == "__main__":
    def h5_to_tensor(h5_path):
        h5_object = h5py.File(h5_path, 'r')
        out = torch.from_numpy(np.array(h5_object['assignment_matrix']))
        return out
    # cell_path = "graph/cell_graphs/test/1_00061_sub0_002.bin"
    # tissue_path = "graph/tissue_graphs/test/1_00061_sub0_002.bin"
    # assignment_path = "graph/assignment_mat/test/1_00061_sub0_002.h5"


    # c_graphs = dgl.load_graphs(cell_path)
    # t_graphs = dgl.load_graphs(cell_path)
    # cg = c_graphs[0][0]
    # tg = t_graphs[0][0]
    # assign_mat = h5_to_tensor(assignment_path)
    import sys
    sys.path.append('../main')
    from dataloader import make_dataloader 
    loader = make_dataloader(
        batch_size = 4,
        split = "test",
        base_data_path = "../../Report-nmi-wsi",
        graph_path = "graph",
        vocab_path = "../../Report-nmi-wsi/vocab_bladderreport.pkl",
        shuffle=True,
        num_workers=0,
        load_in_ram = True
    )
    #a = next(iter(loader))
    for batched_idx, batch_data in loader:
        cg, tg, assign_mat, caption_tokens, label = batch_data  
        encoder = GNNEncoder(cell_conv_method = "GCN", tissue_conv_method = "GCN", pool_method = None, num_layers = 3, aggregate_method = "sum", input_feat = 514,output_size = 256)
        out = encoder(cg,tg,assign_mat)
        print(f"length is {len(assign_mat)}")
        print(f"Outputshape is {out.shape}")
        break

