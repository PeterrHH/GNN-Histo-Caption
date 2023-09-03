import os
import math
import numpy as np
import torch.nn.functional as F
#   Graph Message Passing
from dgl.nn import GraphConv, GATConv, SAGEConv, GINConv, PNAConv, MaxPooling, AvgPooling, SumPooling
#from torch_geometric.nn import GCNConv, GATConv, CuGraphSAGEConv, GINConv, PNAConv
#   Graph Pooling Opeartion
from torch_geometric.nn import dense_diff_pool, dense_mincut_pool
from torch_geometric.nn.norm import GraphNorm, BatchNorm
import dgl
import h5py
import sys
from Vocabulary import Vocabulary



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
            "GCN": GraphConv,
            "GAT": GATConv,
            "GraphSage": SAGEConv,
            # "GIN": GINConv,
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
   
        else:
            raise ValueError("No matching Graph Convolution methods")
        
        if pool_method in self.pool_map:
            self.selected_pool_method = self.pool_map[pool_method]
        self.cell_layer = nn.ModuleList()
        self.tissue_layer = nn.ModuleList()
        for _ in range(1):
            self.cell_layer.append(
                nn.Sequential(
                    GraphConv(514,514),
                    GraphNorm(514),
                    BatchNorm(514)
                )
            )
        for _ in range(3):
            self.tissue_layer.append(
                nn.Sequential(
                    self.selected_tissue_conv_method(1028,1028),
                    GraphNorm(1028),
                    BatchNorm(1028)
                )
            )
        self.conv1 = GraphConv(514,514)
        self.gn = GraphNorm(514)
        self.bn = BatchNorm(514)
        self.cell_lin_out = nn.Linear(514,256)
        # self.conv2 = GraphConv(514,514)
        # self.maxpool = MaxPooling()
 
        # self.t_conv1 = GraphConv(514,514)
        # self.gn = GraphNorm(514)
        # self.bn = BatchNorm(514)
        # self.conv2 = GraphConv(514,514)
        self.maxpool = MaxPooling()
        self.lin_out = nn.Linear(1028,256)
        '''
        Each Layer contains:
        - Message Passing
        - Graph Norm
        - Batch Norm
        - Pooling (optional)
        - Activation (We use RELU here)
        '''

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


    def readout(self,tissue_graph,x):
        '''
        Aggregate tissue graph feature to obtain a embedding that can be apssed into LSTM
        '''
        # tissue_feat = tg['feat']
        if self.aggregate_method == "mean":
            pool = AvgPooling()
        elif self.aggregate_method == "max":
            pool = MaxPooling()
        else:
            #   sum
            pool = SumPooling()
        aggregated_features = pool(tissue_graph,x)
        return aggregated_features
    
    def forward(self,cell_graph,tissue_graph,assignment_mat):

        # #   Cell Aggregation, gets all the features in a cell
        # #   Original paper of HACT_NET use lstm connection
        cell_feat = cell_graph.ndata['feat']
        cell_edge = torch.stack(cell_graph.edges())
        tissue_edge = torch.stack(tissue_graph.edges())
        tissue_feat = tissue_graph.ndata['feat']

        # for layer in self.cell_layer:
        #     cell_feat = layer[0](cell_graph, cell_feat)
        #     cell_feat = layer[1](cell_feat)
        #     cell_feat = layer[2](cell_feat)
        cell_feat = self.conv1(cell_graph,cell_feat)
        cell_feat = self.gn(cell_feat)
        cell_feat = self.bn(cell_feat)
        x = self.readout(cell_graph,cell_feat)
        x = self.cell_lin_out(x)
        # cell_feat = F.relu(cell_feat)
        print("-------------cell_feat-------------")
        print(cell_feat)
        print(f"Cell feat shape {cell_feat.shape}")
        print(f"X shape {x.shape}")
        print("-------------cell_feat ------------")
        # x = self.compute_assigned_feats(cell_feat,cell_graph,assignment_mat,tissue_feat)
        # print(f"SUM TISSUE FEAT SHAPE {x.shape}")
        # for layer in self.tissue_layer:
        #     x = layer[0](tissue_graph,x)
        #     x = layer[1](x)
        #     x = layer[2](x)
        # # x = F.relu(x)
        # x = self.readout(tissue_graph,x)
        # x = self.lin_out(x)
        return x.unsqueeze(1)
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import networkx as nx
    def h5_to_tensor(h5_path):
        h5_object = h5py.File(h5_path, 'r')
        out = torch.from_numpy(np.array(h5_object['assignment_matrix']))
        return out
    import sys
    sys.path.append('../main')
    from dataloader import make_dataloader 
    loader = make_dataloader(
        batch_size = 16,
        split = "test",
        base_data_path = "../../Report-nmi-wsi",
        graph_path = "graph",
        vocab_path = "../../Report-nmi-wsi/vocab_bladderreport.pkl",
        shuffle=True,
        num_workers=0,
        load_in_ram = True
    )
    #a = next(iter(loader))
    for batched_idx, batch_data in enumerate(loader):
        cg, tg, assign_mat, caption_tokens, label,caption = batch_data  
        encoder = GNNEncoder(cell_conv_method = "GCN", tissue_conv_method = "GCN", pool_method = None, num_layers = 3, aggregate_method = "sum", input_feat = 514,output_size = 256)
        # out = encoder(cg,tg,assign_mat)
        # print(f"length is {len(assign_mat)}")
        # print(f"Outputshape is {out.shape}")
        # print(encoder)
        # GCNConv()
        # print(out)
        print(cg.ndata['feat'].shape)
        print(cg.ndata.keys())
        # print(cg.shape)
        # GCN = GraphConv(514,514)
        # # Check for NaN values
        # nan_mask = torch.isnan(cg.ndata['feat'])

        # # Count the number of NaN values
        # num_nan_values = torch.sum(nan_mask).item()

        # print("Number of NaN values in the tensor:", num_nan_values)
        # cell_edge = torch.stack(cg.edges())
        # out = GCN(cg,cg.ndata['feat'])
        # print(out)
        # nan_mask = torch.isnan(out)
        glist = dgl.unbatch(cg)
        # Count the number of NaN values
        # num_nan_values = torch.sum(nan_mask).item()

        # print("Number of NaN values in the out:", out)
        fig, ax = plt.subplots()
        nx.draw(glist[0].to_networkx(), ax=ax)
        ax.set_title('Visualise')
        plt.show()

        break

