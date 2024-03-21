import os
import math
import numpy as np


# pytorch
import torch
import torch.nn as nn

#
from .Graph_Model import GNNEncoder
from .Transformer import LSTMDecoder




class GNN_LSTM(nn.Module):
    def __init__(self,hidden_dim, vocab_size,gnn_param, lstm_param, phase):
        super().__init__()
        self.phase = phase
        self.encoder =  GNNEncoder(
            input_feat = 514,   #   Input feature size for each node
            cell_conv_method = gnn_param["cell_conv_method"],
            tissue_conv_method = gnn_param["tissue_conv_method"],
            pool_method = gnn_param["pool_method"],
            num_layers = gnn_param["num_layers"],
            aggregate_method = gnn_param["aggregate_method"]
        )

        self.decoder = LSTMDecoder(
            dropout = lstm_param["dropout"],
            vocab_size = vocab_size
        )

    def forward(self,cg,tg,assign_mat,captions):
        features = self.encoder(cg,tg,assign_mat)
        outputs = self.decoder(features,captions)
        return outputs

if __name__ == "__main__":
    def h5_to_tensor(h5_path):
        h5_object = h5py.File(h5_path, 'r')
        out = torch.from_numpy(np.array(h5_object['assignment_matrix']))
        return out
    cell_path = "graph/cell_graphs/test/1_00061_sub0_002.bin"
    tissue_path = "graph/tissue_graphs/test/1_00061_sub0_002.bin"
    assignment_path = "graph/assignment_mat/test/1_00061_sub0_002.h5"


    c_graphs = dgl.load_graphs(cell_path)
    t_graphs = dgl.load_graphs(cell_path)
    cg = c_graphs[0][0]
    tg = t_graphs[0][0]
    assign_mat = h5_to_tensor(assignment_path)
    gnn_param = {
        "cell_conv_method":"GCN",
        "tissue_conv_method":"GCN",
        "pool_method":None,


    }
    encoder = GNNEncoder(cell_conv_method = "GCN", tissue_conv_method = "GCN", pool_method = None, num_layers = 3, aggregate_method = "sum", input_feat = 514,output_size = 256)
    out = encoder(cg,tg,assign_mat)
    print(f"Outputshape is {out.shape}")

