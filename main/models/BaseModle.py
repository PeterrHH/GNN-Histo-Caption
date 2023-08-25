import os
import math
import numpy as np


# pytorch
import torch
import torch.nn as nn

#
from .Graph_Model import GNNEncoder
from .LSTM import LSTMDecoder




class GNN_LSTM_cap(nn.Module):
    def __init__(self,encoder, decoder, cg,tg,hidden_dim, vocab_size,gnn_param, lstm_param):
        super().__init__()
        self.encoder =  GNNEncoder(
            input_feat = 514,   #   Input feature size for each node
            cell_layers = gnn_param["cell_layers"],
            tissue_layer = gnn_param["tissue_layers"],
            cell_conv_method = gnn_param["cell_conv_method"],
            tissue_conv_method = gnn_param["tissue_conv_method"],
            pool_method = gnn_param["pool_method"],
        )

        self.decoder = LSTMDecoder(
            dropout = lstm_param["dropout"],
            vocab_size = vocab_size
        )

    def forward(self,images,captions):
        features = self.encoder(images)
        outputs = self.decoder(features,captions)
        return outputs

