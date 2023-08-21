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
    def __init__(self):
        super().__init__()
        self.gnn = GNNEncoder()
        self.LSTM = LSTMDecoder()
    def forward(self,images,captions):
        features = self.gnn(images)
        outputs = self.LSTM(features,captions)

