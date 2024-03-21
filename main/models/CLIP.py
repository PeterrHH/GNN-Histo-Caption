from IPython import display as ipythondisplay
from torch import nn
from tqdm.autonotebook import tqdm
from transformers import AutoTokenizer, AutoModel
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer
import cv2
import gc
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
import torch.nn.functional as F
from models.Graph_Model import GNNEncoder
from models.LSTM2 import LSTMDecoder
from models.GlobalFeatureExtractor import GlobalFeatureExtractor



'''

ClinicalBert, BioBErt,Bert
'''

class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

class TextEncoder(nn.Module):
    def __init__(self, trainable=True):
        super().__init__()

        #     # self.model = DistilBertModel.from_pretrained(model_name)

        #     # Use Bio-ClinicalBERT
        #     self.model = AutoModel.from_pretrained(CFG.clinical_encoder_model)

        # else:
        #     self.model = DistilBertModel(config=DistilBertConfig())

        self.tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
        self.model = AutoModel.from_pretrained("medicalai/ClinicalBERT")
                    
        for p in self.model.parameters():
            p.requires_grad = trainable

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids,attention_masks):

        output = self.model(input_ids=input_ids,attention_mask = attention_masks)
        last_hidden_state = output.last_hidden_state
 
        return last_hidden_state[:, self.target_token_idx, :]

# Get both image and text encodings into a same size matrix
class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim,
        dropout = 0.5
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class CLIPModel(nn.Module):
    def __init__(
        self,
        args,
        device
    ):
        super().__init__()
        self.graph_encoder, self.feature_extractor = self.model_def(args,device)
        self.text_encoder = TextEncoder(trainable=True)
        self.temperature = 1.0
        self.image_projection = ProjectionHead(embedding_dim=512,projection_dim=512)
        self.text_projection = ProjectionHead(embedding_dim=768,projection_dim=512)
        # self.temperature = temperature

    '''
    caption input shape is torch.Size([bs, len])
    '''
    def forward(self, cg,tg,assign_mat,image,caption_token,attention_masks):
        # Getting Image and Text Features

        graph_out = self.graph_encoder(cg,tg,assign_mat,image)
        global_feat = self.feature_extractor(image)
        merged_feat = torch.cat((graph_out, global_feat), dim=1)
        #print(f"caption token shape {caption_token.shape}")
        text_features = self.text_encoder(
            caption_token,attention_masks
        )
        #print(f"merged_feature shape {merged_feat.shape} text features shape {text_features.shape}")
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(merged_feat)
        text_embeddings = self.text_projection(text_features)

        # Calculating the Loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = self.cross_entropy(logits, targets, reduction='none')
        images_loss = self.cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean()

    def cross_entropy(self,preds, targets, reduction='none'):
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()
        
    def model_def(self,args,device):
        '''
        decoder: transformer/lstm
        '''
            #   Define Model, Loss and 
        encoder = GNNEncoder(
            args = args,
            cg_layer = 2, 
            tg_layer = 1,
            aggregate_method = "mean", 
            input_feat = 514,
            hidden_size = 256,
            output_size = 256,
        ).to(device)

        # attention = EncoderLayer(d_model = args['gnn_param']['output_size'], 
        #     nhead = 4, 
        #     dim_feedforward = 1024, 
        #     dropout = 0.2).to(device)
        global_feature_extractor = GlobalFeatureExtractor(
            hidden_size =256,
            output_size = 256,
            dropout_rate = 0.5).to(device)


        return encoder, global_feature_extractor



if __name__ == "__main__":
    textencoder = TextEncoder()
