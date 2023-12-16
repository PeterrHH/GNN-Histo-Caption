import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
'''
Simple ResNet34
'''
class GlobalFeatureExtractor(nn.Module):
    def __init__(self,hidden_size,output_size,dropout_rate = 0.3):
        super().__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.feature_extractor = torchvision.models.resnet34(pretrained=True)
        self.img_downsample = nn.Linear(1000,self.output_size)
        self.dropout = nn.Dropout(p=self.dropout_rate) 

        # self.img_downsample = nn.Sequential(
        #     nn.Linear(input_size, hidden_size),  
        #     nn.ReLU(),    
        #     nn.Dropout(dropout_rate),            
        #     nn.Linear(hidden_size,num_class)  
        # )

    def forward(self,image):
        img_feat = self.feature_extractor(image)
        img_feat = self.img_downsample(img_feat)
        img_feat = self.dropout(img_feat)
        # img_feat = F.sigmoid(img_feat)
        return img_feat