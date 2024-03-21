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
        self.feature_extractor = torchvision.models.resnet18(pretrained=True)
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
        # for param in self.feature_extractor.layer4.parameters():
        #     param.requires_grad = True
        self.img_downsample = nn.Linear(1000,self.output_size)
        self.dropout = nn.Dropout(p=self.dropout_rate) 
        self.batch= nn.BatchNorm1d(output_size,momentum = 0.01)

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
        img_feat = self.batch(img_feat)
        # img_feat = F.sigmoid(img_feat)
        return img_feat
    
if __name__ == "__main__":
    gf = GlobalFeatureExtractor(256,256)
    for name, param in gf.named_parameters():
        print(f"Layer: {name}, Trainable: {param.requires_grad}")