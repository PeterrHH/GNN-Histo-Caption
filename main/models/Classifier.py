import torchvision
import torch
import torch.nn as nn

'''
Simple ResNet34
'''
class Classifier(nn.Module):
    def __init__(self,graph_output_size,global_output_size,hidden_size,num_class,dropout_rate = 0.3):
        super().__init__()

        #Set input size base on specific use case
        self.input_size = graph_output_size + global_output_size
        '''
        self.input_size = global_output_size
        '''
        self.dropout_rate = dropout_rate
        self.fc = nn.Sequential(
            nn.Linear(self.input_size, hidden_size),  
            nn.ReLU(),    
            nn.Dropout(dropout_rate),            
            nn.Linear(hidden_size,num_class)  
        )


    def forward(self,x):
        return self.fc(x).squeeze()