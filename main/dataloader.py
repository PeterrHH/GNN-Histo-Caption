"""BRACS Dataset loader."""
import os
import h5py
import torch.utils.data
import numpy as np
from dgl.data.utils import load_graphs
from torch.utils.data import Dataset
from glob import glob 
import dgl 


from histocartography.utils import set_graph_on_cuda


IS_CUDA = torch.cuda.is_available()
DEVICE = 'cuda:0' if IS_CUDA else 'cpu'
COLLATE_FN = {
    'DGLGraph': lambda x: dgl.batch(x),
    'Tensor': lambda x: x,
    'int': lambda x: torch.LongTensor(x).to(DEVICE)
}

def h5_to_tensor(h5_path):
    h5_object = h5py.File(h5_path, 'r')
    out = torch.from_numpy(np.array(h5_object['assignment_matrix']))
    return out


class DiagnosticDataset(Dataset):
    """NMI-WSI dataset dataset."""
    
    def __init__(self,
            cg_path: str = None,
            tg_path: str = None,
            assign_mat_path: str = None,
            img_path: str = None,
            cap_path: str = None,
            load_in_ram: bool = False,):
        # load data
        super(DiagnosticDataset, self).__init__()
        
        self.cg_path = cg_path
        self.tg_path = tg_path
        self.assign_mat_path = assign_mat_path
        self.img_path = img_path
        self.cap_path = cap_path
        
        # Get list of cell graph
        self.cg = self.get_cell_graph(self.cg_path)
        # Get list of tissue graph
        self.tg = self.get_tissue_graph(self.tg_path)
        
        
        pass
    
    def get_cell_graph(cg_path):
        return None
    
    def get_tissue_graph(tg_graph):
        return None
    
    def __getitem__(self,index):
        # return the cell graph, tissue graph, assignment matrix and the relevant 5 captions
        if self.cg and self.tg:
            return self.hg
        pass
    
    
    def __len__(self):
        assert len(self.cg) == len(self.tg)
        
        return len(self.cg)