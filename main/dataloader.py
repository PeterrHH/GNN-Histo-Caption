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
            split: str = None,
            base_data_path: str = None,
            graph_path: str = None,
            load_in_ram: bool = False,):
        # load data
        super(DiagnosticDataset, self).__init__()
        self.graph_path = graph_path
        self.split = split # Train Test Eval
        self.cg_path = os.path.join(self.graph_path,"cell_graphs",self.split)
        self.tg_path = os.path.join(self.graph_path,"tissue_graph",self.split)
        self.assign_mat_path = os.path.join(self.graph_path,"assignment_mat",self.split)
        self.img_path = os.path.join(self.base_data_path,"Images", self.split)
        self.report_file_name = split+"_annoation.json"
        self.report_path = os.path.join(self.base_data_path,report_file_name)

        # Get list of cell graph
        self.cg = self.get_cell_graph()
        # Get list of tissue graph
        self.tg = self.get_tissue_graph()

        self.assign_mat = self.get__assign_mat()
        

    '''
    Get the captions and the labels
    '''
    def get_captions_labels(list_name, split):
        image_names = [os.path.splitext(os.path.split(i)[-1])[0]  for i in list_name]
        image_file_paths = [os.path.join(self.img_path,self.split,i) for i in image_names]
        with open(self.report_path, 'r') as json_file:
            report_data = json.load(json_file)
        captions = [report_data[key]['caption'] for key in new_file_names if key in report_data.keys()]
        labels = [report_data[key]['label'] for key in new_file_names if key in report_data.keys()]
        return captions, labels
    
    def get_cell_graph(self):
        self.list_cg_path = glob(os.path.join(self.cg_path, '*.bin'))
        self.list_cg_path.sort()
        self.num_cg = len(self.list_cg_path)

        if self.load_in_ram:
            cell_graphs = [load_graphs(single_cg_path) for single_cg_path in self.list_cg_path]
            self.cell_graphs = [entry[0][0] for entry in cell_graphs]
    
    def get_tissue_graph(self):
        self.list_tg_path = glob(os.path.join(self.tg_path, '*.bin'))
        self.list_tg_path.sort()
        self.num_tg = len(self.list_tg_path)

        if self.load_in_ram:
            tissue_graphs = [load_graphs(single_tg_path) for single_tg_path in self.list_tg_path]
            self.tissue_graphs = [entry[0][0] for entry in tissue_graphs]

    def get_assign_mat(self):
        """
        Load assignment matrices 
        """
        self.list_assign_path = glob(os.path.join(self.assign_mat_path, '*.h5'))
        self.list_assign_path.sort()
        self.num_assign_mat = len(self.list_assign_path)
        if self.load_in_ram:
            self.assign_matrices = [
                h5_to_tensor(single_assign_path).float().t()
                    for single_assign_path in self.list_assign_path
            ]
    
    def __getitem__(self,index):
        # return the cell graph, tissue graph, assignment matrix and the relevant 5 captions
        if self.cg and self.tg:
            return self.hg
        pass
    
    
    def __len__(self):
        assert len(self.cg) == len(self.tg)
        
        return len(self.cg)