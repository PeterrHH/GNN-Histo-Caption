"""BRACS Dataset loader."""
import os
import h5py
import torch.utils.data
import numpy as np
from dgl.data.utils import load_graphs
from torch.utils.data import Dataset
from glob import glob 
import dgl 
import json
import sys
sys.path.append('../histocartography/histocartography')
sys.path.append('../histocartography')

from utils import set_graph_on_cuda



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
            load_in_ram: bool = False,
            ):
        # load data
        super(DiagnosticDataset, self).__init__()
        self.graph_path = graph_path
        self.split = split # Train Test Eval
        self.base_data_path = base_data_path
        self.load_in_ram = load_in_ram
        self.cg_path = os.path.join(self.graph_path,"cell_graphs",self.split)
        self.tg_path = os.path.join(self.graph_path,"tissue_graphs",self.split)
        self.assign_mat_path = os.path.join(self.graph_path,"assignment_mat",self.split)
        self.img_path = os.path.join(self.base_data_path,"Images", self.split)
        self.report_file_name = split+"_annotation.json"
        self.report_path = os.path.join(self.base_data_path,self.report_file_name)

        # Get list of cell graph
        self.cg = self.get_cell_graph()
        # Get list of tissue graph
        self.tg = self.get_tissue_graph()

        self.assign_mat = self.get_assign_mat()

        self.get_captions_labels(self.img_path,self.split)
        

    '''
    Get the captions and the labels
    Input:
    - List of image name
    - Split: train, test, eval
    '''
    def get_captions_labels(self,list_name, split):
        image_names = [os.path.splitext(os.path.split(i)[-1])[0]  for i in list_name]
        image_file_paths = [os.path.join(self.img_path,self.split,i) for i in image_names]
        with open(self.report_path, 'r') as json_file:
            report_data = json.load(json_file)
        self.captions = [report_data[key]['caption'] for key in new_file_names if key in report_data.keys()]
        self.labels = [report_data[key]['label'] for key in new_file_names if key in report_data.keys()]
    
    def get_cell_graph(self):
        self.list_cg_path = glob(os.path.join(self.cg_path, '*.bin'))
        self.list_cg_path.sort()
        self.num_cg = len(self.list_cg_path)
        cell_graphs = None
        if self.load_in_ram:
            cell_graphs = [load_graphs(single_cg_path) for single_cg_path in self.list_cg_path]
            cell_graphs = [entry[0][0] for entry in cell_graphs]
        print(cell_graphs)
        return cell_graphs
    
    def get_tissue_graph(self):
        self.list_tg_path = glob(os.path.join(self.tg_path, '*.bin'))
        self.list_tg_path.sort()
        self.num_tg = len(self.list_tg_path)
        tissue_graphs = None
        if self.load_in_ram:
            tissue_graphs = [load_graphs(single_tg_path) for single_tg_path in self.list_tg_path]
            tissue_graphs = [entry[0][0] for entry in tissue_graphs]
        return tissue_graphs

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

        captions = self.captions[index]
        labels = self.labels[index]
        # 1. Hierarchical Graphs
        if hasattr(self, 'num_tg') and hasattr(self, 'num_cg'):
            if self.load_in_ram:
                cg = self.cell_graphs[index]
                tg = self.tissue_graphs[index]
                assign_mat = self.assign_matrices[index]

                '''
                Issue: How to check and guarantee that the cg and tg and matched with the assign_mat for the same index
                '''
            else:
                cg, _ = load_graphs(self.list_cg_path[index])
                cg = cg[0]
                tg, _ = load_graphs(self.list_tg_path[index])
                tg = tg[0]
                assign_mat = h5_to_tensor(self.list_assign_path[index]).float().t()


            cg = set_graph_on_cuda(cg) if IS_CUDA else cg
            tg = set_graph_on_cuda(tg) if IS_CUDA else tg
            assign_mat = assign_mat.cuda() if IS_CUDA else assign_mat

            return cg,tg,assign_mat, captions, labels
        
        #   Use only tissue graph
        elif hasattr(self,'num_tg'):
            if self.load_in_ram:
                tg = self.tissue_graphs[index]
            else:
                tg, _ = load_graphs(self.list_tg_path[index])
                tg = tg[0]
            tg = set_graph_on_cuda(tg) if IS_CUDA else tg
            return tg, assign_mat, captions, labels

        #   Use only cell graph
        else:
            if self.load_in_ram:
                cg = self.cell_graphs[index]
            else:
                cg, _ = load_graphs(self.list_cg_path[index])
                cg = cg[0]
            cg = set_graph_on_cuda(cg) if IS_CUDA else cg
            return cg, assign_mat, captions, labels
    
    
    def __len__(self):
        print(f"CG LENGTH {len(self.cg)} and num is {self.num_cg}-------")
        print(f"TG LENGTH {len(self.tg)} and num is {self.num_tg}-------")
        assert len(self.cg) == len(self.tg)
        
        return len(self.cg)

def collate(batch):
    """
    Collate a batch.
    Args:
        batch (torch.tensor): a batch of examples.
    Returns:
        data: (tuple)
        labels: (torch.LongTensor)
    """
    def collate_fn(batch, id, type):
        return COLLATE_FN[type]([example[id] for example in batch])

    # collate the data
    num_modalities = len(batch[0])  # should 2 if CG or TG processing or 4 if HACT
    batch = tuple([collate_fn(batch, mod_id, type(batch[0][mod_id]).__name__)
                  for mod_id in range(num_modalities)])

    return batch
            # split: str = None,
            # base_data_path: str = None,
            # graph_path: str = None,
            # load_in_ram: bool = False
def make_dataloader(
        batch_size,
        split,
        base_data_path,
        graph_path,
        load_in_ram = False,
        shuffle=True,
        num_workers=0,
    ):
    """
    Create a BRACS data loader.
    """

    dataset = DiagnosticDataset(
                split = split,
                base_data_path = base_data_path,
                graph_path = graph_path,
                load_in_ram = load_in_ram
            )
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate
        )
    return dataloader

if __name__ == "__main__":
    loader = make_dataloader(
        batch_size = 2,
        split = "test",
        base_data_path = "../../Report-nmi-wsi",
        graph_path = "graph",
        shuffle=True,
        num_workers=0,
        load_in_ram = True
    )
    print(next(iter(loader)))
