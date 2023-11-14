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
import nltk
#nltk.download('punkt')
import pickle
import random
import torch
from PIL import Image
import torch.nn as nn
import torchvision.transforms as transforms
sys.path.append('../histocartography/histocartography')
sys.path.append('../histocartography')

from utils import set_graph_on_cuda
from Vocabulary import Vocabulary



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
            vocab_path: str = None,
            load_all: bool = True,
            load_in_ram: bool = False,
            mode: str = "train", # train, eval
            ):
        # load data
        super(DiagnosticDataset, self).__init__()
        self.graph_path = graph_path
        self.split = split # Train Test Eval
        self.mode = mode
        if split != "train":
            self.mode = "eval"
        self.base_data_path = base_data_path
        self.load_all = load_all
        self.load_in_ram = load_in_ram
        self.vocab_path = vocab_path
        self.cg_path = os.path.join(self.graph_path,"cell_graphs",self.split)
        self.tg_path = os.path.join(self.graph_path,"tissue_graphs",self.split)
        self.assign_mat_path = os.path.join(self.graph_path,"assignment_mat",self.split)
        self.img_path = os.path.join(self.base_data_path,"Images", self.split)
        self.report_file_name = split+"_annotation.json"
        self.report_path = os.path.join(self.base_data_path,self.report_file_name)
        self.vocab = pickle.load(open(self.vocab_path,'rb'))
        # self.vocab.add_word('<start>')
        # self.START_TOKEN = self.vocab.word2idx['<start>']
        self.END_TOKEN = self.vocab.word2idx['<end>']
        self.PAD_TOKEN = self.vocab.word2idx['<pad>'] # PAD_TOKEN is used for not supervison
        self.START_TOKEN = self.vocab.word2idx['<start>']
        self.max_length = 90
        self.vocab_size = len(self.vocab.word2idx)
        self.num_feature = 6
        self.max_subseq_len = 15+1
        self.stop_label = 2

        # Get list of cell graph
        self.cg = self.get_cell_graph()
        # Get list of tissue graph
        self.tg = self.get_tissue_graph()

        self.assign_mat = self.get_assign_mat()

        self.get_captions_labels(self.img_path,self.split)
        self.img = self.get_img(self.img_path,self.split)


    '''
    Get the captions and the labels
    Input:
    - List of image name
    - Split: train, test, eval
    '''
    def get_img(self,img_path,split):
        #print(os.path.join(img_path,"Images",split,"*.png"))
        img_list =  glob(os.path.join(img_path,"*.png"))
        img_list.sort()
        img_name = [os.path.splitext(os.path.split(i)[-1])[0]  for i in img_list]
        graph_name = [os.path.splitext(os.path.split(i)[-1])[0]  for i in self.list_cg_path]
        final_img_list = []
        for idx,value in enumerate(img_list):
            if img_name[idx] in graph_name:
                final_img_list.append(value)
        return final_img_list

       
    def get_captions_labels(self,img_path, split):
        list_name = glob(img_path+"/*.png")
        image_names = [os.path.splitext(os.path.split(i)[-1])[0]  for i in list_name]
        image_file_paths = [os.path.join(self.img_path,self.split,i) for i in image_names]
        with open(self.report_path, 'r') as json_file:
            report_data = json.load(json_file)
        sorted_report = {key: report_data[key] for key in sorted(report_data)}
 
        list_sorted_key = [path.split("/")[-1].replace('.bin','') for path in self.list_cg_path]

        self.captions = [sorted_report[key]['caption'] for key in image_names if key in  list_sorted_key]
        self.labels = [sorted_report[key]['label'] for key in image_names if key in  list_sorted_key]
        for idx,value in enumerate(self.labels):
            if value == 0 or value ==3:
                self.labels[idx] = 0
  
    def get_cell_graph(self):
        # print(f"CG PATH IS {self.cg_path}")
        self.list_cg_path = glob(os.path.join(self.cg_path, '*.bin'))
        #print(len(self.list_cg_path))
        self.list_cg_path.sort()

        self.num_cg = len(self.list_cg_path)
        cell_graphs = None
        if self.load_in_ram:
            cell_graphs = [load_graphs(single_cg_path) for single_cg_path in self.list_cg_path]
            cell_graphs = [entry[0][0] for entry in cell_graphs]
        return cell_graphs
    
    def get_tissue_graph(self):
        #print(f"TG PATH IS {self.tg_path}")
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
                h5_to_tensor(single_assign_path).float()
                    for single_assign_path in self.list_assign_path
            ]
    def get_cap_and_token(self, caption):
 #   Process cations and labels


        sentences = caption.rstrip('.').replace(',','').split('. ')
        caption_tokens = [] # convert to tokens for all num_feature sentences
        clean_sentences = sentences
        '''
        If we want not the have the final sentence use sentences[:-1]
        '''
        for s, sentence in enumerate(sentences[:-1]):
            #   if feature (except conclusion) is insufficient information, do not output it
            #   but the conclusion (last one) is insufficient information, we still output it
            # if 'ins' in sentence:
            #     print(f"{sentence} - s is {s}")
            if 'insufficient information' in sentence and s < (len(sentences)-1): 
                # print(f"    get here!!! {sentence[s]}")
                clean_sentences[s] = ' '
                continue
            tokens = nltk.tokenize.word_tokenize(str(sentence).lower())
            clean_sentences[s] = sentence + ' <end>'

            tokens.append('<end>')  # add stop indictor
            tmp = [self.vocab(token) for token in tokens]
            caption_tokens.append(tmp)
        current_masks = torch.zeros(self.max_length, dtype=torch.bool)
        #   Add Padding if necessary
        caption_tokens = [item for sublist in caption_tokens for item in sublist] 
        if len(caption_tokens) < self.max_length:
            current_masks[len(caption_tokens):] = True
            padding = [self.PAD_TOKEN] * (self.max_length - len(caption_tokens)-1)
            caption_tokens = [self.START_TOKEN]+caption_tokens + padding
        caption = ' '.join(clean_sentences) + ''
        return caption_tokens, caption,current_masks

    '''
    Made changes, train with all 5 captions together
    '''
    def __getitem__(self,index):
        # return the cell graph, tissue graph, assignment matrix and the relevant 1 captions
        '''
        graph_id = index
        cap_id_in_img = random.randint(0, 4)
        '''
        if self.mode == "train":
            #cap_id_in_img = index % 5
            #graph_id = int(index / 5)
            graph_id = index
            cap_id_in_img = random.randint(0, 4)
        else:            
            graph_id = index
            cap_id_in_img = None
        image = Image.open(self.img[graph_id]).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((500,500)),
            transforms.ToTensor(),
        ])

        # Apply the transformation to the image
        image = transform(image)

        label = self.labels[graph_id]
        attention_masks = []

        return_caption_tokens = None
        #   pprevious code where in training, load one by one
        if self.mode == "train" and self.load_all is True:
            caption = self.captions[graph_id][cap_id_in_img]
            caption_tokens, caption,current_masks = self.get_cap_and_token(caption)
            return_caption_tokens = torch.tensor(caption_tokens).long()
        else :

            unclean_captions = self.captions[graph_id]
            caption = []
            caption_tokens = []
            for i in unclean_captions:

                caption_token , cap,current_masks = self.get_cap_and_token(i)
                #print(f"with graph_id {graph_id} capto is {caption_token}")
                caption.append(cap)
                caption_tokens.append(torch.tensor(caption_token).long())
            return_caption_tokens = torch.stack(caption_tokens)
        #rint(f"split is {self.split} and length caption {len(caption)} length token {len(caption_token)}")
        '''
        unclean_captions = self.captions[graph_id]
        caption_tokens = []
        caption = []
        # caption_tokens = []
        for i in unclean_captions:

            caption_token , cap = self.get_cap_and_token(i)
            #print(f"with graph_id {graph_id} capto is {caption_token}")
            caption.append(cap)
            caption_tokens.append(torch.tensor(caption_token).long())

        '''
            
        # 1. Hierarchical Graphs
        if hasattr(self, 'num_tg') and hasattr(self, 'num_cg'):
            if self.load_in_ram:
                cg = self.cg[graph_id]
                tg = self.tg[graph_id]
                assign_mat = self.assign_matrices[graph_id]

                '''
                Issue: How to check and guarantee that the cg and tg and matched with the assign_mat for the same index
                '''
            else:
                cg, _ = load_graphs(self.list_cg_path[graph_id])
                cg = cg[0]
                tg, _ = load_graphs(self.list_tg_path[graph_id])
                tg = tg[0]
                assign_mat = h5_to_tensor(self.list_assign_path[graph_id]).float()

            cg.ndata['feat'] = torch.nan_to_num(cg.ndata['feat'], nan=0.0)
            tg.ndata['feat'] = torch.nan_to_num(tg.ndata['feat'], nan=0.0)
            cg = set_graph_on_cuda(cg) if IS_CUDA else cg
            tg = set_graph_on_cuda(tg) if IS_CUDA else tg
            assign_mat = assign_mat.cuda() if IS_CUDA else assign_mat
            #print(len(caption_tokens))
            #print(f"img size {image.shape} typ")
            return cg,tg,assign_mat, return_caption_tokens, label, caption, image
            #return cg,tg,assign_mat, torch.stack(caption_tokens), label, caption
        
        #   Use only tissue graph
        elif hasattr(self,'num_tg'):
            if self.load_in_ram:
                tg = self.tissue_graphs[index]
            else:
                tg, _ = load_graphs(self.list_tg_path[index])
                tg = tg[0]
            tg = set_graph_on_cuda(tg) if IS_CUDA else tg
            return tg, assign_mat, torch.tensor(caption_tokens).long(), label, caption, image

        #   Use only cell graph
        else:
            if self.load_in_ram:
                cg = self.cell_graphs[index]
            else:
                cg, _ = load_graphs(self.list_cg_path[graph_id])
                cg = cg[0]
            cg = set_graph_on_cuda(cg) if IS_CUDA else cg
            return cg, assign_mat, torch.tensor(caption_tokens).long(), label, caption, image
    
    
    def __len__(self): # len(dataloader) self.cg * 5 / batch_size
        assert len(self.cg) == len(self.tg)
        # if self.split == "train" and self.load_all is True:
        #     return len(self.cg)*5
        # else :
        return len(self.cg)
        ''' 
        return len(self.cg)
        '''




def collate(batch):
    """
    Collate a batch.
    Args:
        batch (list): a batch of examples.
    Returns:
        data: (tuple of DGLGraph)
    """
    def collate_fn(batch, id):
        return [example[id] for example in batch]

    # Collate the data
    num_modalities = len(batch[0])  # should 2 if CG or TG processing or 4 if HACT
    batch_collated = [collate_fn(batch, mod_id) for mod_id in range(num_modalities)]
    #print(f"----------Collated_batch-----------")
    #print(batch_collated[-2])
    #batch_collated[-2] = torch.stack(batch_collated[-2])    #   Stack the captions
    batch_collated[0] = dgl.batch(batch_collated[0])
    #if len(batch_collated) == 5:
    batch_collated[1] = dgl.batch(batch_collated[1])
    batch_collated[3] = torch.stack(batch_collated[3])
    batch_collated[4] = torch.tensor(batch_collated[4])
    batch_collated[6] = torch.stack(batch_collated[6])
    # batch_collated[7] = torch.stack(batch_collated[7])
    return batch_collated

def make_dataloader(
        batch_size,
        split,
        base_data_path,
        graph_path,
        vocab_path,
        load_in_ram = False,
        shuffle=True,
        num_workers=0,
        sampler = None,
        mode=  "train"
    ):
    """
    Create a BRACS data loader.
    """


    dataset = DiagnosticDataset(
                split = split,
                base_data_path = base_data_path,
                graph_path = graph_path,
                vocab_path = vocab_path,
                load_in_ram = load_in_ram,
                mode = mode
            )
        #   add sampling
    # dataset_size = len(dataset)
    # indices = list(range(dataset_size)) 
    # sampler = SubsetRandomSampler(indices)
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate
        )
    return dataloader,dataset
def dataset_to_loader(dataset,batch_size,sampler,shuffle = True, num_workers = 0):
    return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate,
            sampler = sampler
        )

if __name__ == "__main__":
    import os
    split = "train"
    splits = ["train","test","eval"]
    loader,_ = make_dataloader(
        batch_size = 16,
        split = split,
        base_data_path = "../../../../../../srv/scratch/bic/peter/Report",
        graph_path = "../../../../../../srv/scratch/bic/peter/full-graph",
        vocab_path = "vocab_bladderreport.pkl",
        shuffle=True,
        num_workers=0,
        load_in_ram = True
    )
    print(f"length data loader for {split} is {len(loader)}")
    for batch_idx, batch_data in enumerate(loader):
    # Your batch processing code here
        cg, tg, assign_mat, caption_tokens, labels, caption, images,attention_masks = batch_data
        print(f"------")
        print(caption)
        for i in caption:
            print(i)
            print(" --- ")
        # print(f"caption token {caption_tokens.shape}")
        # print(f"attention masks {attention_masks.shape}")
        break

    # for sp in splits:
    #     loader,_ = make_dataloader(
    #         batch_size = 16,
    #         split = sp,
    #         base_data_path = "../../../../../../srv/scratch/bic/peter/Report",
    #         graph_path = "../../../../../../srv/scratch/bic/peter/full-graph",
    #         vocab_path = "vocab_bladderreport.pkl",
    #         shuffle=True,
    #         num_workers=2,
    #         load_in_ram = True
    #     )
    #     one = 0
    #     zero = 0
    #     two = 0
    #     for batch_idx, batch_data in enumerate(loader):
    #         # Your batch processing code here
    #         cg, tg, assign_mat, caption_tokens, labels, caption, images = batch_data
    #         for j in labels:
    #             if j == 0:
    #                 zero += 1
    #             elif j == 1:
    #                 one += 1
    #             elif j == 2:
    #                 two += 1
    #             else:
    #                 print(f"     j is {j}")
    #     print(f"--------{sp}----------")
    #     print(f"Label 0: Normal/Insuff is {zero}")
    #     print(f"Label 1: Low Grade is {one}")
    #     print(f"Label 2: High Grade is {two}")
            
 