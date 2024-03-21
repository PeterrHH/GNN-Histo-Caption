
"""BRACS Dataset loader."""
import os
import h5py
import torch.utils.data
import numpy as np
from dgl.data.utils import load_graphs
from torch.utils.data import Dataset
from glob import glob 
import nltk
from nltk.corpus import wordnet
nltk.download('wordnet')
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
from torch.utils.data import WeightedRandomSampler



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
        print(f"Base data path {self.base_data_path}")
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
        self.use_augmentation = True
        if self.split == 'train' and self.use_augmentation:
            self.transform = transforms.Compose([ 
                        transforms.RandomRotation(degrees=5),
                        transforms.Resize(299),
                        transforms.RandomCrop(256),
                        transforms.RandomVerticalFlip(),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor()
                         ])
        else:
            self.transform = transforms.Compose([ 
                        transforms.Resize(299),
                        transforms.CenterCrop(256),
                        transforms.ToTensor(),
                                    ])
        # self.vocab.add_word('<start>')
        # self.START_TOKEN = self.vocab.word2idx['<start>']
        #self.FULL_STOP = self.vocab.word2idx['<full-stop>']
        self.END_TOKEN = self.vocab.word2idx['<end>']
        self.PAD_TOKEN = self.vocab.word2idx['<pad>'] # PAD_TOKEN is used for not supervison
        self.START_TOKEN = self.vocab.word2idx['<start>']
        #print(f"END token is {self.END_TOKEN} pad is {self.PAD_TOKEN} start is {self.START_TOKEN}")
        self.max_length = 80
        self.vocab_size = len(self.vocab.word2idx)
        self.num_feature = 6
        self.max_subseq_len = 15+1
        self.stop_label = 2

        # Get list of cell graph
        self.cg = self.get_cell_graph()
        # Get list of tissue graph
        self.tg = self.get_tissue_graph()
        #print(f"self cg {len(self.cg)} self tg{len(self.tg)}")
        self.assign_mat = self.get_assign_mat()
        #print(f"length cg {len(self.cg)}")
        print(self.list_cg_path[0:2])
        self.reports = self.get_captions_labels(self.img_path,self.split)
        #print(f"length report {len(self.reports)}")
        self.img = self.get_img(self.img_path,self.split)
        self.key_words = [
            ['severe','moderate','normal','mild'],
            [['no signs','no nuclear crowding'],['normal','normally'],'mild','severe',['moderate','moderately']],
            ['normal',['not lost','negligibly lost','no loss'],['is completely lost','complete lack of','lack of cellular polarity'],['some degree','not completely lost','partially lost']],
            ['rare',['are frequently','is frequent','are frequent'],'infrequent'],
            ['inconspicuous',['are prominent','prominent nucleoi','is prominent'],'rare'],
        ]
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
        #print(report_data)
        sorted_report = {key: report_data[key] for key in sorted(report_data)}
 
        list_sorted_key = [path.split("/")[-1].replace('.bin','') for path in self.list_cg_path]

        self.captions = [sorted_report[key]['caption'] for key in image_names if key in  list_sorted_key]
        # print(self.captions)
        
        list_name = [os.path.basename(name)[:-4] for name in self.list_cg_path]
        #print(f"list_name : {list_name[0:2]}")
        self.labels = [sorted_report[key]['label'] for key in image_names if key in  list_sorted_key]
        report_data = {key: value for key, value in report_data.items() if key in list_name}
        
        for key,value in report_data.items():
            # print(value)
            if value['label'] == 0 or value['label'] ==3:
                value['label'] = 0
        #print(f"report data len{report_data}")
        return report_data
  
    def get_cell_graph(self):
        # print(f"CG PATH IS {self.cg_path}")
        print(f"cg path is {self.cg_path}")
        self.list_cg_path = glob(os.path.join(self.cg_path, '*.bin'))
        self.list_cg_path.sort()
        #print(f"length cg is {len(self.list_cg_path)}")
        self.num_cg = len(self.list_cg_path)
        cell_graphs = None
        if self.load_in_ram:
            cell_graphs = [load_graphs(single_cg_path) for single_cg_path in self.list_cg_path]
            cell_graphs = [entry[0][0] for entry in cell_graphs]
        print(f"num cg {self.num_cg} cell graph len{len(cell_graphs)}")
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
    def get_synonyms(self,word):
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
        if word in synonyms:
            synonyms.remove(word)
        return list(synonyms)
    
    def get_sentence_label(self,caption):
        key_idx_list = []
        caption = caption.lower().rstrip('.').replace(',','').split('. ')[:-1]
        

        for index,value in enumerate(caption):
            found_key_word = False
            for key_idx,key_word in enumerate(self.key_words[index]):
                if isinstance(key_word,str):
                    if key_word in value:
                        #print(f"key_word: {self.key_word} in {index}: class is {key_idx}")
                        key_idx_list.append(key_idx)
                        found_key_word = True
                        break
                else:
                    # a list
                    for i in key_word:
                        if i in value:
                            key_idx_list.append(key_idx)
                            found_key_word = True
                            break
            if not found_key_word:
                key_idx_list.append(len(self.key_words[index]))
        return key_idx_list
        

    def get_cap_and_token(self, caption):
 #   Process cations and labels
        key_idx_list = self.get_sentence_label(caption)
        sentences = caption.rstrip('.').replace(',','').split('. ')
        caption_tokens = [] # convert to tokens for all num_feature sentences
        clean_sentences = []
        '''
        If we want not the have the final sentence use sentences[:-1]
        '''
        for s, sentence in enumerate(sentences[:-1]):
            # if 'insufficient information' in sentence and s < (len(sentences)-1): 
            #     # print(f"    get here!!! {sentence[s]}")
            #     clean_sentences[s] = ''
            #     continue
            #print(sentence)
            #print("------------")
            tokens = nltk.tokenize.word_tokenize(str(sentence).lower())
            #toekns = [word for word in tokens if '.' not in word]

            clean_sentences.append(sentence + '<full-stop>')

            tokens.append('<full-stop>')  # add stop indictor
                #clean_sentences.append(sentence + ' <full-stop>')
            if s == len(sentences[:-1])-1:
                clean_sentences.append('<end>')

                tokens.append('<end>')
            
            tmp = [self.vocab(token) for token in tokens]
            caption_tokens.append(tmp)
        #tokens.append('<end>')  # add stop indictor
        #print(clean_sentences)
        current_masks = torch.zeros(self.max_length, dtype=torch.int)
        #   Add Padding if necessary
        caption_tokens = [item for sublist in caption_tokens for item in sublist] 
        if len(caption_tokens) < self.max_length:
            current_masks[len(caption_tokens):] = 1
            padding = [self.PAD_TOKEN] * (self.max_length - len(caption_tokens)-1)
            caption_tokens = [self.START_TOKEN]+caption_tokens + padding
        caption = ' '.join(clean_sentences) + ''
        # caption = [string.strip() for string in caption if string.strip()]
        # print(f"clean sentences:")
        #print(caption)
        if caption.isspace():
            clean_sentences = "<end>"
        return caption_tokens, caption,current_masks, key_idx_list

    '''
    Made changes, train with all 5 captions together
    '''
    def __getitem__(self,index):
        # return the cell graph, tissue graph, assignment matrix and the relevant 1 captions
   
        graph_id = index
        cap_id_in_img = random.randint(0, 4) # for CPC gtraining we do this
        # '''     '''
        
        # if self.mode == "train":
        #     cap_id_in_img = index % 5
        #     graph_id = int(index / 5)
        #     # graph_id = index
        #     # cap_id_in_img = random.randint(0, 4)
        # else:            
        #     graph_id = index
        #     cap_id_in_img = None
        #     # cap_id_in_img = index % 5
        #     # graph_id = int(index / 5)

        image = Image.open(self.img[graph_id]).convert('RGB')

        # Apply the transformation to the image
        image = self.transform(image)

        label = self.reports[os.path.basename(self.img[graph_id])[:-4]]['label']
        attention_masks = []
        '''
        to get caption,
        self.report[self.img[graph_id]]
        '''
        #print(f"REPORT DATA IS {self.report[os.path.basename(self.img[graph_id])[:-4]]}")

        return_caption_tokens = None
        #print(f"g id {graph_id} cap_id img {cap_id_in_img}")
        if self.mode == "train" and self.load_all is True:
            # caption = self.captions[graph_id][cap_id_in_img]
            print(f"cg name {os.path.basename(self.img[graph_id])[:-4]}")
            orig_caption = self.reports[os.path.basename(self.img[graph_id])[:-4]]['caption'][cap_id_in_img]
        #print(f"caption before {caption}")
            caption_tokens, caption,current_masks,key_idx_list = self.get_cap_and_token(orig_caption)
        # print(f"caption after {caption}")
            # print(f"---currmask----")
            # print(current_masks)
            # print(f"--------")
            attention_masks = current_masks
        # attention_masks.append(current_masks)
            return_caption_tokens = torch.tensor(caption_tokens).long()
            key_idx_list = torch.tensor(key_idx_list)

        else :
            all_key_idx_list = []
            unclean_captions = self.reports[os.path.basename(self.img[graph_id])[:-4]]['caption']
            caption = []
            caption_tokens = []
            attention_masks = torch.zeros((1,1))
            for i in unclean_captions:

                caption_token , cap,current_masks,key_idx_list = self.get_cap_and_token(i)
                # print(f"---currmask----")
                # print(current_masks)
                # print(f"--------")
                # attention_masks.append(current_masks[0])
                all_key_idx_list.append(torch.tensor(key_idx_list))
                caption.append(cap)
                caption_tokens.append(torch.tensor(caption_token).long())
            
            return_caption_tokens = torch.stack(caption_tokens)
            key_idx_list = torch.stack(all_key_idx_list)

  
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
            # print(f"caption in loader below:")
            # print(caption)

            return cg,tg,assign_mat, return_caption_tokens, label, caption, image, attention_masks, key_idx_list
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
            return cg, assign_mat, torch.tensor(caption_tokens).long(), label, orig_caption, image
    
    
    def __len__(self): # len(dataloader) self.cg * 5 / batch_size
        assert len(self.cg) == len(self.tg)
        #sreturn len(self.cg)*5
        # if self.mode == "train":
        #     return len(self.cg)*5
        # else :
        # print(f"at the end lencg  {len(self.cg)}")
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
    # print("----------")
    # print(batch_collated[5])
    # print("----------")
    batch_collated[6] = torch.stack(batch_collated[6])
    # print(f"batch_collated type {batch_collated[7]}")
    batch_collated[7] = torch.stack(batch_collated[7]).bool()
    #batch_collated[7] = torch.stack([ten[0] for ten in batch_collated[7]])

    batch_collated[8] = torch.stack(batch_collated[8])
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
            sampler = sampler,
            shuffle = shuffle,
        )

if __name__ == "__main__":

    import os
    print("DATALOADER")
    split = "train"
    
    splits = ["train","test","eval"]
    loader,_ = make_dataloader(
        batch_size = 2,
        split = split,
        base_data_path = "../../../../../../srv/scratch/bic/peter/Report",
        graph_path = "../../../../../../srv/scratch/bic/peter/full-graph-raw",
        vocab_path = "new_vocab_bladderreport.pkl",
        shuffle=True,
        num_workers=0,
        load_in_ram = True
    )
    # for idx,output in enumerate(dataset):
    #     _, _, _, _, labels, _, _= output
    #     print(labels)
    #     class_count[str(labels)] += 1
    #     count += 1

    print(f"length data loader for {split} is {len(loader)}")
    i = 0
    for batch_idx, batch_data in enumerate(loader):
    # Your batch processing code here
        cg, tg, assign_mat, caption_tokens, labels, caption, images, att_mask, idx_list= batch_data
        # print(f"------")
        # print(f"cg shape is {cg.ndata['feat'].shape}")
        # print(f"tg shape is {tg.ndata['feat'].shape}")
        # print(f"assign_mat is {assign_mat[0].shape}")
        print(cg)
        # print(tg)
        print(caption)
        print(len(caption_tokens[0]))
        print("-----")
        break
        # print(att_mask.shape)
        # print(att_mask)
        # print(caption_tokens)
        if i == 2:
            break
        i+=1
        #print(f"idx_list shape is {idx_list.shape}")
    
        #print(caption_tokens.shape)

        # print(attention_masks)
        # for idx,value in enumerate(caption):
        #     print(value)
        #     print(caption_tokens[idx])
        #     print(" --- ")
        # # print(f"caption token {caption_tokens.shape}")
        # print(f"attention masks {attention_masks.shape}")
        #break
    print(f"--------------")
    print(f"Finished")
