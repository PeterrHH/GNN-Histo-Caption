import os
from simple_parsing import ArgumentParser
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle
from tqdm import tqdm
import math
from sklearn.metrics import f1_score, accuracy_score

import yaml
import sys
sys.path.append('../histocartography/histocartography')
sys.path.append('../histocartography')

from evaluation import Scorer
from dataloader import make_dataloader,dataset_to_loader

import wandb

from argparse import ArgumentParser
from models.Attention import EncoderLayer
from Vocabulary import Vocabulary
from models.Graph_Model import GNNEncoder
#from models.LSTM import LSTMDecoder, Beam_LSTMDecoder
from models.LSTM2 import LSTMDecoder
from models.GlobalFeatureExtractor import GlobalFeatureExtractor
from models.Classifier import Classifier
from models.Transformer import TransformerDecoder
from data_plotting import violin_plot
from torch.utils.data import WeightedRandomSampler


def embed2sentence(decode_output, loader, captions,phase):
    pred_dict = {}
    cap_dict = {}

    for i,caption in enumerate(captions):
        # if phase == "train":
        #     captions[i]= ' '.join(caption.split()).replace("<pad>", "").replace("<end>", ".").replace("<start>","").replace('<unk>',"")

        # else:
        for sent_i,sent_cap in enumerate(caption):
            st = ' '.join(sent_cap.split()).replace("<pad>", "").replace("<end>", "").replace("<start>","").replace('<unk>',"").replace("<full-stop>",".")
            st = [s.strip().capitalize() for s in st.split('.')]
            st= '. '.join(st).rstrip('.')
            captions[i][sent_i] = st

                


            
    j = 0
    for idx,embed in enumerate(decode_output):

        sentence = " ".join([loader.dataset.vocab.idx2word[int(idx)] for idx in embed])
        sentence = sentence.replace("<pad>","").replace("<start>","")
        #sentence = ' '.join(sentence.split()).replace("<end>", "").replace("<full-stop>",".")
        sentence = ' '.join(sentence.split()).replace("<end>", "").strip()
        sentence = ' '.join(sentence.split()).replace(" <full-stop>", ".")

            
        sentences = [s.strip().capitalize() for s in sentence.split('.')]
        sentence = '. '.join(sentences).rstrip('.')

        #print(f"length pred_dict {len(pred_dict.keys())} and length cap {len(cap_dict.keys())}")
        if len(pred_dict.keys()) == 0:
            #   Empty
            pred_dict["1"] = [sentence]

            # else:
            '''
            print(f"------Prediction-----")
            print(sentence)
            print(f"------GT----------")
            print(captions[idx][0])
            print("xxxxxxxxxxxx")  
            '''

            cap_dict["1"] = captions[idx]
            pass
        else:
            pred_dict[str(len(pred_dict)+1)] = [sentence]
            # if phase == "train":
            #     cap_dict[str(len(cap_dict)+1)] = [captions[idx]]
            # else:
            cap_dict[str(len(cap_dict)+1)] = captions[idx]
    
    
    return pred_dict,cap_dict


def get_loader_score(encoder,global_feature_extractor, decoder, loader,device,phase):
    test_output = {
        "Bleu1":[],
        "Bleu2":[],
        "Bleu3":[],
        "Bleu4":[],
        "METEOR":[],
        "ROUGE_L":[],
        "CIDEr":[],
        # "SPICE":[]
    }
    batch_size = loader.batch_size
    total_samples = len(loader.dataset)
    total_step = math.ceil(total_samples / batch_size)
    #print(f"total_step is {total_step} and device is {device}")
    for step in range(total_step):
        #print(f"------------step is {step}--------------")
        cg, tg, assign_mat, caption_tokens, labels, captions, images,_,_ = next(iter(loader))
        #print(f"loader loades {caption_tokens.shape}")
        # caption_dict = {str(i + 1): value for i, value in enumerate(captions)}
        # print(f"Length of labels {labels.shape}")
        cg = cg.to( device)
        tg = tg.to(device)
        images =  images.to(device)
        caption_tokens = caption_tokens.to(device)
        with torch.no_grad():
            out = encoder(cg,tg,assign_mat,images)
            #print(f"encoder out shape {out.shape}")
            global_feat = global_feature_extractor(images)
            #merged_feat = out
            merged_feat = torch.cat((out, global_feat), dim=1)
            #print(f"g feat get")
            #merged_feat = global_feat
            #print(f"merged feat {merged_feat}")
            lstm_out, lstm_out_tensor = decoder.predict(merged_feat,80)
            #print(f"lstm_out shape in eval {lstm_out.shape}")
        pred_dict,cap_dict = embed2sentence(lstm_out,loader,captions,phase)
        scorer = Scorer(cap_dict,pred_dict)
        # if eval:
        print(cap_dict["1"])
        print(f"---------CAP ABOVE------")
        print(pred_dict["1"])
        print(f"-------PRED ABOVE-------")
        scores = scorer.compute_scores()
        for key,value in scores.items():
                test_output[key].append(value[0])
        # print(scores)
    final_output = {
        "bleu1":np.mean(test_output["Bleu1"]),
        "bleu2":np.mean(test_output["Bleu2"]),
        "bleu3":np.mean(test_output["Bleu3"]),
        "bleu4":np.mean(test_output["Bleu4"]),
        "meteor":np.mean(test_output["METEOR"]),
        "rouge":np.mean(test_output["ROUGE_L"]),
        "cider":np.mean(test_output["CIDEr"]),     
    }
    print(final_output)
        
        
def test_caption(args,encoder_path,decoder_path, global_feature_extractor_path,
         device):
    train_dl,_ = make_dataloader(
        batch_size =args["batch_size"], # there are 1000 set 1 because we will calculate pair by pair
        split = "train",
        base_data_path = args["dataset_path"],
        graph_path = args["graph_path"],
        vocab_path = args["vocab_path"],
        shuffle=True,
        num_workers=0,
        load_in_ram = True,
        mode = "eval"
    )
    test_dl,_ = make_dataloader(
        batch_size =1000, # there are 1000 set 1 because we will calculate pair by pair
        split = "test",
        base_data_path = args["dataset_path"],
        graph_path = args["graph_path"],
        vocab_path = args["vocab_path"],
        shuffle=True,
        num_workers=0,
        load_in_ram = True
    )

    eval_dl,_ = make_dataloader(
        batch_size = 889, # there are 889 in eval set
        split = "eval",
        base_data_path = args["dataset_path"],
        graph_path = args["graph_path"],
        vocab_path = args["vocab_path"],
        shuffle=True,
        num_workers=0,
        load_in_ram = True
    )
    encoder = None
    decoder = None
    global_feature_extractor = None
    print(f"ENCODER PATH: {encoder_path}")
    if os.path.exists(encoder_path):
        # load
        encoder = torch.load(encoder_path).to(device)
        encoder.eval()
        pass
    else:
        raise Exception(f"Encoder path {encoder_path} not exist")
        
    if os.path.exists(decoder_path):
        # load
        decoder = torch.load(decoder_path).to(device)
        decoder.eval()
        print(decoder)
    else:
        raise Exception(f"Encoder path {decoder_path} not exist")
        
    if os.path.exists(global_feature_extractor_path):
        # load
        global_feature_extractor = torch.load(global_feature_extractor_path).to(device)
        global_feature_extractor.eval()
    else:
        raise Exception(f"Encoder path {decoder_path} not exist")
    print(f"Train Score: \n")
    get_loader_score(encoder,global_feature_extractor,decoder,train_dl,device,"train")
    print(f"Eval Score: \n")
    get_loader_score(encoder,global_feature_extractor, decoder, eval_dl,device,"eval")
    print(f"Test Score: \n")
    get_loader_score(encoder,global_feature_extractor, decoder, test_dl,device,"test")
            
def test_classification(args,encoder,attention, global_feature_extractor, classifier,
         device) :
    pass

if __name__ == "__main__":
    # parser = ArgumentParser()
    # args = parser.parse_args()
    config_file_path = "config/config.yaml"
    with open(config_file_path, "r") as conf_file:
        args = yaml.full_load(conf_file)
    print(f"args are {args}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_base_path = args['model_save_base_path']
    encoder_path = os.path.join(model_base_path,args['load_encoder_name'])
    decoder_path = os.path.join(model_base_path,args['load_decoder'])
    global_feature_extractor_path = os.path.join(model_base_path,args['load_global_extractor_name'])
    test_caption(args,
                 encoder_path = encoder_path,
                 decoder_path = decoder_path, 
                 global_feature_extractor_path = global_feature_extractor_path,device = device)