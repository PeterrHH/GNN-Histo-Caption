import os
from simple_parsing import ArgumentParser
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle
from tqdm import tqdm
import math
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import WeightedRandomSampler

import yaml
import sys
sys.path.append('../histocartography/histocartography')
sys.path.append('../histocartography')

from evaluation import Scorer
from dataloader import make_dataloader, dataset_to_loader

import wandb

from argparse import ArgumentParser
from models.Attention import EncoderLayer
from Vocabulary import Vocabulary
from models.Graph_Model import GNNEncoder
from models.LSTM import LSTMDecoder
from models.GlobalFeatureExtractor import GlobalFeatureExtractor
from models.Classifier import Classifier
from models.Transformer import TransformerDecoder
from data_plotting import violin_plot


'''

WEighted Sample
'''
def get_sample_samplier(dataset,batch_size):
    class_count = {
        '0': 0,
        '1': 0,
        '2': 0,
        }

    count = 0
    class_weight = []
    for idx,output in enumerate(dataset):
        _, _, _, _, labels, _, _,_= output
        if count % 500 == 0 :
            print(count)
        class_count[str(labels)] += 1
        count += 1

    print(class_count)
    for key,value in class_count.items():
        class_weight.append(1/value)
    print(class_weight)
    sample_weights = [0]*len(dataset)

    for idx, output in enumerate(dataset):
        _, _, _, _, labels, _, _,_= output
        # print(labels.shape)
        class_count[str(labels)] += 1
        sample_weights[idx] = class_weight[labels]
    sampler = WeightedRandomSampler(weights = sample_weights,num_samples = len(dataset),replacement = True)
    dl = dataset_to_loader(dataset, sampler = sampler, batch_size = batch_size)
    return dl

def get_all_models(args,encoder_path, global_feat_path, decoder_path,device): 
    if os.path.exist(encoder_path):
        # load
        encoder = torch.load(torch.load(encoder_path)).to(device)
        pass
    else:
        raise Exception(f"Encoder path {encoder_path} not exist")
    
    if os.path.exist(global_feat_path):
        # load
        global_feat_extractor = torch.load(torch.load(global_feat_path)).to(device)
        pass
    else:
        raise Exception(f"Encoder path {encoder_path} not exist")

    if os.path.exist(decoder_path):
        # load
        decoder = torch.load(torch.load(decoder_path)).to(device)
    else:
        raise Exception(f"Encoder path {decoder_path} not exist")

    classifier = Classifier(
        graph_output_size = args['gnn_param']['output_size'],
        global_output_size = args["global_class_param"]["output_size"],
        hidden_size = args["classifier_param"]["hidden_size"],
        num_class = args["classifier_param"]["num_class"],
        dropout_rate = args["classifier_param"]["dropout_rate"]).to(device)
    
    return encoder, global_feat_extractor, decoder, classifier
'''
Encoder
Global Feature Extractor
TransformerDecoder/LSTM Decoder
trained already for the captioning tasks
Then we use it to train the classifier

'''
def train_classifier(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    #   make the dl here
    #   !!!!!!!!!!! Change it back to train
    _,train_dataset = make_dataloader(
        batch_size = args["batch_size"],
        split = "train",
        base_data_path = args["dataset_path"],
        graph_path = args["graph_path"],
        vocab_path = args["vocab_path"],
        shuffle=True,
        num_workers=0,
        load_in_ram = True
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

    train_dl = get_sample_samplier(train_dataset,args["batch_size"])


    encoder, global_feature_extractor, decoder, classifier = get_all_models(args,encoder_path, global_feat_path, decoder_path,device)
    criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()

    classifier_optimizer = torch.optim.Adam(params = list(classifier.parameters()), lr = args["learning_rate"], weight_decay=args["weight_decay"])
    torch.autograd.set_detect_anomaly(True)

    total_samples = len(train_dl)
    batch_size = args["batch_size"]
    total_step = math.ceil(total_samples / batch_size)

    for epoch in range(args["epochs"]):
        total_loss = []
        for step in range(total_step):
            cg, tg, assign_mat, caption_tokens, labels, caption, images,_ = next(iter(train_dl))
            #print(f"caption tokens type{type(caption_tokens)} shape {caption_tokens.shape}")
            cg = cg.to(device)
            tg = tg.to(device)
            labels = labels.to(device)
            images = images.to(device)

            '''
            if we want to make encoder, global feat extractor trainable do it here
            '''
            encoder.eval()
            decoder.eval()

            out = encoder(cg,tg,assign_mat,images) # (batch_size, 1, embedding)
            global_feat = global_feature_extractor(images)

            merged_feat = torch.cat((out, global_feat), dim=1).unsqueeze(1)
            pred_matrix = classifier(merged_feat).to(device)

            '''
            calculate loss
            '''
            caption_loss = criterion(pred_matrix, labels)
            classifier_optimizer.zero_grad()
            classifier_optimizer.step()