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

import yaml
import sys
sys.path.append('../histocartography/histocartography')
sys.path.append('../histocartography')

from evaluation import Scorer
from dataloader import make_dataloader

import wandb

from argparse import ArgumentParser
from models.BaseModel import GNN_LSTM
from Vocabulary import Vocabulary
from models.Graph_Model import GNNEncoder
from models.LSTM import LSTMDecoder



# parser = ArgumentParser()
# parser.add_argument('echo', help = 'echo the given string')
# parser.add_argument('-n','--number',help = "number", type = int, default = 0, nargs = '?')
# parser.add_argument('-v','--verbose',help = "Provide disc", action = "store_true") # if --verbose has value in command line, then, it is true, 
# parser.add_argument('-w','--weight',help = "Wegihts", type = int, choices = [0,1,2]) # 

# args = parser.parse_args()


# if args.verbose:
#     print(args.echo)
# else:
#     print("None verbose")
    
# if args.weight == 0:
#     print("Weight is 0")
# elif args.weight == 1:
#     print("Weight is 1")
# else:
#     print(f"{args.weight}")
import sys
sys.path.append('../histocartography/histocartography')
from preprocessing import (
    VahadaneStainNormalizer,         # stain normalizer
    NucleiExtractor,                 # nuclei detector 
    DeepFeatureExtractor,            # feature extractor 
    KNNGraphBuilder,                 # kNN graph builder
    ColorMergedSuperpixelExtractor,  # tissue detector
    DeepFeatureExtractor,            # feature extractor
    RAGGraphBuilder,                 # build graph
    AssignmnentMatrixBuilder         # assignment matrix 
)

print("OK")

''' 
Update relevant arguments
input: args read and the content of config_file
'''
def update_argparser(args, config_file_path = "config/config.yaml"):
    if args.config_path:
        config_file_path = args.config_path
    with open(config_file_path, "r") as conf_file:
        config = yaml.full_load(conf_file)
    
    if args.phase:
        config["phase"] = args.phase
    if args.in_ram:
        config["in_ram"] = True
    else:
        config["in_ram"] = False
    if args.epochs:
        config["epochs"] = args.epoch
    return config

'''
Parse Argument
'''
def argparser():
    parser = ArgumentParser()

    parser.add_argument(
        '--phase',
        help = "Enter either: train, test or eval, represent the stage of model",
        type = str,
        choices = ["train","test","eval"]
        )

    parser.add_argument('--config_path',
        help = "path to the config path used",
        type = str)

    parser.add_argument(
        '--in_ram',
        help='if the data should be stored in RAM.',
        action='store_true',
    )

    parser.add_argument(
        '--epochs', type=int, help='epochs.', required=False
    )

    parser.add_argument(
        '-lr','--learning_rate', type = float, help = "set learning rate"
    )

    return parser

def embed2sentence(decode_output, loader, captions, pred_dict, cap_dict):
    max_indices = torch.argmax(decode_output, dim=2) 

    assert len(pred_dict) == len(cap_dict)


    for i,caption in enumerate(captions):
        for sent_i,sent_cap in enumerate(caption):
            captions[i][sent_i] = ' '.join(sent_cap.split()).replace("<pad>", "").replace("<end>", ".")

    for idx,embed in enumerate(max_indices):
        #print(embed)
        sentence = " ".join([loader.dataset.vocab.idx2word[int(idx)] for idx in embed])
        sentence = sentence.replace("<pad>","")
        sentence = ' '.join(sentence.split()).replace("<end>", ".")
        print(f"{sentence}")
        print("\n")
        print("-----------------CORRECT SETNENCE---------")
        print(captions[idx])
        print("\n")
        # for caption in enumerate(captions:


        if len(pred_dict.keys()) == 0:
            #   Empty
            pred_dict["1"] = [sentence]
            cap_dict["1"] = captions[idx]
            pass
        else:
            pred_dict[str(len(pred_dict)+1)] = [sentence]
            cap_dict[str(len(cap_dict)+1)] = captions[idx]

    return pred_dict,cap_dict


def eval(eval_loader,encoder,decoder,device, batch_size) :
    total_samples = len(eval_loader)
    total_step = math.ceil(total_samples / batch_size)
    pred_dict = {}
    cap_dict = {}
    for step in tqdm(range(total_step)):
        cg, tg, assign_mat, caption_tokens, labels, captions = next(iter(eval_loader))
        # caption_dict = {str(i + 1): value for i, value in enumerate(captions)}
        cg = cg.to( device)
        tg = tg.to(device)
        encoder , decoder = encoder.to(device) , decoder.to(device)
        with torch.no_grad():
            out = encoder(cg,tg,assign_mat)
            lstm_out = decoder(out,caption_tokens)
        #   Evaluate
        pred_dict,cap_dict = embed2sentence(lstm_out,eval_loader,captions,pred_dict,cap_dict)
    print(pred_dict)
    print("\n")
    print(cap_dict)
    print(f"total step is {total_step}")
    scorer = Scorer(pred_dict,cap_dict)
    scores = scorer.compute_scores()
    return scores
       # max_indices = torch.argmax(lstm_out, dim=2)  # Shape: (batch_size, position)
        # print(max_indices.shape)
        # print(max_indices[0])
        #print(f"length {loader.dataset.vocab.idx2word[88]}")
        # print(max_indices)
        #print(caption_dict)
        # for idx,embed in enumerate(max_indices):
        #     print(embed)
        #     sentence = " ".join([eval_loader.dataset.vocab.idx2word[int(idx)] for idx in embed])

        #     sentence = ' '.join(sentence.split()).replace("<pad>", "").replace("<end>", ".")
        #     # caption_dict[str(idx+1)]
        #     print(f"{sentence}")
        #     print("!!!!! Correct!!!!!!")
        #     captions[idx] = ' '.join(captions[idx].split()).replace("<pad>", "").replace("<end>", ".")
        #     print(captions[idx])
        #     print("\n")
        #     print("-------------")
        #     print("\n")


def unpack_score(scores):
    bleu = scores[0]
    meteor = scores[1]
    rouge = scores[2]
    cider = scores[3]
    spice = scores[4]
    return bleu[0], bleu[1], bleu[2], bleu[3], meteor, rouge, cider, spice

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_model = False
    save_model = True

    #   Get all the parser
    parser = argparser()
    args = parser.parse_args()
    print(args)
    print("Parse")
    args = update_argparser(args)
    print(args)
    learning_rate = args["learning_rate"]
    gnn_param = args["gnn_param"]
    lstm_param = args["lstm_param"]
    #   set path to save checkpoints

    #   make the dl here
    #   !!!!!!!!!!! Change it back to train
    train_dl = make_dataloader(
        batch_size = args["batch_size"],
        split = "train",
        base_data_path = args["dataset_path"],
        graph_path = args["graph_path"],
        vocab_path = args["vocab_path"],
        shuffle=True,
        num_workers=0,
        load_in_ram = True
    )

    test_dl = make_dataloader(
        batch_size = 1000, # there are 1000
        split = "test",
        base_data_path = args["dataset_path"],
        graph_path = args["graph_path"],
        vocab_path = args["vocab_path"],
        shuffle=True,
        num_workers=0,
        load_in_ram = True
    )

    eval_dl = make_dataloader(
        batch_size = 889, # there are 889 in test set
        split = "eval",
        base_data_path = args["dataset_path"],
        graph_path = args["graph_path"],
        vocab_path = args["vocab_path"],
        shuffle=True,
        num_workers=0,
        load_in_ram = True
    )
    #   Define Model, Loss and 
    vocab_size = len(train_dl.dataset.vocab)
    encoder = GNNEncoder(
        cell_conv_method = args["gnn_param"]["cell_conv_method"], 
        tissue_conv_method = args["gnn_param"]["tissue_conv_method"], 
        pool_method = None, 
        num_layers = 3, 
        aggregate_method = "sum", 
        input_feat = 514,
        output_size = 256
    )


    decoder = LSTMDecoder(
        vocab_size = vocab_size, 
        embed_size = 1028, 
        hidden_size = 128,  
        batch_size= args["batch_size"], 
        device = DEVICE
    )
    encoder , decoder = encoder.to(DEVICE) , decoder.to(DEVICE)
 
    criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()
    all_params = list(decoder.parameters())  + list( encoder.parameters() )
    optimizer = torch.optim.Adam(params = all_params, lr= args["learning_rate"])
    
    MAX_LENGTH = 100
    # model = GNN_LSTM(encoder, decoder,hidden_dim, vocab_size,gnn_param, lstm_param, phase)
    # model.to(DEVICE)
    torch.autograd.set_detect_anomaly(True)
    if args["phase"] == "train":
        #   Model training

        #   set the wandb project where this run will be logged
        # wandb.init(
        #     project="GNN simplifcation and application in histopathology image captioning",
        #     #   track hyperparameters and run metadata
        #     config={
        #         "architecture": "GNN-LSTM",
        #         "dataset": "Nmi-Wsi-Diagnosis",
        #         "epoch": args["epochs"]
        #     }
        # )
        total_samples = len(train_dl)
        batch_size = 4
        total_step = math.ceil(total_samples / batch_size)
        
        print(f"Number of steps per epoch: {total_step}")
        print(type(train_dl))
        for epoch in tqdm(range(args["epochs"])):
            total_loss = 0.0
            # for batched_idx, batch_data in enumerate(tqdm(train_dl)):
            for step in tqdm(range(total_step)):
                #if args["graph_model_type"] == "Hierarchical":
                cg, tg, assign_mat, caption_tokens, labels, caption = next(iter(train_dl))

                cg = cg.to(DEVICE)
                tg = tg.to(DEVICE)
                # assign_mat = assign_mat.to(DEVICE)
                caption_tokens = caption_tokens.to(DEVICE) # (batch_size, num_sentences, num_words_in_sentence) num_sentence = 6, num_words = 16
                # print(encoder)
                encoder.zero_grad()    
                decoder.zero_grad()
                # print(f"Input shape is {cg}")
                out = encoder(cg,tg,assign_mat) # (batch_size, 1, embedding)
                #print(f"Out shape is {out.shape}")
                # print(out)
                # print(f"------out--------")
                lstm_out = decoder(out,caption_tokens)
                #print(f"caption shape {caption_tokens.shape} lstm shape is {lstm_out.shape}")
            #   At the end, run eval set  
                #wandb.log({"loss":loss})
                lstm_out_prep = lstm_out.view(-1, vocab_size)
                caption_prep = caption_tokens.view(-1) 
                #print(f"LSTM view shape {lstm_out.view(-1, vocab_size).shape} and cap_tok {caption_tokens.view(-1).shape}")
                #print(f"LSTM OUT PREP type {type(lstm_out_prep)} and caption_prep {type(caption_prep)}")
                loss = criterion(lstm_out.view(-1, vocab_size) , caption_tokens.view(-1) )
                #loss = criterion(lstm_out.view(-1, vocab_size) , caption_tokens)
                print(f"\nAt epoch {epoch}, step {step} Cross Entropy Loss is {loss} !!!!!!")
                loss.backward()
                nn.utils.clip_grad_norm_(all_params, 1.0)
                optimizer.step()
                #print(f"FEAT SIZE IS {cg.ndata['feat'].shape}")
                # print(cg.ndata['feat'])
        scores = eval(eval_dl,encoder,decoder,DEVICE,889)
        bleu1, bleu2, bleu2, bleu4, meteor, rouge, cider, spice = unpack_score(scores)
        wandb.log({
            'bleu1':bleu1,
            'bleu2':bleu2,
            'bleu3':bleu3,
            'bleu4':bleu4,
            'meteor':meteor,
            'rouge':rouge,
            'cider':cider,
            'spice':spice
        })

    else:
        #   Only run the Test set
        print("test")



if __name__ == "__main__":
    main()
    # from dataloader import make_dataloader
    # from Vocabulary import Vocabulary
    # from models.Graph_Model import GNNEncoder
    # from models.LSTM import LSTMDecoder
    # import torch
    # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(DEVICE)
    # loader = make_dataloader(
    #     batch_size = 4,
    #     split = "test",
    #     base_data_path = "../../Report-nmi-wsi",
    #     graph_path = "graph",
    #     vocab_path = "../../Report-nmi-wsi/vocab_bladderreport.pkl",
    #     shuffle=True,
    #     num_workers=0,
    #     load_in_ram = True
    # )
    # vocab_size = len(loader.dataset.vocab)
    # encoder = GNNEncoder(cell_conv_method = "GCN", tissue_conv_method = "GCN", pool_method = None, num_layers = 3, aggregate_method = "sum", input_feat = 514,output_size = 256)
    # decoder = LSTMDecoder(vocab_size = vocab_size, embed_size = 256, hidden_size = 128,  batch_size=4, device = DEVICE)

    # print(len(list(encoder.parameters())))
    # print(len(list(decoder.parameters())))
    # # 
    # for batched_idx, batch_data in enumerate(loader):
        
    #     cg, tg, assign_mat, caption_tokens, labels, caption = batch_data

    #     cg = cg.to(DEVICE)
    #     tg = tg.to(DEVICE)
    #     # assign_mat = assign_mat.to(DEVICE)
    #     caption_tokens = caption_tokens.to(DEVICE)
    #     print(f"--------------Caption -----------")
    #     print(caption)
    #     print(len(caption))
    #     print(f"--------------Caption -----------")
    #     encoder , decoder = encoder.to(DEVICE) , decoder.to(DEVICE)
    #     encoder.zero_grad()    
    #     decoder.zero_grad()
    #     out = encoder(cg,tg,assign_mat)
    #     print(f"Out shape is {out.shape}")
    #     lstm_out = decoder(out,caption_tokens)
    #     print(f"caption shape {caption_tokens.shape} lstm shape is {lstm_out.shape}")
    #     break
        # max_indices = torch.argmax(lstm_out, dim=2)  # Shape: (batch_size, position)
        # print(max_indices.shape)
        # print(max_indices[0])
        # #print(f"length {loader.dataset.vocab.idx2word[88]}")
        # for embed in max_indices:
        #     sentence = " ".join([loader.dataset.vocab.idx2word[int(idx)] for idx in embed])
        #     print(sentence)
        #     print("\n")
        #     print("-------------")
        #     print("\n")
        # break
    

    