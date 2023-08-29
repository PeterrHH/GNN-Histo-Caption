import os
from simple_parsing import ArgumentParser
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import pickle

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
    dataloader = make_dataloader(
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
        batch_size = args["batch_size"],
        split = "test",
        base_data_path = args["dataset_path"],
        graph_path = args["graph_path"],
        vocab_path = args["vocab_path"]
        shuffle=True,
        num_workers=0,
        load_in_ram = True
    )

    eval_dl = make_dataloader(
        batch_size = args["batch_size"],
        split = "test",
        base_data_path = args["dataset_path"],
        graph_path = args["graph_path"],
        vocab_path = args["vocab_path"]
        shuffle=True,
        num_workers=0,
        load_in_ram = True
    )
    #   Define Model, Loss and 
    vocab_size = len(dataloader.dataset.vocab)
    encoder = GNNEncoder(
        cell_conv_method = "GCN", 
        tissue_conv_method = "GCN", 
        pool_method = None, 
        num_layers = 3, 
        aggregate_method = "sum", 
        input_feat = 514,
        output_size = 256)


    decoder = LSTMDecoder(
        vocab_size = vocab_size, 
        embed_size = 256, 
        hidden_size = 128,  
        batch_size= args["batch_size"], 
        device = DEVICE)
 
    criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)
    
    MAX_LENGTH = 100
    # model = GNN_LSTM(encoder, decoder,hidden_dim, vocab_size,gnn_param, lstm_param, phase)
    # model.to(DEVICE)
    if args.phase == "train":
        #   Model training

        #   set the wandb project where this run will be logged
        wandb.init(
            project="GNN simplifcation and application in histopathology image captioning",
            #   track hyperparameters and run metadata
            config={
                "architecture": "GNN-LSTM",
                "dataset": "Nmi-Wsi-Diagnosis",
                "epoch": args["epochs"]
            }
        )
        for epoch in range(args["epochs"]):
            total_loss = 0.0
            for batched_idx, batch_data in tqdm(enumerate(train_dataloader)):
                if args["graph_model_type"] == "Hierarchical":
                    cg, tg, assign_mat, caption_tokens, labels = batch_data
                    cg = cg.to(DEVICE)
                    tg = tg.to(DEVICE)
                    assign_mat = assign_mat.to(DEVICE)
                    caption_tokens = caption_tokens.to(DEVICE)
                    encoder , decoder = encoder.to(device) , decoder.to(device)
                    encoder.zero_grad()    
                    decoder.zero_grad()
                    out = encoder(cg,tg,assign_mat)
                    lstm_out = decoder(out,caption_tokens)
                    loss = criterion(lstm_out.view(-1, vocab_size) , caption_tokens.view(-1) )
                    loss.backward()
                    optimizer.step()
            #   At the end, run eval set     
                wandb.log({"loss":loss})
                print(f"At epoch {epoch}, step {batched_idx+1} loss is {loss}")
                optimizer.zero_grad()
                model
                loss = F.cross_entropy(pred, labels)
                total_loss += loss
                loss.backward()
                optimizer.step()
                

    else:
        #   Only run the Test set
        print("test")



if __name__ == "__main__":
    # main()
    from dataloader import make_dataloader
    from Vocabulary import Vocabulary
    from models.Graph_Model import GNNEncoder
    from models.LSTM import LSTMDecoder
    import torch
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(DEVICE)
    loader = make_dataloader(
        batch_size = 4,
        split = "test",
        base_data_path = "../../Report-nmi-wsi",
        graph_path = "graph",
        vocab_path = "../../Report-nmi-wsi/vocab_bladderreport.pkl",
        shuffle=True,
        num_workers=0,
        load_in_ram = True
    )
    vocab_size = len(loader.dataset.vocab)
    for batched_idx, batch_data in enumerate(loader):
        print(batch_data[0])
        cg, tg, assign_mat, caption_tokens, label = batch_data  
       # print(assign_mat)
        print(caption_tokens[0])
        encoder = GNNEncoder(cell_conv_method = "GCN", tissue_conv_method = "GCN", pool_method = None, num_layers = 3, aggregate_method = "sum", input_feat = 514,output_size = 256)
        out = encoder(cg,tg,assign_mat)
        print(f"length is {len(assign_mat)}")
        print(f"GNN out shape is {out.shape}")
        decoder = LSTMDecoder(vocab_size = vocab_size, embed_size = 256, hidden_size = 128,  batch_size=4, device = DEVICE)
        lstm_out = decoder(out,caption_tokens)
        print(f"LSTM out shape {lstm_out.shape}")
        max_indices = torch.argmax(lstm_out, dim=2)  # Shape: (batch_size, position)
        print(max_indices.shape)
        print(max_indices[0])
        print(f"length {loader.dataset.vocab.idx2word[88]}")
        for embed in max_indices:
            sentence = " ".join([loader.dataset.vocab.idx2word[int(idx)] for idx in embed])
            print(sentence)
            print("\n")
            print("-------------")
            print("\n")
            
        # first = lstm_out[0].tolist()
        # output_words = []
        # vocab = pickle.load(open('vocab_bladderreport.pkl','rb'))
        # for batch in first:
        #     batch_words = [loader.dataset.vocab.idx2word[idx] for idx in batch]
        #     output_words.append(batch_words)
        #     print(output_words)
        # print(loader.dataset.vocab.idx2word[3])

        break
    