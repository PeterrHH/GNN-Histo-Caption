import os
from simple_parsing import ArgumentParser
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import tensorflow as tf
import yaml
import sys
sys.path.append('../histocartography/histocartography')
sys.path.append('../histocartography')

from evaluation import Scorer
from dataloader import make_dataloader

import wandb

from argparse import ArgumentParser

# from ml.models import CellGraphModel, TissueGraphModel
from torch_geometric.nn.pool.diff_pool import DiffPool

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
    device = torch.device("cuda" if torch.cuda.is_avaliable() else "cpu")
    load_model = False
    save_model = True

    #   Get all the parser
    parser = argparser()
    args = parser.parse_args()
    print(args)
    print("Parse")
    args = update_argparser(args)
    print(args)

    #   set path to save checkpoints

    
    #   make the dl here
    train_dl = make_dataloader(
        split = "train",
        base_data_path = args.dataset_path,
        graph_path = args.graph_path,
        load_in_ram = args.in_ram,
        batch_size=args.batch_size,
    )

    test_dl = make_dataloader(
        split = "test",
        base_data_path = args.dataset_path,
        graph_path = args.graph_path,
        load_in_ram = args.in_ram,
        batch_size=args.batch_size,
    )

    eval_dl = make_dataloader(
        split = "eval",
        base_data_path = args.dataset_path,
        graph_path = args.graph_path,
        load_in_ram = args.in_ram,
        batch_size=args.batch_size,
    )
    
    MAX_LENGTH = 100
    if args.phase == "train":
        #   Model training

        #   set the wandb project where this run will be logged
        wandb.init(
            project="GNN simplifcation and application in histopathology image captioning",
            #   track hyperparameters and run metadata
            config={
                "architecture": "GNN-LSTM",
                "dataset": "Nmi-Wsi-Diagnosis",
                "epoch": args.epochs    
            }
        )
        with torch.no_grad():
            x = self.encoderGNN(image).unsqueeze(0)
            states = None

            for _ in max_length:

    else:
        #   Only run the Test set
        print("test")



if __name__ == "__main__":
    main()