import os
from simple_parsing import ArgumentParser
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import tensorflow as tf
import yaml


from evaluation import Scorer
from dataloader import make_dataloader

import wandb

from argparse import ArgumentParser

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
    return config

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
    return parser
    # parser.add_argument('echo', help = 'echo the given string')
    # parser.add_argument('-n','--number',help = "number", type = int, default = 0, nargs = '?')
    # parser.add_argument('-v','--verbose',help = "Provide disc", action = "store_true") # if --verbose has value in command line, then, it is true, 
    # parser.add_argument('-w','--weight',help = "Wegihts", type = int, choices = [0,1,2]) # 

    # args = parser.parse_args()

def main():
    #   Get all the parser
    parser = argparser()
    args = sparser.parse_args()
    args = update_argparser(args)
    
    make_dataloader(
        # batch_size = args.batchsize,
        # num_workers = 
    )


