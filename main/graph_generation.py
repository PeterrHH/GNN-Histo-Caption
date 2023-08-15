import os
import sys
from glob import glob
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch 
from dgl.data.utils import save_graphs
import h5py
import yaml

# Read the YAML configuration file
with open("config/base.yaml", 'r') as config_file:
    config_data = yaml.safe_load(config_file)

# Retrieve the value for the key "Place"
place_value = config_data.get("histocartography_path")
print(place_value)

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