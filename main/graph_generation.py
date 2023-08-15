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

class GraphBuilding:
    def __init__(self,image_target_path):
        self.image_path = image_target_path
        self.stain_normalizer = VahadaneStainNormalizer(target_path=self.image_path)

        self.assignment_mat_builder = AssignmnentMatrixBuilder()

        # Store Image name that might have failed
        self.image_failed = []
    
    def cell_graph_builder(self):
        self.nuclei_detector = NucleiExtractor()

        # b define feature extractor: Extract patches of 72x72 pixels around each
        # nucleus centroid, then resize to 224 to match ResNet input size.
        self.nuclei_feature_extractor = DeepFeatureExtractor(
            architecture='resnet34',
            patch_size=72,
            resize_size=224
        )

        # c define k-NN graph builder with k=5 and thresholding edges longer
        # than 50 pixels. Add image size-normalized centroids to the node features.
        # For e.g., resulting node features are 512 features from ResNet34 + 2
        # normalized centroid features.
        self.knn_graph_builder = KNNGraphBuilder(k=5, thresh=50, add_loc_feats=True)
    
    def build_tissue_graph_builder(self):
        # a define nuclei extractor    
        self.tissue_detector = ColorMergedSuperpixelExtractor(
            superpixel_size=500,
            compactness=20,
            blur_kernel_size=1,
            threshold=0.05,
            downsampling_factor=4
        )

        # b define feature extractor: Extract patches of 144x144 pixels all over 
        # the tissue regions. Each patch is resized to 224 to match ResNet input size.
        self.tissue_feature_extractor = DeepFeatureExtractor(
            architecture='resnet34',
            patch_size=144,
            resize_size=224
        )

        # c define RAG builder. Append normalized centroid to the node features. 
        self.rag_graph_builder = RAGGraphBuilder(add_loc_feats=True)
    
    def build_cg(self, image):
        nuclei_map, nuclei_centroids = self.nuclei_detector.process(image)
        features = self.nuclei_feature_extractor.process(image, nuclei_map)
        graph = self.knn_graph_builder.process(nuclei_map, features)
        return graph, nuclei_centroids

    def build_tg(self, image):
        superpixels, _ = self.tissue_detector.process(image)
        features = self.tissue_feature_extractor.process(image, superpixels)
        graph = self.rag_graph_builder.process(superpixels, features)
        return graph, superpixels
    
    '''
    Images_folder: pass in where the image are stored in
    Store_path: Where to store the Graphs
    Split: Whether it is train or test
    '''
    def build(self, images_folder, store_path, split):
        #   Get a list of image and store them
        folder = os.path.join(images_folder, split)
        images_path = [file for file in os.listdir(folder) if file.endswith('.png')]

        for image_path in tqdm(images_path):
            _, image_name = os.path.split(image_path)
            image = np.array(Image.open(image_path))
        #   Get the store path
            cg_out = os.path.join(store_path, 'cell_graphs', split, image_name.replace('.png', '.bin'))
            tg_out = os.path.join(store_path, 'tissue_graphs', split, image_name.replace('.png', '.bin'))
            assign_out = os.path.join(store_path, 'assignment_matrices', split, image_name.replace('.png', '.h5'))

        #   Stain Normalisation
        try: 
            image = self.normalizer.process(image)
        except:
            print('Warning: {} failed during stain normalization.'.format(image_path))
            self.image_failed.append(image_path)
            pass

        #   Build Cell Graph and save it
        try: 
            cg, nuclei_centroid = self.build_cg(image)
            save_graphs(
                cg_out,
                g_list = [cg]
            )
        except:
            print('Warning: {} failed during cell graph building.'.format(image_path))
            self.image_failed.append(image_path)
            pass

        #   Build Tissue Graph and save it
        try:
            tissue_graph, tissue_map = self.build_tg(image)
            save_graphs(
                tg_out,
                g_list = [tissue_graph]
            )
        except:
            print('Warning: {} failed during tissue graph building.'.format(image_path))
            self.image_failed.append(image_path)
            pass

        try: 
            assignment_matrix = self.assignment_matrix_builder.process(nuclei_centroid, tissue_map)
            with h5py.File(assign_out, "w") as output_file:
                output_file.create_dataset(
                    "assignment_matrix",
                    data=assignment_matrix,
                    compression="gzip",
                    compression_opts=9,
                )
        except:
            print('Warning: {} failed during assignment matrix generation.'.format(image_path))
            self.image_failed.append(image_path)
            pass
    
        image_ids_failing = list(set(image_ids_failing))
        print('Out of {} images, {} successful graph generations.'.format(
            len(images_path),
            len(images_path) - len(image_ids_failing)
        ))

