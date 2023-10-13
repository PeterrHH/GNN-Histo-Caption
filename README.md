# GNN-Histo-Caption
GNN for Histopathology Image Captioning (working on it now)

## Dataset split
After downloading, first engage in dataset splitting. Origianl dataset only split train and test. We will further split test to create a valuation set and a testing set. We will also split the image according to the data split. 

The proper dataset set up should be as below

    .
    ├── ...
    ├── Images                    # Image Folder
    │   ├── ...                   # All the images are here
    ├── train_annotation.json     # original training annotation file
    ├── test_annotation.json      # original testing annotation file
    └── ...


To split data, run the file data_split.py.

    
    python3 data_split.py --dataset_path /path/to/image/data_path
    

After running data_split, data structure should be as below

    .
    ├── ...
    ├── Images                    # Image Folder
    │   ├── train                 # All the images for training set
    │   ├── eval                  # All the images for evaluation set
    │   └── test                  # All the images for testing set
    ├── train_annotation.json     # original training annotation file
    ├── eval_annotation.json      # original testing annotation file
    ├── test_annotation.json      # original testing annotation file
    └── ...



    
## Graph Building
Graph Building part of the code mainly followed the process in Hact-Net.
To run Graph Building, Run the graph building file.


    python3 graph_generation.py

It will generate a cell graph, tissue graph and assignment matrix for each image.


### Cell Graph <br />
1. **Nuclei Detection** <br />
specifically cell detection. This is done using the histocartography package. General rule is to use a pretrained HoverNet. Then extract a square patches around the detected nuclei. Patch size is set at 72 x 72 pixels. Feature extractor is a ResNet34 pretrained on Imagenet. <br />

2. **Tissue Detection**


3. **Hierarchical Graph**
