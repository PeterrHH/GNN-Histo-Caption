# GNN-Histo-Caption
GNN for Histopathology Image Captioning

## Dataset split
After downloading, first engage in dataset splitting. Origianl dataset only split train and test. We will further split test to create a valuation set and a testing set. We will also split the image according to the data split. To split data, run the file data_split.py.
## Graph Building
Graph Building part of the code mainly followed the process in Hact-Net
### Cell Graph <br />
1. **Nuclei Detection** <br />
specifically cell detection. This is done using the histocartography package. General rule is to use a pretrained HoverNet. Then extract a square patches around the detected nuclei. Patch size is set at 72 x 72 pixels. Feature extractor is a ResNet34 pretrained on Imagenet. <br />

2. **Tissue Detection**


3. **Hierarchical Graph**
