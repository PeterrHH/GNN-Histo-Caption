import matplotlib.pyplot as plt
import os
import dgl
import torch

directory = "graph/cell_graphs/test"
graph_files = [f for f in os.listdir(directory) if f.endswith('.bin')]
# graph_files = ["1_00143_sub0_049.bin"]
print(len(graph_files))
graph_list = []
for graph_file in graph_files:
    graph_path = os.path.join(directory, graph_file)
    
    # Use DGL's function to read the binary graph
    graph = dgl.load_graphs(graph_path)[0][0]  # Assuming there's only one graph per file
    print(f"Name : {graph_file}")

    # # print(graph.ndata["feat"])
    # a = 1
    feat = graph.ndata["feat"]
    print(feat)
    # print(f"For file {graph_file} the NA is {torch.any(torch.isnan(feat))}")
    # if torch.any(torch.isnan(feat)) == True:
    #     print(feat)
    #     print(torch.isnan(feat))
    # graph_list.append(graph)
    # # print(graph)