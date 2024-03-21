import dgl
import torch
from dataloader import make_dataloader
from dgl.data import CoraGraphDataset
from models.DiffPool import DiffPoolLayer
from torch_geometric.utils import to_dense_batch, to_dense_adj
# from dgl.nn import TopKPooling
from models.Transformer import TransCapDecoder
import pickle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
phase = "train"
dl,dataset = make_dataloader(
    batch_size = 2,
    split = phase,
    base_data_path = "../../../../../../srv/scratch/bic/peter/Report",
    graph_path = "../../../../../../srv/scratch/bic/peter/full-graph",
    vocab_path = "new_vocab_bladderreport.pkl",
    shuffle=True,
    num_workers=0,
    load_in_ram = True
)
cg, tg, assignment_mat, caption_tokens, labels, caption, images, att_mask, idx_list= next(iter(dl))
feature_size = 512  # Feature size from CNN
d_model = 512
nhead = 8
num_decoder_layers = 3
dim_feedforward = 2048
max_seq_length = 70
SOS_TOKEN = 0  # Assuming 0 is the index of the start-of-sentence token
with open("new_vocab_bladderreport.pkl", 'rb') as file:
    vocabs = pickle.load(file)
print(att_mask)
model =  TransCapDecoder(
    vocabs = vocabs,
    embed_size =  feature_size,
    nhead = nhead, 
    num_layers = 3, 
    dim_feedforward= dim_feedforward, 
    dropout= 0.3,
    device = device,
)


'''
# Load a sample DGL graph (Cora dataset for example)
batched_graph = cg
print(batched_graph.batch_size)
# Convert DGL graph to PyTorch Geometric format
unbatch = dgl.unbatch(batched_graph)
for i in unbatch:
    print(i.ndata['feat'].shape)

x_dense = batched_graph.ndata['feat'].unsqueeze(0) 
adj_dense = batched_graph.adjacency_matrix().to_dense().unsqueeze(0)
print(f"x dense {x_dense.shape}")
print(f"adj dense {adj_dense.shape}")
# Define dimensions

dim_input = x_dense.size(2)

dim_hidden = 64
dim_embedding = 32
current_num_clusters = adj_dense.size(1)
no_new_clusters = 10
pooling_type = 'gnn'
invariant = False

# Instantiate the DiffPoolLayer
diff_pool_layer = DiffPoolLayer(dim_input, dim_hidden, dim_embedding, 
                                current_num_clusters, no_new_clusters, 
                                pooling_type, invariant).to(device)

# Forward pass
x_pooled, adj_pooled, l, e = diff_pool_layer(x_dense, adj_dense)

print("Pooled features shape:", x_pooled.shape)
print("Pooled adjacency matrix shape:", adj_pooled.shape)
'''