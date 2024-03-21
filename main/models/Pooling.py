from torch_geometric.nn import dense_diff_pool, dense_mincut_pool
import torch
import torch.nn as nn
import dgl

#   Build a GNN to learnable generate assignment matrix
class ScoringFunction(nn.Module):
    def __init__(self, feature_dim,method = "concat"):
        super(ScoringFunction, self).__init__()
        self.feature_dim = feature_dim
        self.position_dim = 4
        self.method = method
        if self.method == "concat":
            
            self.feat_linear = nn.Linear(self.feature_dim, 1)  # Output is a single score per node
            self.pos_linear = nn.Linear(self.position_dim,1)
            self.weight_feat = nn.Parameter(torch.tensor([1.0]))  # Learnable weight for feature score
            self.weight_pos = nn.Parameter(torch.tensor([1.0]))   # Learnable weight for position score
        else:
            self.linear = nn.Linear(self.feature_dim+self.position_dim,1)
    def forward(self, g):
        feat = g.ndata['feat']
        pos = g.ndata['position']
        # weighted sum apporach or linera apporach
        if self.method == "concat":
            feat_score = self.feat_linear(feat)
            pos_score = self.pos_linear(pos)
            return self.weight_feat * feat_score + self.weight_pos * pos_score
        else:
            combine_feat = torch.cat((feat,pos),dim = 2)
            out = self.linear(combine_feat)
        return out
    
def topk_pooling_individual(batched_graph, ratio, scoring_fn):
    # Compute scores for each node using the scoring function
    scores = scoring_fn(batched_graph).squeeze()
    

    # Initialize a mask tensor based on the existing mask or all True if not existing
    if 'mask' in batched_graph.ndata:
        mask = batched_graph.ndata['mask'].clone() 
    else:
        mask = torch.ones(batched_graph.num_nodes(), dtype=torch.bool, device=scores.device)

    # Apply the existing mask to the scores
    before_score = scores
    scores = scores * mask.float()
    after_score = scores
    print(scores)
    print(before_score == after_score)
    # Unbatched graph into individual graphs
    individual_graphs = dgl.unbatch(batched_graph)

    start_index = 0

    for graph, graph_scores in zip(individual_graphs, scores.split(batched_graph.batch_num_nodes().tolist())):

        # Calculate the number of nodes to retain based on the ratio, also consider previous mask if exists
        k = max(1, int(torch.sum(mask[start_index:start_index + graph.num_nodes()]).item() * ratio))

        # Select top-K nodes based on scores
        _, topk_indices = torch.topk(graph_scores, k)
        # Update the mask for the selected nodes
        new_mask = torch.zeros_like(mask[start_index:start_index + graph.num_nodes()], dtype=torch.bool)
        new_mask[topk_indices] = True
        mask[start_index:start_index + graph.num_nodes()] = new_mask

        # Update the start index for the next graph
        start_index += graph.num_nodes()

    # Store the mask in the batched graph's node data
    batched_graph.ndata['mask'] = mask

    return batched_graph
