from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

import networkx as nx
#import walker

# create a random graph
#G = nx.random_partition_graph([1000] * 15, .01, .001)

# you can generate random walks from specified starting nodes
#X = walker.random_walks(G, n_walks=50, walk_len=4, start_nodes=[0, 1, 2])

#print(X)

dataset = PygNodePropPredDataset(name='ogbn-products')
split_idx = dataset.get_idx_split()
print(len(split_idx['train']), len(split_idx['valid']), len(split_idx['test']))
data = dataset[0]
print(data.x.shape)
print(data.x[123])

print(data.edge_index.shape)
print(data.pos_edge_label_index)
