import os
import argparse

import random
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F

import torch_scatter
import torch_geometric as pyg
from torch_geometric.data import GraphSAINTRandomWalkSampler, NeighborSampler
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import subgraph

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from logger import Logger

from utils import *


def get_samples(methods, ratio):
    ranks = []
    for fold in methods.split(','):
        ranks.append(aug_rank_by_pct(load_rank_list(fold)))
    ensembled_rank = dict(ranks[0])
    for i in range(1, len(ranks)):
        for tp in ranks[i]:
            ensembled_rank[tp[0]] += tp[1]
    ensembled_rank = sorted([(k, v) for k, v in ensembled_rank.items()], key=lambda x:x[1], reverse=True)

    return [tp[0] for tp in ensembled_rank[:int(ratio*len(ensembled_rank))]], [tp[0] for tp in ensembled_rank[-int(ratio*len(ensembled_rank)):]]


def main():
    parser = argparse.ArgumentParser(description='OGBN-Products (GraphSAINT)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=32)
    # exp relatead
    parser.add_argument('--al', type=str, default='mem')
    parser.add_argument('--ratio', type=float, default=0.2)
    args = parser.parse_args()
    print(args)

    dataset = PygNodePropPredDataset('ogbn-papers100M', root="/mnt/ogb_datasets")
    split_idx = dataset.get_idx_split()
    data = dataset[0]
    nid_remap = torch.zeros((data.num_nodes,), dtype=torch.long)
    torch_scatter.scatter(torch.arange(len(split_idx['train']) + len(split_idx['valid']) + len(split_idx['test'])), torch.cat([split_idx['train'], split_idx['valid'], split_idx['test']]), out=nid_remap)

    sgc_dict = torch.load('/mnt/ogb_datasets/ogbn_papers100M/sgc_dict.pt')
    y = sgc_dict['label']
    print(y.shape, y.dtype)
    num_nodes = y.size(0)
    edge_index = torch.load('/mnt/ogb_datasets/ogbn_papers100M/edge_index.pt')
    edge_index = torch.stack([nid_remap[edge_index[0]], nid_remap[edge_index[1]]])

    hard_sample, easy_sample = get_samples(args.al, args.ratio)
    print("#hard = {}, #easy = {}".format(len(hard_sample), len(easy_sample)))

    # class distribution
    #cnt_by_lb = torch.zeros((172,)).long()
    #torch_scatter.scatter(torch.ones_like(y.squeeze(1)).long(), y.squeeze(1), out=cnt_by_lb)
    #ratios = sorted([(cid, ratio) for cid, ratio in enumerate((cnt_by_lb / cnt_by_lb.sum()).cpu().numpy().tolist())], key=lambda x:x[1], reverse=True)
    #print(ratios[:50])
    ## hard
    #cnt_by_lb = torch.zeros((172,)).long()
    #torch_scatter.scatter(torch.ones((len(hard_sample),)).long(), y.squeeze(1)[torch.Tensor(hard_sample).long()], out=cnt_by_lb)
    #ratios = sorted([(cid, ratio) for cid, ratio in enumerate((cnt_by_lb / cnt_by_lb.sum()).cpu().numpy().tolist())], key=lambda x:x[1], reverse=True)
    #print(ratios[:50])
    ## easy
    #cnt_by_lb = torch.zeros((172,)).long()
    #torch_scatter.scatter(torch.ones((len(easy_sample),)).long(), y.squeeze(1)[torch.Tensor(easy_sample).long()], out=cnt_by_lb)
    #ratios = sorted([(cid, ratio) for cid, ratio in enumerate((cnt_by_lb / cnt_by_lb.sum()).cpu().numpy().tolist())], key=lambda x:x[1], reverse=True)
    #print(ratios[:50])

    # degree
    degrees_t = pyg.utils.degree(edge_index[0], num_nodes)
    degrees = degrees_t.numpy()
    print("Avg degree of this graph is {}".format(degrees.mean()))
    degrees_of_hard = degrees[hard_sample]
    degrees_of_easy = degrees[easy_sample]
    mean_of_hard, std_of_hard = np.mean(degrees_of_hard), np.std(degrees_of_hard)
    mean_of_easy, std_of_easy = np.mean(degrees_of_easy), np.std(degrees_of_easy)
    print("mean degree (std): {} ({}) of hard v.s. {} ({}) of easy".format(mean_of_hard, std_of_hard, mean_of_easy, std_of_easy))

    # page rank score
    pr = torch.load(os.path.join('age', "pr.pt"))
    pr = np.asarray([len(pr) * pr[i] for i in range(len(pr))])
    print("Avg pr of this graph is {}".format(pr.mean()))
    pr_of_hard = pr[hard_sample]
    pr_of_easy = pr[easy_sample]
    mean_of_hard, std_of_hard = np.mean(pr_of_hard), np.std(pr_of_hard)
    mean_of_easy, std_of_easy = np.mean(pr_of_easy), np.std(pr_of_easy)
    print("mean pr (std): {} ({}) of hard v.s. {} ({}) of easy".format(mean_of_hard, std_of_hard, mean_of_easy, std_of_easy))

    # chaotic neighborhood
    # avg homophilic level
    lb_of_src, lb_of_tgt = y.squeeze(1)[edge_index[0]], y.squeeze(1)[edge_index[1]]
    same_lb_flag = (lb_of_src==lb_of_tgt).long()
    print("Homophilic level of this graph is {}".format( torch.sum(same_lb_flag).item() / len(lb_of_src) ))
    cnt_by_node = torch.zeros_like(degrees_t).long()
    torch_scatter.scatter(same_lb_flag, edge_index[0], out=cnt_by_node)

    hard_sample, easy_sample = torch.Tensor(hard_sample).long(), torch.Tensor(easy_sample).long()
    homo_of_hard = (cnt_by_node[hard_sample] / (1e-8 + degrees_t[hard_sample])).cpu().numpy()
    homo_of_easy = (cnt_by_node[easy_sample] / (1e-8 + degrees_t[easy_sample])).cpu().numpy()
    """
    bs = args.batch_size
    # homophilic levels of specific node(s)
    homo_of_hard = []
    for i in tqdm(range(0, len(hard_sample), bs)):
        node_batch = hard_sample[i:min(i+bs, len(hard_sample))]
        idx = (edge_index[0].unsqueeze(1) == node_batch.unsqueeze(0)).nonzero()
        idx_ = idx[:,0]
        inv_idx_ = idx[:,1]
        src_edge_batch, tgt_edge_batch = edge_index[0][idx_], edge_index[1][idx_]
        lb_of_src, lb_of_tgt = y.squeeze(1)[src_edge_batch], y.squeeze(1)[tgt_edge_batch]
        batch_same_cnt, batch_degree = torch.zeros_like(node_batch), torch.zeros_like(node_batch)
        torch_scatter.scatter((lb_of_src==lb_of_tgt).long(), inv_idx_, out=batch_same_cnt)
        torch_scatter.scatter(torch.ones_like(lb_of_src).long(), inv_idx_, out=batch_degree)
        homo_of_hard.append(batch_same_cnt / (batch_degree + 1e-8))
    homo_of_hard = torch.cat(homo_of_hard).numpy()
    homo_of_easy = []
    for i in tqdm(range(0, len(easy_sample), bs)):
        node_batch = easy_sample[i:min(i+bs, len(easy_sample))]
        idx = (edge_index[0].unsqueeze(1) == node_batch.unsqueeze(0)).nonzero()
        idx_ = idx[:,0]
        inv_idx_ = idx[:,1]
        src_edge_batch, tgt_edge_batch = edge_index[0][idx_], edge_index[1][idx_]
        lb_of_src, lb_of_tgt = y.squeeze(1)[src_edge_batch], y.squeeze(1)[tgt_edge_batch]
        batch_same_cnt, batch_degree = torch.zeros_like(node_batch), torch.zeros_like(node_batch)
        torch_scatter.scatter((lb_of_src==lb_of_tgt).long(), inv_idx_, out=batch_same_cnt)
        torch_scatter.scatter(torch.ones_like(lb_of_src).long(), inv_idx_, out=batch_degree)
        homo_of_easy.append(batch_same_cnt / (batch_degree + 1e-8))
    homo_of_easy = torch.cat(homo_of_easy).numpy()
    """
    mean_of_hard, std_of_hard = np.mean(homo_of_hard), np.std(homo_of_hard)
    mean_of_easy, std_of_easy = np.mean(homo_of_easy), np.std(homo_of_easy)
    print("mean homo (std): {} ({}) of hard v.s. {} ({}) of easy".format(mean_of_hard, std_of_hard, mean_of_easy, std_of_easy))


if __name__ == "__main__":
    main()
