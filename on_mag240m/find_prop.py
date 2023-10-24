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
from torch_geometric.utils import is_undirected, to_undirected

from ogb.lsc import MAG240MDataset, MAG240MEvaluator

from root import ROOT
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
    parser = argparse.ArgumentParser(description='MAG240M (GraphSAINT)')
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

    dataset = MAG240MDataset(ROOT)
    split_idx = dataset.get_idx_split()
    #data = dataset[0]
    num_nodes = dataset.num_papers
    edge_index = dataset.edge_index('paper', 'cites', 'paper')
    edge_index = torch.from_numpy(edge_index)
    if not is_undirected(edge_index, num_nodes=num_nodes):
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    y = torch.from_numpy(dataset.all_paper_label).unsqueeze(1)
    assert num_nodes == y.size(0), "Inconsistent statistics {} vs {}".format(num_nodes, y.size(0))

    hard_sample, easy_sample = get_samples(args.al, args.ratio)
    print("#hard = {}, #easy = {}".format(len(hard_sample), len(easy_sample)))

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
    same_lb_flag = torch.logical_and((lb_of_src==lb_of_tgt), torch.logical_not(torch.isnan(lb_of_src))).long()
    same_cnt = torch.sum(same_lb_flag).item()
    denu = torch.logical_and(torch.logical_not(torch.isnan(lb_of_src)), torch.logical_not(torch.isnan(lb_of_tgt))).long().sum().item()
    print("Homophilic level of this graph is {} / {} = {}".format(same_cnt, denu, float(same_cnt) / denu))
    cnt_by_node = torch.zeros_like(degrees_t).long()
    torch_scatter.scatter(same_lb_flag, edge_index[0], out=cnt_by_node)

    hard_sample, easy_sample = torch.Tensor(hard_sample).long(), torch.Tensor(easy_sample).long()
    dgr_cnt_by_node = torch.zeros_like(degrees_t).long()
    valid_degrees = torch_scatter.scatter(torch.logical_and(torch.logical_not(torch.isnan(lb_of_src)), torch.logical_not(torch.isnan(lb_of_tgt))).long(), edge_index[0], out=dgr_cnt_by_node)
    homo_of_hard = (cnt_by_node[hard_sample] / (1e-8 + valid_degrees[hard_sample])).cpu().numpy()
    homo_of_easy = (cnt_by_node[easy_sample] / (1e-8 + valid_degrees[easy_sample])).cpu().numpy()

    """
    hard_sample, easy_sample = torch.Tensor(hard_sample).long(), torch.Tensor(easy_sample).long()
    bs = args.batch_size
    # homophilic levels of specific node(s)
    homo_of_hard = []
    for i in tqdm(range(0, len(hard_sample), bs)):
        node_batch = hard_sample[i:min(i+bs, len(hard_sample))]
        idx = (data.edge_index[0].unsqueeze(1) == node_batch.unsqueeze(0)).nonzero()
        idx_ = idx[:,0]
        inv_idx_ = idx[:,1]
        #idx1 = (data.edge_index[1].unsqueeze(1) == node_batch.unsqueeze(0)).nonzero()
        #idx1_ = idx1[:,0]
        #inv_idx1_ = idx1[:,1]

        src_edge_batch, tgt_edge_batch = data.edge_index[0][idx_], data.edge_index[1][idx_]
        lb_of_src, lb_of_tgt = data.y.squeeze(1)[src_edge_batch], data.y.squeeze(1)[tgt_edge_batch]
        batch_same_cnt, batch_degree = torch.zeros_like(node_batch), torch.zeros_like(node_batch)
        torch_scatter.scatter((lb_of_src==lb_of_tgt).long(), inv_idx_, out=batch_same_cnt)
        torch_scatter.scatter(torch.ones_like(lb_of_src).long(), inv_idx_, out=batch_degree)
        homo_of_hard.append(batch_same_cnt / (batch_degree + 1e-8))
    homo_of_hard = torch.cat(homo_of_hard).numpy()
    homo_of_easy = []
    for i in tqdm(range(0, len(easy_sample), bs)):
        node_batch = easy_sample[i:min(i+bs, len(easy_sample))]
        idx = (data.edge_index[0].unsqueeze(1) == node_batch.unsqueeze(0)).nonzero()
        idx_ = idx[:,0]
        inv_idx_ = idx[:,1]
        src_edge_batch, tgt_edge_batch = data.edge_index[0][idx_], data.edge_index[1][idx_]
        lb_of_src, lb_of_tgt = data.y.squeeze(1)[src_edge_batch], data.y.squeeze(1)[tgt_edge_batch]
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
