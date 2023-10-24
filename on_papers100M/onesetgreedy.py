import os
import argparse

from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from utils import *


def main():
    parser = argparse.ArgumentParser(description='OGBN-papers100M (MLP)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num_sgc_iterations', type=int, default = 3)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_pick', type=int, default=16)
    parser.add_argument('--patience', type=int, default=15)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    sgc_dict = torch.load('/mnt/ogb_datasets/ogbn_papers100M/sgc_dict.pt')
    x = sgc_dict['sgc_embedding'][args.num_sgc_iterations]
    print(x.shape)
    split_idx = sgc_dict['split_idx']
    train_idx = split_idx['train'].to(device)
    test_idx = split_idx['test'].to(device)
    print(len(train_idx), len(split_idx['valid']), len(test_idx))

    train_zs = x[split_idx['train']].to(device)
    test_zs = x[split_idx['test']].to(device)
    #test_idx = split_idx['test'].to(device)

    def calc_dist(a, b):
        pairwise_dist = ((a.unsqueeze(1) - b.unsqueeze(0)) ** 2).sum(-1).sqrt()
        return pairwise_dist

    pagerank = load_rank_list('centrality')

    no_inc = 0
    relaxed = False
    rank = [pagerank[0]]
    tr_flag = torch.ones_like(train_idx)
    tr_flag[rank[0]] = 0
    tr_flag = tr_flag.bool()
    cur_dist = calc_dist(train_zs[rank[0]].unsqueeze(0), train_zs).squeeze(0)
    test_dist = calc_dist(train_zs[rank[0]].unsqueeze(0), test_zs).squeeze(0)
    ts_flag = torch.ones_like(test_idx).bool()
    pbar = tqdm(total=len(train_idx))

    f = open("naive.out", 'w')

    while len(rank) < len(train_idx):
        pick_val, pick_idx = torch.topk(cur_dist, 1)
        useful_flag = pick_val > torch.zeros_like(pick_val)
        pick_idx = pick_idx[useful_flag]
        rank.extend(pick_idx)

        if len(pick_idx) > 0:
            cur_dist = torch.minimum(cur_dist, calc_dist(train_zs[pick_idx], train_zs).min(0)[0])
            test_dist = torch.minimum(test_dist, calc_dist(train_zs[pick_idx], test_zs).min(0)[0])

            # for analysis
            if 2*len(rank) >= len(test_idx):
                break

            outlier_val, _ = torch.topk(test_dist, len(rank))
            mf_val, _ = torch.topk(test_dist, 2*len(rank))
            mf_avg_val = (mf_val.sum() - outlier_val.sum()).cpu().item() / len(rank)
            f.write("{}\t{}\n".format(len(rank), mf_avg_val))
            if 2*len(rank) >= len(train_idx):
                break
        else:
            break

        pbar.update(len(pick_idx))

    pbar.close()

    f.close()
    #with open(os.path.join("ours", "train_p{}.tsv".format(args.patience)), 'w') as ops:
    #    for i in range(len(rank)):
    #        ops.write("{}\t{}\n".format(rank[i], i))


if __name__ == "__main__":
    main()
