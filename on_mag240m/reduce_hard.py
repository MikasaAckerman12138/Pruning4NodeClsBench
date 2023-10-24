import os
import argparse

from typing import Optional, List, NamedTuple

from tqdm.auto import tqdm

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torch_sparse import SparseTensor
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.data import NeighborSampler
from pytorch_lightning import (LightningDataModule, LightningModule, Trainer,
                               seed_everything)

from ogb.lsc import MAG240MDataset

from root import ROOT
from utils import *


class Batch(NamedTuple):
    x: Tensor
    adjs_t: List[SparseTensor]

    def to(self, *args, **kwargs):
        return Batch(
            x=self.x.to(*args, **kwargs),
            adjs_t=[adj_t.to(*args, **kwargs) for adj_t in self.adjs_t],
        )


@torch.no_grad()
def infer(model, subgraph_loader, device):
    model.eval()

    pbar = tqdm(total=1339433)
    pbar.set_description('Propagation')

    all_x = []

    for mini_batch in subgraph_loader:
        mini_batch = mini_batch.to(device)
        x = mini_batch.x
        adjs_t = mini_batch.adjs_t

        for i, adj_t in enumerate(adjs_t):
            x = model(x, adj_t)

        all_x.append(x.cpu())
        pbar.update(x.size(0))

    pbar.close()

    return torch.cat(all_x, dim=0)


def main():
    parser = argparse.ArgumentParser(description='on MAG240M')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--mode', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_pick', type=int, default=2)
    parser.add_argument('--patience', type=int, default=15)
    args = parser.parse_args()
    print(args)

    seed_everything(args.seed)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = MAG240MDataset(root=ROOT)

    if args.mode == 0:
        train_idx = torch.from_numpy(dataset.get_idx_split('train'))
        #train_idx.share_memory_()
        print(train_idx.shape)
        val_idx = torch.from_numpy(dataset.get_idx_split('valid'))
        #val_idx.share_memory_()
        print(val_idx.shape)
        test_idx = torch.from_numpy(dataset.get_idx_split('test-dev'))
        #test_idx.share_memory_()
        print(test_idx.shape)
        all_idx = torch.cat([train_idx, val_idx, test_idx])
        all_idx.share_memory_()

        all_x = torch.from_numpy(dataset.all_paper_feat).share_memory_().to(torch.float)

        path = f'{dataset.dir}/paper_to_paper_symmetric.pt'
        adj_t = torch.load(path)
        # Pre-compute GCN normalization.
        adj_t = adj_t.set_diag()
        deg = adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-1.0)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj_t = deg_inv_sqrt.view(-1, 1) * adj_t

        def convert_batch(batch_size, n_id, adjs):
            x = all_x[n_id].to(torch.float)
            return Batch(x=x, adjs_t=[adj_t for adj_t, _, _ in adjs])

        subgraph_loader = NeighborSampler(adj_t, node_idx=all_idx, sizes = args.num_layers * [160], return_e_id=False,
                                          transform=convert_batch, batch_size=args.batch_size, shuffle=False)

        conv = GCNConv(all_x.shape[-1], all_x.shape[-1], normalize=False, bias=False)
        conv.to(device)
        #conv.reset_parameters()
        conv.lin.weight.data = torch.eye(all_x.shape[-1], device=device)

        all_x = infer(conv, subgraph_loader, device)

        torch.save(all_x, os.path.join(dataset.dir, "rw_feat_{}.pt".format(args.seed)))
    else:
        zs = .0
        for seed in [12, 42, 123]:
            zs += torch.load(os.path.join(dataset.dir, "rw_feat_{}.pt".format(seed)))
        zs = zs / 3
        print(zs.shape, zs.dtype)

        original_train_idx = dataset.get_idx_split('train').tolist()
        original_val_idx = dataset.get_idx_split('valid').tolist()
        original_test_idx = dataset.get_idx_split('test-dev').tolist()

        train_idx = torch.arange(len(original_train_idx)).to(device)
        test_idx = torch.arange(len(original_train_idx), len(original_train_idx)+len(original_val_idx)).to(device)

        train_zs = zs[train_idx].to(device)
        test_zs = zs[test_idx].to(device)

        def calc_dist(a, b):
            pairwise_dist = ((a.unsqueeze(1) - b.unsqueeze(0)) ** 2).sum(-1).sqrt()
            return pairwise_dist

        pagerank = load_rank_list('centrality')

        no_inc = 0
        relaxed = False
        init_tr_idx = original_train_idx.index(pagerank[0])
        rank = [pagerank[0]]
        tr_flag = torch.ones_like(train_idx)
        tr_flag[init_tr_idx] = 0
        tr_flag = tr_flag.bool()
        cur_dist = calc_dist(train_zs[init_tr_idx].unsqueeze(0), test_zs).squeeze(0)
        ts_flag = torch.ones_like(test_idx).bool()
        pbar = tqdm(total=len(train_idx))

        f = open("fancy.out", "w")

        while len(rank) < len(train_idx):
            considered_dist = ts_flag * cur_dist
            batch_val, batch_idx = torch.topk(considered_dist, args.num_pick)
            valid_query_idx_mask = ts_flag[batch_idx]
            batch_val = batch_val[valid_query_idx_mask]
            batch_idx = batch_idx[valid_query_idx_mask]
            if len(batch_idx) == 0:
                # refresh
                print("Refresh test flag at 1.0!")
                ts_flag = torch.ones_like(test_idx).bool()
                init_tr_idx = -1
                for pr_idx in pagerank:
                    if pr_idx not in rank:
                        init_tr_idx = original_train_idx.index(pr_idx)
                        rank.append(pr_idx)
                        break
                assert init_tr_idx != -1, "No left for initiation!!!"
                print("Starting from {}".format(pr_idx))
                cur_dist = calc_dist(train_zs[init_tr_idx].unsqueeze(0), test_zs).squeeze(0)
                tr_flag[init_tr_idx] = False
                relaxed = False
                continue

            query_zs = test_zs[batch_idx]
            key_zs = train_zs[tr_flag]
            pairwise_dist = calc_dist(key_zs, query_zs)
            min_dist, min_dist_idx = pairwise_dist.min(0)

            if relaxed:
                pick_idx = tr_flag.nonzero().squeeze(1)[min_dist_idx]
                picked_dist = calc_dist(train_zs[pick_idx], test_zs)
                useful_idx = (picked_dist < cur_dist).sum(axis=1).nonzero().squeeze(1)
                pick_idx = pick_idx[useful_idx]
                pick_tr_idx = list(set([original_train_idx[j] for j in pick_idx.cpu().tolist()]))
                rank.extend(pick_tr_idx)
                tr_flag[pick_idx] = False
                if len(pick_tr_idx) > 0:
                    cur_dist = torch.minimum(cur_dist, picked_dist.min(0)[0])

                    # for analysis
                    if 2*len(rank) >= len(test_idx):
                        break
                    outlier_val, _ = torch.topk(cur_dist, len(rank))
                    mf_val, _ = torch.topk(cur_dist, 2*len(rank))
                    mf_avg_val = (mf_val.sum() - outlier_val.sum()).cpu().item() / len(rank)
                    f.write("{}\t{}\n".format( len(rank), mf_avg_val))

            else:
                useful_flag = min_dist < batch_val
                pick_idx = min_dist_idx[useful_flag]
                pick_idx = tr_flag.nonzero().squeeze(1)[pick_idx]
                pick_tr_idx = list(set([original_train_idx[j] for j in train_idx[pick_idx].cpu().tolist()]))
                rank.extend(pick_tr_idx)
                tr_flag[pick_idx] = False
                #ts_flag[batch_idx[torch.logical_not(useful_flag)]] = False
                if len(pick_tr_idx) > 0: 
                    cur_dist = torch.minimum(cur_dist, calc_dist(train_zs[pick_idx], test_zs).min(0)[0])

                    # for analysis
                    if 2*len(rank) >= len(test_idx):
                        break
                    outlier_val, _ = torch.topk(cur_dist, len(rank))
                    mf_val, _ = torch.topk(cur_dist, 2*len(rank))
                    mf_avg_val = (mf_val.sum() - outlier_val.sum()).cpu().item() / len(rank)
                    f.write("{}\t{}\n".format( len(rank), mf_avg_val))

            ts_flag[batch_idx] = False

            if len(pick_tr_idx) > 0:
                no_inc = 0
            else:
                no_inc += 1
                if no_inc >= args.patience:
                    if len(rank) >= 0.8 * len(train_idx):
                        print("Greedily picked {} training nodes".format(len(rank)))
                        rank_as_set = set(rank)
                        other_nodes = [vid for vid in original_train_idx if vid not in rank_as_set]
                        np.random.shuffle(other_nodes)
                        rank.extend(other_nodes)
                        break
                    else:
                        print("Early no inc happens at {}!".format(len(rank)/float(len(train_idx))))
                        if not relaxed:
                            relaxed = True
                            no_inc = 0
                            print("relax selection criterion!")
                        else:
                            print("Refresh test flag at {}!".format( (torch.logical_not(ts_flag)).sum().cpu().item() / float(len(ts_flag)) ))
                            ts_flag = torch.ones_like(test_idx).bool()
                            init_tr_idx = -1
                            for pr_idx in pagerank:
                                if pr_idx not in rank:
                                    init_tr_idx = original_train_idx.index(pr_idx)
                                    rank.append(pr_idx)
                                    break
                            assert init_tr_idx != -1, "No left for initiation!!!"
                            print("Starting from {}".format(pr_idx))
                            cur_dist = calc_dist(train_zs[init_tr_idx].unsqueeze(0), test_zs).squeeze(0)
                            tr_flag[init_tr_idx] = False
                            relaxed = False
                            no_inc = 0

            pbar.update(len(pick_tr_idx))

        pbar.close()

        f.close()
        #with open(os.path.join("ours", "train.tsv"), 'w') as ops:
        #    for i in range(len(rank)):
        #        ops.write("{}\t{}\n".format(rank[i], float(len(rank)-i) / len(rank)))


if __name__=='__main__':
    main()
