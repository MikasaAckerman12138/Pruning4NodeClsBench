import os
import time
import glob
import argparse
import os.path as osp
from tqdm.auto import tqdm
from typing import Optional, List, NamedTuple

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import ModuleList, Sequential, Linear, BatchNorm1d, ReLU, Dropout
from torch.optim.lr_scheduler import StepLR
import pytorch_lightning
#from torchmetrics import Accuracy
from pytorch_lightning.metrics import Accuracy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import (LightningDataModule, LightningModule, Trainer,
                               seed_everything)

from torch_sparse import SparseTensor
from torch_geometric.nn import SAGEConv, GATConv
from torch_geometric.data import NeighborSampler

from ogb.lsc import MAG240MDataset, MAG240MEvaluator

from root import ROOT

WITHOUT_LIGHTNING_V2 = int(pytorch_lightning.__version__.split('.')[0]) < 2


def get_compact_indices(target, subset):

    hashed_subset = []
    reshaped_target = target.unsqueeze(1)

    for i in range(0, len(subset), 16):
        batch = subset[16*i:min(16*(i+1), len(subset))]
        batch_idx = (reshaped_target == batch).any(-1).nonzero()
        hashed_subset.append(batch_idx)

    return torch.cat(hashed_subset, dim=0).squeeze(1)


def main():
    parser = argparse.ArgumentParser(description='MAG240M')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--model', type=str, default='gat',
                        choices=['gat', 'graphsage'])
    parser.add_argument('--sizes', type=str, default='25-15')
    parser.add_argument('--hidden_channels', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--in-memory', action='store_true')
    # exp relatead
    parser.add_argument('--mode', type=int, default=0)
    parser.add_argument('--start_sample_id', type=int, default=0)
    parser.add_argument('--end_sample_id', type=int, default=32)
    parser.add_argument('--fold', type=str, default='/mnt/ogb_datasets/mag240m_kddcup2021/ckpts')
    parser.add_argument('--infl', type=str, default='')
    args = parser.parse_args()
    print(args)

    if args.mode == 0:
        # use infer_corr.py
        pass
    else:
        dataset = MAG240MDataset(ROOT)
        train_idx = torch.from_numpy(dataset.get_idx_split('train'))

        train_masks = []
        for t in tqdm(range(args.start_sample_id, args.end_sample_id)):
            sub_train_idx = torch.load(os.path.join(args.fold, "{}.pt".format(t)))
            sub_train_idx = get_compact_indices(train_idx, sub_train_idx)
            mask = torch.zeros_like(train_idx, dtype=torch.bool).scatter_(0, sub_train_idx, True)
            train_masks.append(mask)
        trainset_mask = np.vstack(train_masks)
        inv_mask = np.logical_not(trainset_mask)

        cors = dict()
        for fn in os.listdir(args.fold):
            #if fn.endswith(".pth"):
            if fn.endswith("infer.pt"):
                trial = fn[:fn.find('_')]
                content = torch.load(os.path.join(args.fold, fn))
                cors[trial+"_train"] = content['train_corr']
                cors[trial+"_test"] = content['test_corr']

        correctness = (args.end_sample_id - args.start_sample_id) * [None]
        test_correctness = (args.end_sample_id - args.start_sample_id) * [None]
        for k, v in cors.items():
            if k.endswith('train'):
                correctness[int(k[:k.find('_')])] = v.numpy()
            elif k.endswith('test'):
                test_correctness[int(k[:k.find('_')])] = v.numpy()

        correctness = np.vstack(correctness)
        test_correctness = np.vstack(test_correctness)

        if args.infl == 'max':
            device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

            test_correctness = torch.from_numpy(test_correctness).float().to(device)
            # (M, Tr)
            trainset_mask = torch.from_numpy(trainset_mask).float().to(device)
            inv_mask = torch.from_numpy(inv_mask).float().to(device)
            eps = torch.zeros_like(torch.sum(trainset_mask, axis=0, keepdims=True)) + 1e-8
            test_indices = torch.arange(len(dataset.get_idx_split('valid'))).to(device)
            max_infl = torch.zeros_like(train_idx, dtype=torch.float32).to(device) - 1.5

            def _masked_dot(x, mask, esp=1e-8):
                # (B, M) * (M, Tr) = (B, Tr)
                return torch.matmul(x, mask) / torch.maximum(torch.sum(mask, axis=0, keepdims=True), esp)

            for i in tqdm(range(0, len(test_indices), 16)):
                batch_test_indices = test_indices[i:min(i+16, len(test_indices))]

                # (B, M)
                batch_test_correctness = test_correctness.T[batch_test_indices]

                # (B, Tr)
                batch_infl_est = _masked_dot(batch_test_correctness, trainset_mask, eps) - _masked_dot(batch_test_correctness, inv_mask, eps)

                # (Tr, )
                batch_max_infl = torch.max(batch_infl_est, 0)[0]

                max_infl = torch.maximum(max_infl, batch_max_infl)

            indices = train_idx.numpy().tolist()
            results = [(indices[i], float(max_infl[i].item())) for i in range(len(indices))]
            results = sorted(results, key=lambda x:x[1], reverse=True)
            with open(os.path.join("infl-{}".format(args.infl), "train.tsv"), 'w') as ops:
                for i in range(len(results)):
                    idx, infl = results[i]
                    ops.write("{}\t{}\n".format(idx, infl))
        elif args.infl == 'sum-abs':
            raise NotImplementedError("Don't use this")
        else:
            def _masked_avg(x, mask, axis=0, esp=1e-10):
                return (np.sum(x * mask, axis=axis) / np.maximum(np.sum(mask, axis=axis), esp)).astype(np.float32)

            def _masked_dot(x, mask, esp=1e-10):
                x = x.T.astype(np.float32)
                return (np.matmul(x, mask) / np.maximum(np.sum(mask, axis=0, keepdims=True), esp)).astype(np.float32)

            mem_est = _masked_avg(correctness, trainset_mask) - _masked_avg(correctness, inv_mask)

            indices = train_idx.numpy().tolist()
            results = [(indices[i], float(mem_est[i])) for i in range(len(indices))]
            results = sorted(results, key=lambda x:x[1], reverse=True)
            with open(os.path.join("mem", "train.tsv"), 'w') as ops:
                for i in range(len(results)):
                    idx, mem = results[i]
                    ops.write("{}\t{}\n".format(idx, mem))


if __name__ == "__main__":
    main()
