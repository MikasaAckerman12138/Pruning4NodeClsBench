import os
import os.path as osp
import argparse
import time
import glob

from typing import Optional, List, NamedTuple
from tqdm.auto import tqdm
import networkx as nx
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch_geometric.utils import subgraph, to_undirected

from torch_sparse import SparseTensor
import torch_geometric as pyg
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

from utils import *

from ogb.lsc import MAG240MDataset

from root import ROOT

import scipy as sp
from scipy import stats

WITHOUT_LIGHTNING_V2 = int(pytorch_lightning.__version__.split('.')[0]) < 2


class Batch(NamedTuple):
    x: Tensor
    y: Tensor
    adjs_t: List[SparseTensor]

    def to(self, *args, **kwargs):
        return Batch(
            x=self.x.to(*args, **kwargs),
            y=self.y.to(*args, **kwargs),
            adjs_t=[adj_t.to(*args, **kwargs) for adj_t in self.adjs_t],
        )


class MAG240M(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, sizes: List[int],
                 in_memory: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.sizes = sizes
        self.in_memory = in_memory

    @property
    def num_features(self) -> int:
        return 768

    @property
    def num_classes(self) -> int:
        return 153

    def prepare_data(self):
        dataset = MAG240MDataset(self.data_dir)
        path = f'{dataset.dir}/paper_to_paper_symmetric.pt'
        if not osp.exists(path):
            t = time.perf_counter()
            print('Converting adjacency matrix...', end=' ', flush=True)
            edge_index = dataset.edge_index('paper', 'cites', 'paper')
            edge_index = torch.from_numpy(edge_index)
            adj_t = SparseTensor(
                row=edge_index[0], col=edge_index[1],
                sparse_sizes=(dataset.num_papers, dataset.num_papers),
                is_sorted=True)
            torch.save(adj_t.to_symmetric(), path)
            print(f'Done! [{time.perf_counter() - t:.2f}s]')

    def setup(self, stage: Optional[str] = None):
        t = time.perf_counter()
        print('Reading dataset...', end=' ', flush=True)
        dataset = MAG240MDataset(self.data_dir)

        self.train_idx = torch.from_numpy(dataset.get_idx_split('train'))
        self.train_idx = self.train_idx
        self.train_idx.share_memory_()
        self.val_idx = torch.from_numpy(dataset.get_idx_split('valid'))
        self.val_idx.share_memory_()
        self.test_idx = torch.from_numpy(dataset.get_idx_split('test-dev'))
        self.test_idx.share_memory_()

        if self.in_memory:
            self.x = torch.from_numpy(dataset.all_paper_feat).share_memory_()
        else:
            self.x = dataset.paper_feat
        self.y = torch.from_numpy(dataset.all_paper_label)

        path = f'{dataset.dir}/paper_to_paper_symmetric.pt'
        self.adj_t = torch.load(path)
        print(f'Done! [{time.perf_counter() - t:.2f}s]')

    def train_dataloader(self):
        return NeighborSampler(self.adj_t, node_idx=self.train_idx,
                               sizes=self.sizes, return_e_id=False,
                               transform=self.convert_batch,
                               batch_size=self.batch_size, shuffle=False,
                               num_workers=4)

    def val_dataloader(self):
        return NeighborSampler(self.adj_t, node_idx=self.val_idx,
                               sizes=self.sizes, return_e_id=False,
                               transform=self.convert_batch,
                               batch_size=self.batch_size, num_workers=2)

    def test_dataloader(self):  # Test best validation model once again.
        return NeighborSampler(self.adj_t, node_idx=self.val_idx,
                               sizes=self.sizes, return_e_id=False,
                               transform=self.convert_batch,
                               batch_size=self.batch_size, num_workers=2)

    def hidden_test_dataloader(self):
        return NeighborSampler(self.adj_t, node_idx=self.test_idx,
                               sizes=self.sizes, return_e_id=False,
                               transform=self.convert_batch,
                               batch_size=self.batch_size, num_workers=3)

    def convert_batch(self, batch_size, n_id, adjs):
        if self.in_memory:
            x = self.x[n_id].to(torch.float)
        else:
            x = torch.from_numpy(self.x[n_id.numpy()]).to(torch.float)
        y = self.y[n_id[:batch_size]].to(torch.long)
        return Batch(x=x, y=y, adjs_t=[adj_t for adj_t, _, _ in adjs])


class GNN(LightningModule):
    def __init__(self, model: str, in_channels: int, out_channels: int,
                 hidden_channels: int, num_layers: int, heads: int = 4,
                 dropout: float = 0.5):
        super().__init__()
        self.save_hyperparameters()
        self.model = model.lower()
        self.dropout = dropout

        self.convs = ModuleList()
        self.norms = ModuleList()
        self.skips = ModuleList()

        if self.model == 'gat':
            self.convs.append(
                GATConv(in_channels, hidden_channels // heads, heads))
            self.skips.append(Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 1):
                self.convs.append(
                    GATConv(hidden_channels, hidden_channels // heads, heads))
                self.skips.append(Linear(hidden_channels, hidden_channels))

        elif self.model == 'graphsage':
            self.convs.append(SAGEConv(in_channels, hidden_channels))
            for _ in range(num_layers - 1):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        for _ in range(num_layers):
            self.norms.append(BatchNorm1d(hidden_channels))

        self.mlp = Sequential(
            Linear(hidden_channels, hidden_channels),
            BatchNorm1d(hidden_channels),
            ReLU(inplace=True),
            Dropout(p=self.dropout),
            Linear(hidden_channels, out_channels),
        )

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

    def forward(self, x: Tensor, adjs_t: List[SparseTensor]) -> Tensor:
        for i, adj_t in enumerate(adjs_t):
            x_target = x[:adj_t.size(0)]
            x = self.convs[i]((x, x_target), adj_t)
            if self.model == 'gat':
                x = x + self.skips[i](x_target)
                x = F.elu(self.norms[i](x))
            elif self.model == 'graphsage':
                x = F.relu(self.norms[i](x))
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.mlp(x)

    def training_step(self, batch, batch_idx: int):
        y_hat = self(batch.x, batch.adjs_t)
        train_loss = F.cross_entropy(y_hat, batch.y)
        self.train_acc(y_hat.softmax(dim=-1), batch.y)
        self.log('train_acc', self.train_acc, prog_bar=True, on_step=False,
                 on_epoch=True)
        return train_loss

    def validation_step(self, batch, batch_idx: int):
        y_hat = self(batch.x, batch.adjs_t)
        self.val_acc(y_hat.softmax(dim=-1), batch.y)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx: int):
        y_hat = self(batch.x, batch.adjs_t)
        self.test_acc(y_hat.softmax(dim=-1), batch.y)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = StepLR(optimizer, step_size=25, gamma=0.25)
        return [optimizer], [scheduler]

def my_pagerank_scipy(
    N,
    A,
    alpha=0.85,
    max_iter=100,
    tol=1.0e-10,
    weight="weight",
):
    if N == 0:
        return {}

    nodelist = list(range(N))
    print(A.shape)
    S = A.sum(axis=1)
    print(S.shape)
    S[S != 0] = 1.0 / S[S != 0]
    # TODO: csr_array
    Q = sp.sparse.csr_array(sp.sparse.spdiags(S.T, 0, *A.shape))
    A = Q @ A

    x = np.repeat(1.0 / N, N)

    # Personalization vector
    p = np.repeat(1.0 / N, N)
    
    # Dangling nodes
    dangling_weights = p
    is_dangling = np.where(S == 0)[0]

    # power iteration: make up to max_iter iterations
    for _ in tqdm(range(max_iter)):
        xlast = x
        x = alpha * (x @ A + sum(x[is_dangling]) * dangling_weights) + (1 - alpha) * p
        # check convergence, l1 norm
        err = np.absolute(x - xlast).sum()
        print(err)
        if err < N * tol:
            return dict(zip(nodelist, map(float, x)))
    raise nx.PowerIterationFailedConvergence(max_iter)


def val2pct(vals):
    arr = [(i, vals[i]) for i in range(len(vals))]
    arr = sorted(arr, key=lambda x:x[1], reverse=True)
    pct = len(arr) * [None]
    for i in range(len(arr)):
        pct[arr[i][0]] = (len(arr) - i) / float(len(arr))
    return np.asarray(pct)


def export_rank(metric, output_fold, indices=None):
    if not isinstance(next(iter(metric)), tuple):
        results = [(int(indices[i]), metric[i]) for i in range(len(indices))] if indices is not None else [(i, metric[i]) for i in range(len(metric))]
    else:
        results = metric
    results = sorted(results, key=lambda x:x[1], reverse=True)
    with open(os.path.join(output_fold, "train.tsv"), 'w') as ops:
        for i in range(len(results)):
            idx, val = results[i]
            ops.write("{}\t{}\n".format(idx, val))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate AGE')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--sizes', type=str, default='25-15')
    parser.add_argument('--in-memory', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--model', type=str, default='gat',
                        choices=['gat', 'graphsage'])
    # exp relatead
    parser.add_argument('--mode', type=int, default=0)
    parser.add_argument('--num_trials', type=int, default=8)
    parser.add_argument('--metric', type=str, default='uncertainty')
    parser.add_argument('--alpha', type=float, default=0.3333)
    parser.add_argument('--beta', type=float, default=0.3333)
    parser.add_argument('--fold', type=str, default='age')
    args = parser.parse_args()
    args.sizes = [int(i) for i in args.sizes.split('-')]
    print(args)

    if args.mode == 0:
        if args.metric == 'uncertainty':
            datamodule = MAG240M(ROOT, args.batch_size, args.sizes, args.in_memory)

            dirs = glob.glob(f'logs/{args.model}/lightning_logs/*')
            logdir = f'logs/{args.model}/lightning_logs/version_0'
            print(f'Evaluating saved model in {logdir}...')
            ckpt = glob.glob(f'{logdir}/checkpoints/*')[0]
            if WITHOUT_LIGHTNING_V2:
              trainer = Trainer(gpus=args.device, resume_from_checkpoint=ckpt)
            else:
              trainer = Trainer(devices=len(args.device.split(',')), resume_from_checkpoint=ckpt)
            model = GNN.load_from_checkpoint(checkpoint_path=ckpt,
                                             hparams_file=f'{logdir}/hparams.yaml')

            datamodule.batch_size = 512
            datamodule.sizes = [160] * len(args.sizes)  # (Almost) no sampling...

            trainer.test(model=model, datamodule=datamodule)

            loader = datamodule.train_dataloader()

            results = dict()
            for t in range(args.num_trials):
                if t:
                    dirs = glob.glob(f'logs/{args.model}/lightning_logs/*')
                    logdir = f'logs/{args.model}/lightning_logs/version_{t}'
                    print(f'Evaluating saved model in {logdir}...')
                    ckpt = glob.glob(f'{logdir}/checkpoints/*')[0]
                    model = GNN.load_from_checkpoint(checkpoint_path=ckpt,
                                                     hparams_file=f'{logdir}/hparams.yaml')
                model.eval()
                device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
                model.to(device)
                ents = []

                for batch in tqdm(loader):
                    batch = batch.to(device)
                    with torch.no_grad():
                        out = model(batch.x, batch.adjs_t)
                        out = F.softmax(out, dim=1)
                        m = torch.distributions.categorical.Categorical(probs=out)
                        uncertainty = m.entropy()
                        ents.append(uncertainty.cpu())

                results[str(t)] = torch.cat(ents, 0)
                print("accumulated {} trials".format(t+1))

            torch.save(results, os.path.join('age', 'uncertainty.pt'))

        elif args.metric == 'centrality':
            path = '/mnt/ogb_datasets/mag240m_kddcup2021/paper_to_paper_symmetric.pt'
            adj_t = torch.load(path)

            N = adj_t.sparse_sizes()[0]
            print(N)

            #adj = adj_t.to_scipy(layout='csr')
            row, col = adj_t.storage.row(), adj_t.storage.col()
            adj = sp.sparse.coo_array((np.ones(len(row)), (row.numpy(), col.numpy())), shape=(N, N), dtype=float)
            adj = adj.asformat('csr')
            print(adj.shape, adj.dtype)

            # Pagerank score calculation
            pr = my_pagerank_scipy(N, adj)

            torch.save(pr, os.path.join('age', 'pr.pt'))

        elif args.metric == 'density':
            datamodule = MAG240M(ROOT, args.batch_size, args.sizes, args.in_memory)

            dirs = glob.glob(f'logs/{args.model}/lightning_logs/*')
            logdir = f'logs/{args.model}/lightning_logs/version_0'
            print(f'Evaluating saved model in {logdir}...')
            ckpt = glob.glob(f'{logdir}/checkpoints/*')[0]
            if WITHOUT_LIGHTNING_V2:
              trainer = Trainer(gpus=args.device, resume_from_checkpoint=ckpt)
            else:
              trainer = Trainer(devices=len(args.device.split(',')), resume_from_checkpoint=ckpt)
            model = GNN.load_from_checkpoint(checkpoint_path=ckpt,
                                             hparams_file=f'{logdir}/hparams.yaml')

            datamodule.batch_size = 512
            datamodule.sizes = [160] * len(args.sizes)  # (Almost) no sampling...

            trainer.test(model=model, datamodule=datamodule)

            loader = datamodule.train_dataloader()

            model.eval()
            device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
            model.to(device)

            train_reps = []
            for batch in tqdm(loader):
                batch = batch.to(device)
                with torch.no_grad():
                    out = model(batch.x, batch.adjs_t)
                    out = F.softmax(out, dim=1)
                    train_reps.append(out.cpu())

            train_reps = torch.cat(train_reps, 0)

            loader = datamodule.test_dataloader()

            test_reps = []
            for batch in tqdm(loader):
                batch = batch.to(device)
                with torch.no_grad():
                    out = model(batch.x, batch.adjs_t)
                    out = F.softmax(out, dim=1)
                    test_reps.append(out.cpu())

            test_reps = torch.cat(test_reps, 0)
 
            prob = torch.cat([train_reps, test_reps], axis=0).numpy()

            # k-means
            from sklearn.cluster import KMeans
            from sklearn.metrics.pairwise import euclidean_distances
            NCL = datamodule.num_classes // 2
            kmeans = KMeans(n_clusters=NCL, random_state=42).fit(prob)
            ed = euclidean_distances(prob, kmeans.cluster_centers_)
            ed_score = np.min(ed, axis=1)#the larger ed_score is, the far that node is away from cluster centers, the less representativeness the node is
            #edprec = np.asarray([percd(ed_score, i) for i in range(len(ed_score))])
            torch.save(torch.from_numpy(ed_score), os.path.join('age', 'density.pt'))
        else:
            raise ValueError("No this metric now")

    else:
        dataset = MAG240MDataset(ROOT)
        train_idx = dataset.get_idx_split()['train']

        pr = torch.load(os.path.join(args.fold, "pr.pt"))
        pr = np.asarray([(1112392 + 138949 + 88092) * pr[i] for i in range(len(pr))])
        pr = [float(pr[train_idx[i]]) for i in range(len(train_idx))]
        #pr = [(int(train_idx[i]), float(pr[train_idx[i]])) for i in range(len(train_idx))]

        uncertainty = torch.load(os.path.join(args.fold, "uncertainty.pt"))
        u = .0
        for k, v in uncertainty.items():
            u += v
        u = u / len(uncertainty)
        u = u.cpu().numpy().tolist()

        d = torch.load(os.path.join(args.fold, "density.pt"))
        d = d[torch.arange(len(train_idx))].numpy().tolist()

        export_rank(pr, "centrality", train_idx)
        export_rank(u, "uncertainty", train_idx)
        export_rank(d, "density", train_idx)

        pr = val2pct(pr)
        u = val2pct(u)
        d = val2pct(d)
        metric = args.alpha * u + args.beta * pr + (1.0 - args.beta - args.alpha) * d
        export_rank(metric.tolist(), "age", train_idx)
