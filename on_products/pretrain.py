import argparse
import os.path as osp
import time

from tqdm.auto import tqdm

import torch
import torch_geometric as pyg
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import GAE, VGAE, SAGEConv
from torch_geometric.utils import negative_sampling

from ogb.nodeproppred import PygNodePropPredDataset

from utils import *


class PositiveLinkNeighborSampler(pyg.loader.NeighborSampler):
    def __init__(self, edge_index, sizes, num_nodes=None, **kwargs):
        edge_idx = torch.arange(edge_index.size(1))
        super(PositiveLinkNeighborSampler,
              self).__init__(edge_index, sizes, edge_idx, num_nodes, **kwargs)

    def sample(self, edge_idx):
        if not isinstance(edge_idx, torch.Tensor):
            edge_idx = torch.tensor(edge_idx)
        row, col, _ = self.adj_t.coo()
        batch = torch.cat([row[edge_idx], col[edge_idx]], dim=0)
        return super(PositiveLinkNeighborSampler, self).sample(batch)


class VariationalSAGEEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=256, num_layers=3,
                 dropout=0.25):
        super(VariationalSAGEEncoder, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.conv_logstd = SAGEConv(hidden_channels, out_channels)

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.conv_logstd.reset_parameters()

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            if i != len(self.convs)-1:
                x = self.convs[i]((x, x_target), edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            else:
                mu, logstd = self.convs[i]((x, x_target), edge_index), self.conv_logstd((x, x_target), edge_index)

        return mu, logstd

    def inference(self, x_all, subgraph_loader, device):
        pbar = tqdm(total=x_all.size(0) * len(self.convs))
        pbar.set_description('Evaluating')

        for i, conv in enumerate(self.convs):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = conv((x, x_target), edge_index)
                if i != len(self.convs) - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all


def train(data, pos_loader, model, optimizer, device):
    pbar = tqdm(total=len(pos_loader))
    pbar.set_description('Training')

    model.train()
    total_loss, total_kl_loss = 0, 0

    for i, pos_data in enumerate(pos_loader):
        optimizer.zero_grad()
        batch_size, n_id, adjs = pos_data
        x = data.x[n_id].to(device)
        adjs = [adj.to(device) for adj in adjs]
        z = model.encode(x, adjs)
        pos_lb = torch.stack([torch.arange(start=0, end=z.size(0)//2, device=device), torch.arange(start=z.size(0)//2, end=z.size(0), device=device)])
        loss = model.recon_loss(z, pos_lb, negative_sampling(pos_lb, z.size(0), 4*pos_lb.size(1)))
        total_loss += loss.item()
        kl_loss = model.kl_loss()
        total_kl_loss += kl_loss.item()
        loss = loss + kl_loss
        loss.backward()
        optimizer.step()
        pbar.update(1)

    pbar.close()

    return total_loss / len(pos_loader), total_kl_loss / len(pos_loader)


@torch.no_grad()
def test(data, model, subgraph_loader, pos_lbs, neg_lbs, device):
    model.eval()
    z = model.encoder.inference(data.x, subgraph_loader, device)
    auc, ap = model.test(z, pos_lbs, neg_lbs)
    return z, auc, ap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=20000)
    parser.add_argument('--latent_dim', type=int, default=16)
    parser.add_argument('--out_fn', type=str, default='emb.pt')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--sizes', type=str, default='12,8,4')
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    setup_seed(args.seed)

    device = torch.device('cuda:{}'.format(args.device))
    
    dataset = PygNodePropPredDataset(name='ogbn-products')
    data = dataset[0]

    sizes = [int(val) for val in args.sizes.split(',')]
    edge_reindex = torch.randperm(data.edge_index.size(1))
    pos_loader = PositiveLinkNeighborSampler(data.edge_index[:,edge_reindex[:8*(len(edge_reindex)//10)]], sizes=sizes,
                                             num_nodes=data.x.size(0),
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=12)
    subgraph_loader = NeighborSampler(data.edge_index, sizes=[-1],
                                      batch_size=4096, shuffle=False,
                                      num_workers=12)
    pos_test_lbs = data.edge_index[:,edge_reindex[8*(len(edge_reindex)//10):]]
    neg_test_lbs = negative_sampling(data.edge_index[:,edge_reindex[8*(len(edge_reindex)//10):]],
                                     data.x.size(0),
                                     8*(len(edge_reindex)//10))

    in_channels, out_channels = data.x.size(-1), args.latent_dim
    
    model = VGAE(VariationalSAGEEncoder(in_channels, out_channels, num_layers=len(sizes), dropout=args.dropout))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs // 5, gamma=0.5)

    best_auc = .0
    for epoch in range(1, args.epochs + 1):
        loss, kl_loss = train(data, pos_loader, model, optimizer, device)
        print(f'Epoch: {epoch:03d}, train_loss: {loss:.4f}, kl_loss: {kl_loss:.4f}')
        z, auc, ap = test(data, model, subgraph_loader, pos_test_lbs, neg_test_lbs, device)
        print(f'Epoch: {epoch:03d}, AUC: {auc:.4f}, AP: {ap:.4f}')
        if auc > best_auc:
            best_auc = auc
            torch.save(z, osp.join('/mnt/ogb_datasets/ogbn_products', args.out_fn))
        scheduler.step()


if __name__=="__main__":
    main()
