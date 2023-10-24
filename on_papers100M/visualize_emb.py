import argparse

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import torch
import torch_scatter
import torch_geometric.transforms as T

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def main():
    parser = argparse.ArgumentParser(description='OGBN-papers100M (MLP)')
    parser.add_argument('--mode', type=int, default=0)
    parser.add_argument('--pos', type=int, default=-1)
    parser.add_argument('--neg', type=int, default=-1)
    args = parser.parse_args()
    print(args)

    sgc_dict = torch.load('/mnt/ogb_datasets/ogbn_papers100M/sgc_dict.pt')
    split_idx = sgc_dict['split_idx']
    labels = sgc_dict['label'].squeeze(-1)
    num_nodes = labels.size(0)

    # downsample
    if args.mode == 0:
        #emb = torch.load("/mnt/ogb_datasets/ogbn_papers100M/latent_feats/42rep.pt")
        emb = sgc_dict['sgc_embedding'][3]
        sample_idx = torch.randperm(num_nodes)[:num_nodes//5]
        X_tsne = TSNE(metric="cosine", random_state=61998).fit_transform(emb[sample_idx].cpu().numpy())
        #torch.save({'emb': torch.from_numpy(X_tsne), 'index': sample_idx}, "/mnt/ogb_datasets/ogbn_papers100M/latent_feats/42tsne.pt")
        torch.save({'emb': torch.from_numpy(X_tsne), 'index': sample_idx}, "/mnt/ogb_datasets/ogbn_papers100M/latent_feats/rw_tsne.pt")
        return

    #content = torch.load("/mnt/ogb_datasets/ogbn_papers100M/latent_feats/42tsne.pt")
    content = torch.load("/mnt/ogb_datasets/ogbn_papers100M/latent_feats/rw_tsne.pt")
    X_tsne = content['emb'].cpu().numpy()
    sample_idx = content['index']
    sample_mask = torch.zeros((num_nodes,), dtype=torch.bool)
    sample_mask[sample_idx] = True

    count = torch.zeros(172, dtype=torch.long)
    torch_scatter.scatter(torch.ones_like(labels), labels, out=count)
    count = [(i, val) for i, val in enumerate(count.numpy().tolist())]
    count = sorted(count, key=lambda x:x[1], reverse=True)
    print(count)

    if args.pos != -1 and args.neg != -1:
        pos, neg = args.pos, args.neg
    else:
        pos, neg = count[0][0], count[1][0]

    pos_mask = labels == pos
    neg_mask = labels == neg
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[split_idx['train']] = True
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[split_idx['test']] = True

    pos_train = torch.logical_and(torch.logical_and(pos_mask, train_mask), sample_mask)
    neg_train = torch.logical_and(torch.logical_and(neg_mask, train_mask), sample_mask)
    pos_test = torch.logical_and(torch.logical_and(pos_mask, test_mask), sample_mask)
    neg_test = torch.logical_and(torch.logical_and(neg_mask, test_mask), sample_mask)

    pos_train_idx = pos_train.nonzero().squeeze(1)
    neg_train_idx = neg_train.nonzero().squeeze(1)
    pos_test_idx = pos_test.nonzero().squeeze(1)
    neg_test_idx = neg_test.nonzero().squeeze(1)

    pos_train_idx = pos_train_idx[torch.randperm(len(pos_train_idx))[:max(500, len(pos_train_idx))]]
    neg_train_idx = neg_train_idx[torch.randperm(len(neg_train_idx))[:max(500, len(neg_train_idx))]]
    pos_test_idx = pos_test_idx[torch.randperm(len(pos_test_idx))[:max(50, len(pos_test_idx))]]
    neg_test_idx = neg_test_idx[torch.randperm(len(neg_test_idx))[:max(50, len(neg_test_idx))]]
    idx = torch.cat([pos_train_idx, neg_train_idx, pos_test_idx, neg_test_idx])

    labels = np.asarray(len(pos_train_idx)*["class{} train".format(pos)] + len(neg_train_idx)*["class{} train".format(neg)] + len(pos_test_idx)*["class{} test".format(pos)] + len(neg_test_idx)*["class{} test".format(neg)])
    sample_idx = dict([(idx, i) for i, idx in enumerate(sample_idx.cpu().numpy().tolist())])
    idx = [sample_idx[i] for i in torch.cat([pos_train_idx, neg_train_idx, pos_test_idx, neg_test_idx]).cpu().numpy().tolist()]
    X_tsne = X_tsne[np.asarray(idx),:]

    df_all = pd.DataFrame(X_tsne.T[0])
    df_all['t-SNE dim1'] = X_tsne.T[0]
    df_all['t-SNE dim2'] = X_tsne.T[1]
    df_all['label'] = labels

    cus_palette = sns.color_palette("colorblind", as_cmap=True)
    # sns.set(style='whitegrid',font_scale=4)
    # plt.figure(figsize=(15, 15), dpi=200)
    plt.figure(figsize=(6, 6))
    ax = sns.scatterplot(
        data=df_all, 
        x="t-SNE dim1",
        y="t-SNE dim2",
        hue="label",
        palette=cus_palette,
        # palette=sns.color_palette("deep", 1),
        # color='#6FA8DC',
        s=50,
        alpha=0.75,
        legend=True
    )
    # ax.set_xlabel("t-SNE dimension 1", fontsize=14)
    # ax.set_ylabel("t-SNE dimension 2", fontsize=14)
    ax.set_xticklabels([], fontsize=24)
    ax.set_yticklabels([], fontsize=24)
    # plt.title('')
    # sns.despine()
    # plt.xlabel()
    # plt.ylabel('t-SNE dimension 2', fontsize=13)
    plt.savefig("{}-{}.pdf".format(pos, neg), transparent='True')


if __name__=="__main__":
    main()
