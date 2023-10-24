import os
import argparse

import numpy as np
from pytorch_lightning import (LightningDataModule, LightningModule, Trainer,
                               seed_everything)

from ogb.lsc import MAG240MDataset, MAG240MEvaluator

from root import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=str, default='random')
    args = parser.parse_args()
    print(args)

    #seed_everything(123)
    seed_everything(42)
    dataset = MAG240MDataset(ROOT)
    train_idx = dataset.get_idx_split('train')
    print(train_idx)
    print(train_idx.shape)
    np.random.shuffle(train_idx)
    print(train_idx)
    with open(os.path.join(args.fold, 'train.tsv'), 'w') as ops:
        train_idx = train_idx.tolist()
        num_tr_nodes = len(train_idx)
        for i in range(num_tr_nodes):
            ops.write("{}\t{}\n".format(train_idx[i], float(num_tr_nodes - i) / num_tr_nodes))


if __name__=="__main__":
    main()
