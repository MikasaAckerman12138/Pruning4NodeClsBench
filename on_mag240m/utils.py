import os
import math
import random

import numpy as np
import torch


def load_rank_list(al):
    ranks = []
    with open(os.path.join(al, "train.tsv"), 'r') as ips:
        for line in ips:
            idx, _ = line.strip().split('\t')
            idx = int(idx)
            ranks.append(idx)
    return ranks

def aug_rank_by_pct(rank):
    return [(idx, float(len(rank)-i)/len(rank)) for i, idx in enumerate(rank)]

def random_splits(num_nodes, num_classes, y, idx, rsv):
    # * round(reserve_rate*len(data)/num_classes) * num_classes labels for training

    assert rsv >= .0 and rsv <= 1.0, "Invalid value {} for reserve_rate".format(rsv)

    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[idx] = True

    indices = []
    new_lens = []
    for i in range(num_classes):
        index = torch.logical_and((y == i), mask).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

        expected_l = rsv * len(index)
        l = math.floor(expected_l)
        l = l + (1 if (expected_l - l) >= random.uniform(0, 1) else 0)
        new_lens.append(l)

    index = torch.cat([arr[:new_lens[i]] for i, arr in enumerate(indices)], dim=0)

    #if Flag is 0:
    #    rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
    #    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    #    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    #    data.val_mask = index_to_mask(rest_index[:val_lb], size=data.num_nodes)
    #    data.test_mask = index_to_mask(
    #    rest_index[val_lb:], size=data.num_nodes)
    #else:
    #    val_index = torch.cat([i[percls_trn:percls_trn+val_lb]
    #                           for i in indices], dim=0)
    #    rest_index = torch.cat([i[percls_trn+val_lb:] for i in indices], dim=0)
    #    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    #    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    #    data.val_mask = index_to_mask(val_index, size=data.num_nodes)
    #    data.test_mask = index_to_mask(rest_index, size=data.num_nodes)
    #return data

    return index


def select_by_al(num_nodes, num_classes, y, idx, rsv, ranks):
    # * round(reserve_rate*len(data)/num_classes) * num_classes labels for training

    assert rsv >= .0 and rsv <= 1.0, "Invalid value {} for reserve_rate".format(rsv)

    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[idx] = True

    indices = []
    new_lens = []
    for i in range(num_classes):
        index = torch.logical_and((y == i), mask).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

        expected_l = 0.5 * rsv * len(index)
        l = math.floor(expected_l)
        l = l + (1 if (expected_l - l) >= random.uniform(0, 1) else 0)
        new_lens.append(l)

    index = torch.cat([arr[:new_lens[i]] for i, arr in enumerate(indices)], dim=0)

    num_to_select = round(rsv * len(idx)) - np.sum(new_lens)
    existing = set(index.cpu().numpy().tolist())
    to_select = []
    for i in ranks:
        if i not in existing:
            to_select.append(i)
            num_to_select -= 1
            if num_to_select <= 0:
                break
    to_select = np.asarray(to_select, dtype=np.int64)
    index = torch.cat([index, torch.from_numpy(to_select)], dim=0)

    return index
