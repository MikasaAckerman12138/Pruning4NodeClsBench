import torch

path = '/mnt/ogb_datasets/mag240m_kddcup2021/paper_to_paper_symmetric.pt'
adj_t = torch.load(path)
print(adj_t)
