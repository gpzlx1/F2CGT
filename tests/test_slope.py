import torch
from ogb import nodeproppred
import time
from bifeat.cache import compute_adj_slope, compute_feat_slope

torch.manual_seed(1)

data = nodeproppred.DglNodePropPredDataset(name="ogbn-products", root="/data")
g = data[0][0]
print(g)
indptr, indices, _ = g.adj_tensors("csc")
fan_out = [5, 10, 15]
batch_size = 1000
seeds = torch.arange(0, 196615)

tic = time.time()
slope, input_num = compute_adj_slope(indptr, indices, seeds, fan_out,
                                     batch_size, indptr[1:] - indptr[:-1])
toc = time.time()
print(toc - tic)

features = torch.randn((
    g.num_nodes(),
    100,
))
tic = time.time()
slope = compute_feat_slope(features, indptr[1:] - indptr[:-1], input_num)
toc = time.time()
print(toc - tic)
