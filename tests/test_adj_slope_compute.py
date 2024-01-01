import torch
from ogb import nodeproppred
import time
from bifeat.cache import compute_adj_slope

torch.manual_seed(1)

data = nodeproppred.DglNodePropPredDataset(name="ogbn-products", root="/data")
g = data[0][0]
print(g)
indptr, indices, _ = g.adj_tensors("csc")
fan_out = [5, 10, 15]
batch_size = 1000
seeds = torch.arange(0, 196615)

heat = torch.load("/data/presampling-heat/products-5,10,15-adj-heat.pt")
# heat = indptr[1:] - indptr[:-1]

tic = time.time()
slope = compute_adj_slope(indptr, indices, seeds, fan_out, batch_size, heat)
toc = time.time()
print(toc - tic)
