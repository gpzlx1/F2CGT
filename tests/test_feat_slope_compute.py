import torch
from ogb import nodeproppred
import time
from bifeat.cache import compute_feat_slope

torch.manual_seed(1)

data = nodeproppred.DglNodePropPredDataset(name="ogbn-products", root="/data")
g = data[0][0]
print(g)
indptr, indices, _ = g.adj_tensors("csc")
fan_out = [5, 10, 15]
batch_size = 1000
seeds = torch.arange(0, 196615)
feature = torch.randn((g.num_nodes(), 100))
# heat = indptr[1:] - indptr[:-1]
heat = torch.load("/data/presampling-heat/products-5,10,15-heat.pt")

tic = time.time()
slope = compute_feat_slope(feature, heat, indptr, indices, seeds, fan_out,
                           batch_size)
toc = time.time()
print(toc - tic)
