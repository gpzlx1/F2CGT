import torch
import dgl
from ogb import nodeproppred
import time
import numpy as np
from bifeat import StructureCacheServer

torch.manual_seed(1)

data = nodeproppred.DglNodePropPredDataset(name="ogbn-products", root="/data")
g = data[0][0]
print(g)

indptr, indices, _ = g.adj_tensors('csc')
print(indptr)
print(indices)

seeds = torch.arange(196615).long()
batch_size = 1000

# part cache
part_server = StructureCacheServer(indptr, indices)
train_nids = torch.arange(g.num_nodes() // 5).long()
part_server.cache_data(train_nids)
time_list = []
for i in range(10):
    seed_nids = seeds[i * batch_size:(i + 1) * batch_size]
    begin = time.time()
    frontier, seed_nids, blocks = part_server.sample_neighbors(
        seed_nids, [15, 15, 15], False)
    torch.cuda.synchronize()
    end = time.time()
    time_list.append(end - begin)
    print(blocks)
print(np.mean(time_list[2:]) * 1000)
del part_server

# part cache
full_server = StructureCacheServer(indptr, indices)
train_nids = torch.arange(g.num_nodes()).long()
full_server.cache_data(train_nids)
time_list = []
for i in range(10):
    seed_nids = seeds[i * batch_size:(i + 1) * batch_size]
    begin = time.time()
    frontier, seed_nids, blocks = full_server.sample_neighbors(
        seed_nids, [15, 15, 15], False)
    torch.cuda.synchronize()
    end = time.time()
    time_list.append(end - begin)
    print(blocks)
print(np.mean(time_list[2:]) * 1000)
del full_server

# no cache
empty_server = StructureCacheServer(indptr, indices)
train_nids = torch.tensor([]).long()
empty_server.cache_data(train_nids)
time_list = []
for i in range(10):
    seed_nids = seeds[i * batch_size:(i + 1) * batch_size]
    begin = time.time()
    frontier, seed_nids, blocks = empty_server.sample_neighbors(
        seed_nids, [15, 15, 15], False)
    torch.cuda.synchronize()
    end = time.time()
    time_list.append(end - begin)
    print(blocks)
print(np.mean(time_list[2:]) * 1000)
del empty_server
