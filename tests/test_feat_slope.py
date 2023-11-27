import torch
import time
from bifeat.cache import FeatureCacheServer, StructureCacheServer
from bifeat.dataloading import SeedGenerator
from ogb import nodeproppred
import numpy as np

torch.manual_seed(1)

data = nodeproppred.DglNodePropPredDataset(name="ogbn-products", root="/data")
g = data[0][0]
train_idx = torch.arange(196615).long().cuda()
indptr, indices, _ = g.adj_tensors("csc")
sampler = StructureCacheServer(indptr, indices, [5, 10, 15])
sampler.cache_data(torch.tensor([]))
dataloader = SeedGenerator(train_idx.cuda(), 1000)

# presampling_epoch = 1
# heat = torch.zeros((g.num_nodes(), ), dtype=torch.float32)
# for epoch in range(presampling_epoch):
#     for it, seeds in enumerate(dataloader):
#         input_nodes, _, _ = sampler.sample_neighbors(seeds)
#         heat[input_nodes.cpu()] += 1
# torch.save(heat, "/data/presampling-heat/products-5,10,15-heat.pt")
heat = torch.load("/data/presampling-heat/products-5,10,15-heat.pt")

sorted_idx = torch.argsort(heat, descending=True)
sorted_heat = heat[sorted_idx]
slope = 0.00016

for dim in [32, 64, 128, 256]:
    length = g.num_nodes()
    feat = torch.randn((length, dim)).float()

    feature_cache = FeatureCacheServer(feat)

    # no cache
    cache_index = torch.tensor([])
    feature_cache.cache_data(cache_index)
    # warm up
    for it, seeds in enumerate(dataloader):
        input_nodes, _, _ = sampler.sample_neighbors(seeds)
        buff = feature_cache[input_nodes]
    epoch_time0 = 0
    for it, seeds in enumerate(dataloader):
        input_nodes, _, _ = sampler.sample_neighbors(seeds)
        torch.cuda.synchronize()
        tic = time.time()
        buff = feature_cache[input_nodes]
        torch.cuda.synchronize()
        epoch_time0 += time.time() - tic

    for cache_rate in [0.25, 0.5, 0.75, 1]:
        print("dim {}, cache rate {:.2f}".format(dim, cache_rate))

        # with cache
        cache_num = int(cache_rate * length)
        cache_index = sorted_idx[:cache_num]
        torch.cuda.empty_cache()
        feature_cache.clear_cache()
        feature_cache.cache_data(cache_index)
        # warm up
        for it, seeds in enumerate(dataloader):
            input_nodes, _, _ = sampler.sample_neighbors(seeds)
            buff = feature_cache[input_nodes]
        epoch_time1 = 0
        for it, seeds in enumerate(dataloader):
            input_nodes, _, _ = sampler.sample_neighbors(seeds)
            torch.cuda.synchronize()
            tic = time.time()
            buff = feature_cache[input_nodes]
            torch.cuda.synchronize()
            epoch_time1 += time.time() - tic
        hit_times = torch.sum(sorted_heat[:cache_num]).item()

        print(hit_times)
        print(epoch_time0 - epoch_time1)
        print(slope * hit_times * dim / 1000000)

    del feature_cache
del sampler
