import torch
import time
import dgl
from bifeat.cache import StructureCacheServer
from bifeat.dataloading import SeedGenerator
from ogb import nodeproppred

torch.manual_seed(1)

data = nodeproppred.DglNodePropPredDataset(name="ogbn-products", root="/data")
g = data[0][0]
train_idx = torch.arange(196615).long().cuda()
indptr, indices, _ = g.adj_tensors("csc")
sampler = StructureCacheServer(indptr, indices, [5, 10, 15])
dataloader = SeedGenerator(train_idx.cuda(), 1000)

# presampling_epoch = 1
# sampler.cache_data(torch.tensor([]))
# heat = torch.zeros((g.num_nodes(), ), dtype=torch.float32)
# for epoch in range(presampling_epoch):
#     for it, seeds in enumerate(dataloader):
#         _, _, blocks = sampler.sample_neighbors(seeds)
#         for block in blocks:
#             layer_seeds = block.ndata[dgl.NID]["_N"][block.dstnodes()].cpu()
#             heat[layer_seeds] += 1
# torch.save(heat, "/data/presampling-heat/products-5,10,15-adj-heat.pt")
# sampler.clear_cache()
# torch.cuda.empty_cache()

heat = torch.load("/data/presampling-heat/products-5,10,15-adj-heat.pt")

sorted_idx = torch.argsort(heat, descending=True)
sorted_heat = heat[sorted_idx]
slope = 0.03132

length = g.num_nodes()

# no cache
cache_index = torch.tensor([])
sampler.cache_data(cache_index)
# warm up
for it, seeds in enumerate(dataloader):
    _ = sampler.sample_neighbors(seeds)
torch.cuda.synchronize()
tic = time.time()
for it, seeds in enumerate(dataloader):
    _ = sampler.sample_neighbors(seeds)
torch.cuda.synchronize()
epoch_time0 = time.time() - tic

for cache_rate in [0.25, 0.5, 0.75, 1]:
    print("cache rate {:.2f}".format(cache_rate))

    # with cache
    cache_num = int(cache_rate * length)
    cache_index = sorted_idx[:cache_num]
    torch.cuda.empty_cache()
    sampler.clear_cache()
    sampler.cache_data(cache_index)
    # warm up
    for it, seeds in enumerate(dataloader):
        _ = sampler.sample_neighbors(seeds)
    torch.cuda.synchronize()
    tic = time.time()
    for it, seeds in enumerate(dataloader):
        _ = sampler.sample_neighbors(seeds)
    torch.cuda.synchronize()
    epoch_time1 = time.time() - tic

    hit_times = torch.sum(sorted_heat[:cache_num]).item()

    print(hit_times)
    print(epoch_time0 - epoch_time1)
    print(slope * hit_times / 1000000)

del sampler
