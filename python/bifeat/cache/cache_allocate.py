import torch
from ..shm import dtype_sizeof
from .adj_cache import StructureCacheServer
from ..dataloading import SeedGenerator
import time
import numpy as np


def compute_adj_slope(indptr,
                      indices,
                      seeds,
                      fan_out,
                      batch_size,
                      step=0.1,
                      num_epochs=5):
    seeds_loader = SeedGenerator(seeds, batch_size, shuffle=True)
    cache_rate = 0
    num_nodes = indptr.shape[0] - 1
    nids = torch.arange(num_nodes).long()

    # warmup
    cached_nids = torch.tensor([])
    sampler = StructureCacheServer(indptr, indices, fan_out)
    sampler.cache_data(cached_nids)
    for it, seeds in enumerate(seeds_loader):
        _ = sampler.sample_neighbors(seeds)

    stats = []
    for i in range(num_epochs):
        sampler.clear_cache()
        torch.cuda.empty_cache()
        cache_num = int(num_nodes * cache_rate)
        cached_nids = nids[:cache_num]
        try:
            sampler.cache_data(cached_nids)
            torch.cuda.synchronize()
            tic = time.time()
            for it, seeds in enumerate(seeds_loader):
                _ = sampler.sample_neighbors(seeds, count_hit=True)
            torch.cuda.synchronize()
            toc = time.time()
            epoch_time = toc - tic
        except:
            break
        cache_rate += step
        print(
            "Cache rate: {:.3f}, cached num: {}, hit times: {}, epoch time(s): {:.3f}"
            .format(cache_rate, cache_num,
                    sampler.get_hit_rates()[1], epoch_time))
        stats.append((sampler.get_hit_rates()[1], epoch_time * 1000000))
    sampler.clear_cache()
    torch.cuda.empty_cache()
    stats = np.array(stats)
    return -np.polyfit(stats[:, 0], stats[:, 1], 1)[0]


def compute_feat_slope():
    pass


def compute_feat_sapce(feat_dim, feat_dtype):
    return feat_dim * dtype_sizeof(feat_dtype)


def compute_adj_space_tensor(indptr, indptr_dtype, indices_dtype):
    degree = indptr[1:] - indptr[:-1]
    return degree * dtype_sizeof(indices_dtype) + dtype_sizeof(indptr_dtype)


def cache_idx_select(
    hotness_core_nodes,
    hotness_other_nodes,
    hotness_adj,
    core_feat_slope,
    other_feat_slope,
    adj_slope,
    core_feat_space,
    other_feat_space,
    adj_space_tensor,
    gpu_capacity,
):

    core_num = hotness_core_nodes.shape[0]
    other_num = hotness_other_nodes.shape[0]
    adj_num = core_num + other_num

    unified_core_hotness = hotness_core_nodes * core_feat_slope / core_feat_space
    unified_other_hotness = hotness_other_nodes * other_feat_slope / other_feat_space
    unified_adj_hotness = hotness_adj * adj_slope / adj_space_tensor

    index_threshold_other = core_num
    index_threshold_adj = adj_num

    unified_hotness = torch.cat(
        [unified_core_hotness, unified_other_hotness, unified_adj_hotness])
    unified_space = torch.cat([
        torch.full((core_num, ), core_feat_space),
        torch.full((other_num, ), other_feat_space),
        adj_space_tensor,
    ])

    sorted_index = torch.argsort(unified_hotness, descending=True)
    sorted_space = unified_space[sorted_index]
    space_prefix_sum = torch.cumsum(sorted_space)
    cached_index = sorted_index[space_prefix_sum <= gpu_capacity]
    core_cached_index = cached_index[cached_index < index_threshold_other]
    other_cached_index = cached_index[(cached_index < index_threshold_adj) &
                                      (cached_index >= index_threshold_other)]
    adj_cached_index = cached_index[cached_index >= index_threshold_adj]

    core_cached_idx = core_cached_index
    other_cached_idx = other_cached_index - index_threshold_other
    adj_cached_nids = adj_cached_index - index_threshold_adj

    return core_cached_idx, other_cached_idx, adj_cached_nids
