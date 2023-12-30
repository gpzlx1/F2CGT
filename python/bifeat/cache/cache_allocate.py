import torch
from ..shm import dtype_sizeof
from .adj_cache import StructureCacheServer
from .feature_cache import FeatureCacheServer
from ..dataloading import SeedGenerator
import time
import numpy as np


def compute_adj_slope(indptr,
                      indices,
                      seeds,
                      fan_out,
                      batch_size,
                      hotness,
                      step=0.05,
                      num_epochs=10):
    seeds_loader = SeedGenerator(seeds, batch_size, shuffle=True)
    cache_rate = 0
    num_nodes = hotness.shape[0]
    nids = torch.argsort(hotness, descending=True).long()

    # warmup
    cached_nids = torch.tensor([])
    sampler = StructureCacheServer(indptr, indices, fan_out, count_hit=True)
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
                _ = sampler.sample_neighbors(seeds)
            torch.cuda.synchronize()
            toc = time.time()
            epoch_time = toc - tic
        except:
            break
        print(
            "Cache rate: {:.3f}, cached num: {}, hit times: {}, epoch time(s): {:.3f}"
            .format(cache_rate, cache_num,
                    sampler.get_hit_rates()[1], epoch_time))
        stats.append((sampler.get_hit_rates()[1], epoch_time * 1000000))
        cache_rate += step
    sampler.clear_cache()
    torch.cuda.empty_cache()
    stats = np.array(stats)
    slope = -np.polyfit(stats[:, 0], stats[:, 1], 1)[0]

    print("Adj slope = {:.5f}".format(slope))

    return slope


def compute_feat_slope(features,
                       hotness,
                       indptr,
                       indices,
                       seeds,
                       fan_out,
                       batch_size,
                       step=0.2,
                       num_epochs=5):
    idx = torch.argsort(hotness, descending=True).long()
    cache_rate = 0
    num_nodes = hotness.shape[0]

    seeds_loader = SeedGenerator(seeds, batch_size, shuffle=True)
    sampler = StructureCacheServer(indptr, indices, fan_out, count_hit=True)
    sampler.cache_data(torch.tensor([]))

    # warmup
    cached_nids = torch.tensor([])
    feature_cache = FeatureCacheServer(features, count_hit=True)
    feature_cache.cache_data(cached_nids)
    for it, seeds in enumerate(seeds_loader):
        input_nodes, _, _ = sampler.sample_neighbors(seeds)
        _ = feature_cache[input_nodes]

    stats = []
    for i in range(num_epochs):
        feature_cache.clear_cache()
        torch.cuda.empty_cache()
        cache_num = int(num_nodes * cache_rate)
        cached_nids = idx[:cache_num]
        try:
            feature_cache.cache_data(cached_nids)
            epoch_time = 0
            for it, seeds in enumerate(seeds_loader):
                input_nodes, _, _ = sampler.sample_neighbors(seeds)
                torch.cuda.synchronize()
                tic = time.time()
                _ = feature_cache[input_nodes]
                torch.cuda.synchronize()
                toc = time.time()
                epoch_time += toc - tic
        except:
            break
        print(
            "Cache rate: {:.3f}, cached num: {}, hit times: {}, epoch time(s): {:.3f}"
            .format(cache_rate, cache_num,
                    feature_cache.get_hit_rates()[1], epoch_time))
        stats.append((feature_cache.get_hit_rates()[1], epoch_time * 1000000))
        cache_rate += step
    feature_cache.clear_cache()
    torch.cuda.empty_cache()
    stats = np.array(stats)
    slope = -np.polyfit(stats[:, 0], stats[:, 1], 1)[0] / features.shape[1]

    print("Feature slope = {:.5f}".format(slope))
    return slope


def compute_feat_sapce(feat_dim, feat_dtype):
    space = feat_dim * dtype_sizeof(feat_dtype)
    # with hashmap size
    space += 4 * (dtype_sizeof(torch.int64) + dtype_sizeof(torch.int32))
    return space


def compute_adj_space_tensor(indptr, indptr_dtype, indices_dtype):
    degree = indptr[1:] - indptr[:-1]
    space = degree * dtype_sizeof(indices_dtype) + dtype_sizeof(indptr_dtype)
    # with hashmap size
    space += 4 * (dtype_sizeof(torch.int64) + dtype_sizeof(torch.int32))
    return space


def cache_idx_select(
    hotness_feat_list,
    hotness_adj,
    feat_slope_list,
    adj_slope,
    feat_space_list,
    adj_space_tensor,
    gpu_capacity,
):
    num_feat_type = len(hotness_feat_list)
    unified_hotness_list = []
    unified_space_list = []
    type_range = [0]
    for i in range(num_feat_type):
        num_idx = hotness_feat_list[i].shape[0]
        unified_hotness_list.append(hotness_feat_list[i] * feat_slope_list[i] /
                                    feat_space_list[i])
        unified_space_list.append(torch.full((num_idx, ), feat_space_list[i]))
        range_max = num_idx + type_range[i]
        type_range.append(range_max)
    unified_hotness_list.append(hotness_adj * adj_slope / adj_space_tensor)
    unified_space_list.append(adj_space_tensor)

    unified_hotness = torch.cat(unified_hotness_list)
    unified_space = torch.cat(unified_space_list)
    # valid_mask = unified_hotness > 0
    sorted_index = torch.argsort(unified_hotness, descending=True)
    del unified_hotness
    # sorted_index = sorted_index[valid_mask[sorted_index]]
    # del valid_mask
    sorted_space = unified_space[sorted_index]
    del unified_space
    space_prefix_sum = torch.cumsum(sorted_space, 0)
    del sorted_space
    cached_index = sorted_index[space_prefix_sum <= gpu_capacity]
    del space_prefix_sum

    cached_index_list = []
    for i in range(num_feat_type):
        this_type_cached_index = cached_index[
            (cached_index >= type_range[i])
            & (cached_index < type_range[i + 1])] - type_range[i]
        cached_index_list.append(this_type_cached_index)
    adj_cached_index = cached_index[cached_index >=
                                    type_range[-1]] - type_range[-1]

    return cached_index_list, adj_cached_index
