import torch
from ..shm import dtype_sizeof


def compute_adj_slope():
    pass


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
