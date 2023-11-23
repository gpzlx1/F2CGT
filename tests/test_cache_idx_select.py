from bifeat.cache import cache_idx_select
import torch

hotness_feat_list = [
    torch.randint(0, 100, (10, )),
    torch.randint(0, 200, (5, )),
    torch.randint(0, 50, (20, ))
]
hotness_adj = torch.randint(0, 150, (35, ))
feat_slope_list = [1, 1.5, 0.75]
adj_slope = 0.9
feat_space_list = [8, 64, 32]
adj_space = torch.randint(1, 65, (35, ))
gpu_capacity = 1024

feat_idx_list, adj_idx = cache_idx_select(hotness_feat_list, hotness_adj,
                                          feat_slope_list, adj_slope,
                                          feat_space_list, adj_space,
                                          gpu_capacity)
print(feat_idx_list)
print(adj_idx)
