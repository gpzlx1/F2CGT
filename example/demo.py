import torch
import time
from pagraph import FeatureCache

if __name__ == "__main__":
    cpu_data = torch.arange(0, 10000).reshape(100, 100).float()
    index = torch.randint(0, 100, (100, )).long().cuda()

    # part cache
    gpu_capacity = 20 * 4 * 100
    part_cache_server = FeatureCache(cpu_data)
    part_cache_server.create_cache(gpu_capacity)
    assert part_cache_server[index].equal(cpu_data[index.cpu().long()].cuda())
    del part_cache_server

    # full cache
    gpu_capacity = 100 * 4 * 100
    full_cache_server = FeatureCache(cpu_data)
    full_cache_server.create_cache(gpu_capacity)
    assert full_cache_server[index].equal(cpu_data[index.cpu().long()].cuda())
    del full_cache_server

    # no cache
    gpu_capacity = 0
    no_cache_server = FeatureCache(cpu_data)
    no_cache_server.create_cache(gpu_capacity)
    assert no_cache_server[index].equal(cpu_data[index.cpu().long()].cuda())
    del no_cache_server
