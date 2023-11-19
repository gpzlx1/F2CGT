import torch
import time
from pagraph import FeatureCacheServer

if __name__ == "__main__":
    cpu_data = torch.arange(0, 10000).reshape(100, 100).float()
    index = torch.randint(0, 100, (100, )).long().cuda()

    # part cache
    cache_nids = torch.randint(0, 100, (20, )).int().cuda()
    part_cache_server = FeatureCacheServer(cpu_data)
    part_cache_server.cache_feature(cache_nids, False)
    assert part_cache_server.fetch_data(index).equal(
        cpu_data[index.cpu().long()].cuda())
    del part_cache_server

    # full cache
    full_cache_server = FeatureCacheServer(cpu_data)
    full_cache_server.cache_feature(torch.tensor([]).int().cuda(), True)
    assert full_cache_server.fetch_data(index).equal(
        cpu_data[index.cpu().long()].cuda())
    del full_cache_server

    # no cache
    no_cache_server = FeatureCacheServer(cpu_data)
    no_cache_server.cache_feature(torch.tensor([]).int().cuda(), False)
    assert no_cache_server.fetch_data(index).equal(
        cpu_data[index.cpu().long()].cuda())
    del no_cache_server
