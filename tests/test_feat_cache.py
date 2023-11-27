import torch
import time
from bifeat.cache import FeatureCacheServer

if __name__ == "__main__":

    # core_data = torch.arange(1000, 2000).reshape(10, 100).float()
    # core_nids = torch.cat([torch.arange(50, 55), torch.arange(95, 100)])
    # other_data = torch.arange(0, 1000).reshape(100, 10).float()
    # other_nids = torch.cat(
    #     [torch.arange(0, 50),
    #      torch.arange(55, 95),
    #      torch.arange(100, 110)])
    # idx_map = torch.zeros((110, )).long()
    # idx_map[core_nids] = torch.arange(0, 10)
    # idx_map[other_nids] = torch.arange(0, 100)
    # core_mask = torch.zeros((110, ), dtype=torch.bool)
    # core_mask[core_nids] = True

    # fetch_nids = torch.randperm(110)[:30]

    # for _ in range(5):
    #     tic = time.time()
    #     mask_ = core_mask[fetch_nids]
    #     core_ = core_data[idx_map[fetch_nids[mask_]]].cuda()
    #     other_ = other_data[idx_map[fetch_nids[~mask_]]].cuda()
    #     torch.cuda.synchronize()
    #     toc = time.time()
    #     print(toc - tic)

    # # part cache
    # core_cache_idx = torch.randperm(10)[:4]
    # other_cache_idx = torch.randperm(100)[:40]
    # part_cache_server = FeatureCacheServer(core_data, core_nids, other_data,
    #                                        other_nids)
    # part_cache_server.cache_data(core_cache_idx, other_cache_idx)
    # for _ in range(5):
    #     tic = time.time()
    #     core, other, mask = part_cache_server[fetch_nids.cuda()]
    #     torch.cuda.synchronize()
    #     toc = time.time()
    #     print(toc - tic)
    # assert core.equal(core_)
    # assert other.equal(other_)
    # assert mask.equal(mask_.cuda())
    # del part_cache_server

    # # full cache
    # core_cache_idx = torch.randperm(10)[:10]
    # other_cache_idx = torch.randperm(100)[:100]
    # full_cache_server = FeatureCacheServer(core_data, core_nids, other_data,
    #                                        other_nids)
    # full_cache_server.cache_data(core_cache_idx, other_cache_idx)
    # for _ in range(5):
    #     tic = time.time()
    #     core, other, mask = full_cache_server[fetch_nids.cuda()]
    #     torch.cuda.synchronize()
    #     toc = time.time()
    #     print(toc - tic)
    # assert core.equal(core_)
    # assert other.equal(other_)
    # assert mask.equal(mask_.cuda())
    # del full_cache_server

    # # empty cache
    # core_cache_idx = torch.tensor([])
    # other_cache_idx = torch.tensor([])
    # empty_cache_server = FeatureCacheServer(core_data, core_nids, other_data,
    #                                         other_nids)
    # empty_cache_server.cache_data(core_cache_idx, other_cache_idx)
    # for _ in range(5):
    #     tic = time.time()
    #     core, other, mask = empty_cache_server[fetch_nids.cuda()]
    #     torch.cuda.synchronize()
    #     toc = time.time()
    #     print(toc - tic)
    # assert core.equal(core_)
    # assert other.equal(other_)
    # assert mask.equal(mask_.cuda())
    # del empty_cache_server

    cpu_data = torch.arange(0, 10000).reshape(100, 100).float()
    index = torch.randint(0, 100, (100, )).long().cuda()

    # part cache
    cache_nids = torch.randint(0, 100, (20, )).unique().int()
    part_cache_server = FeatureCacheServer(cpu_data)
    part_cache_server.cache_data(cache_nids)
    assert part_cache_server[index].equal(cpu_data[index.cpu().long()].cuda())
    del part_cache_server

    # full cache
    full_cache_server = FeatureCacheServer(cpu_data)
    full_cache_server.cache_data(torch.arange(0, 100).int())
    assert full_cache_server[index].equal(cpu_data[index.cpu().long()].cuda())
    del full_cache_server

    # no cache
    no_cache_server = FeatureCacheServer(cpu_data)
    no_cache_server.cache_data(torch.tensor([]).int())
    assert no_cache_server[index].equal(cpu_data[index.cpu().long()].cuda())
    del no_cache_server
