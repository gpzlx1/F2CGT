import torch
import time
from bifeat.cache import FeatureCacheServer

torch.manual_seed(1)

for dim in [1, 4, 32, 64, 128, 256]:
    for cache_rate in [0.25, 0.5, 0.75, 1]:
        print("dim {}, cache rate {:.2f}".format(dim, cache_rate))

        length = 10_000_000
        feat = torch.randn((length, dim)).float()
        heat = torch.randint(0, 200, (length, ))
        sorted_idx = torch.argsort(heat, descending=True)

        step = 0.1
        num_step = 10

        num_iters = 30
        fetch_size = 1_000_000

        slope = 0.00012741346924645213

        feature_cache = FeatureCacheServer(feat, count_hit=True)

        # no cache
        cache_index = torch.tensor([])
        feature_cache.cache_data(cache_index)
        epoch_time0 = 0
        for iter in range(num_iters):
            index = torch.randint(0, length, (fetch_size, )).unique().cuda()
            torch.cuda.synchronize()
            tic = time.time()
            buff = feature_cache[index]
            torch.cuda.synchronize()
            if iter >= 10:
                epoch_time0 += time.time() - tic

        # with cache
        cache_num = int(cache_rate * length)
        cache_index = sorted_idx[:cache_num]
        torch.cuda.empty_cache()
        feature_cache.clear_cache()
        feature_cache.cache_data(cache_index)
        epoch_time1 = 0
        for iter in range(num_iters):
            index = torch.randint(0, length, (fetch_size, )).unique().cuda()
            torch.cuda.synchronize()
            tic = time.time()
            buff = feature_cache[index]
            torch.cuda.synchronize()
            if iter >= 10:
                epoch_time1 += time.time() - tic
        hit_times = feature_cache.get_hit_rates()[1]

        print(hit_times)
        print(epoch_time0 - epoch_time1)
        print(slope * hit_times * dim / 1000000)

        del feature_cache
