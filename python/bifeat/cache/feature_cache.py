import torch
import time
import BiFeatLib as capi


class FeatureCacheServer:

    def __init__(self, feature, count_hit=False):
        self.feature = feature
        self.cached_feature = None
        self.hashmap_key = None
        self.hashmap_value = None
        self.full_cached = False
        self.no_cached = False

        capi._CAPI_pin_tensor(self.feature)

        self.access_times = 0
        self.hit_times = 0

        self._count_hit = count_hit

    def __del__(self):
        capi._CAPI_unpin_tensor(self.feature)

    def cache_data(self, cache_nids):
        start = time.time()

        if cache_nids.shape[0] == self.feature.shape[0]:
            self.full_cached = True
            self.cached_feature = self.feature.cuda()
            cache_size = self.cached_feature.numel(
            ) * self.cached_feature.element_size()
            hashmap_size = 0

        elif cache_nids.shape[0] > 0:
            self.cached_feature = self.feature[cache_nids].cuda()
            self._hashmap = capi.BiFeatHashmaps(1, [cache_nids.int().cuda()])

            cache_size = self.cached_feature.numel(
            ) * self.cached_feature.element_size()
            hashmap_size = self._hashmap.get_memory_usage()

        else:
            self.no_cached = True
            cache_size = 0
            hashmap_size = 0

        torch.cuda.synchronize()
        end = time.time()

        print(
            "GPU {} takes {:.3f} s to cache feature data, cached size = {:.3f} GB, cache rate = {:.3f}"
            .format(
                torch.cuda.current_device(), end - start,
                cache_size / 1024 / 1024 / 1024, cache_size /
                (self.feature.element_size() * self.feature.numel())))
        print("GPU {} Hashmap size = {:.3f} GB".format(
            torch.cuda.current_device(), hashmap_size / 1024 / 1024 / 1024))

    def __getitem__(self, index):
        '''
        index is a gpu tensor
        return fetched feature (a gpu tensor)
        '''
        if self.full_cached:
            if self._count_hit:
                self.access_times += index.shape[0]
                self.hit_times += index.shape[0]
            return self.cached_feature[index]
        elif self.no_cached:
            if self._count_hit:
                self.access_times += index.shape[0]
            return capi._CAPI_fetch_feature_data(self.feature, index)
        else:
            index = index.int()
            local_nids = self._hashmap.query(index, 0)
            result = capi._CAPI_fetch_feature_data_with_caching_v2(
                self.feature, self.cached_feature, index, local_nids)

            if self._count_hit:
                self.access_times += index.shape[0]
                self.hit_times += torch.sum(local_nids >= 0).item()

            return result

    def clear_cache(self):
        self.cached_feature = None

        self.hashmap_key = None
        self.hashmap_value = None

        self.full_cached = False
        self.no_cached = False

        self.access_times = 0
        self.hit_times = 0

    def get_hit_rates(self):
        if self.access_times == 0:
            return (0, 0, 0.0)
        else:
            return (
                self.access_times,
                self.hit_times,
                self.hit_times / self.access_times,
            )

    def reset_hit_counts(self):
        self.access_times = 0
        self.hit_times = 0


class FeatureLoadServer:

    def __init__(self,
                 core_compressed_feature,
                 core_idx,
                 compressed_feature,
                 decompresser,
                 count_hit=False):
        self._core_compressed_feature = core_compressed_feature
        self._compressed_feature = compressed_feature
        self._core_idx = core_idx
        self._decompresser = decompresser
        self._feat_dim = decompresser.feat_dim

        capi._CAPI_pin_tensor(self._compressed_feature)
        capi._CAPI_pin_tensor(self._core_compressed_feature)

        self.full_cached = False
        self.no_cached = True

        self.core_full_cached = False
        self.core_no_cached = True

        self.access_times = 0
        self.core_hit_times = 0
        self.other_hit_times = 0

        self._count_hit = count_hit

        self._core_hash_map = capi.BiFeatHashmaps(
            1, [self._core_idx.int().cuda()])
        print("GPU {} Core index hashmap size = {:.3f} GB".format(
            torch.cuda.current_device(),
            self._core_hash_map.get_memory_usage() / 1024 / 1024 / 1024))

    def __del__(self):
        capi._CAPI_unpin_tensor(self._compressed_feature)
        capi._CAPI_unpin_tensor(self._core_compressed_feature)

    def cache_feature(self, cache_nids):
        start = time.time()

        if cache_nids.shape[0] == self._compressed_feature.shape[0]:
            self.full_cached = True
            self.no_cached = False
            self._cached_feature = self._compressed_feature.cuda()
            cache_size = self._cached_feature.numel(
            ) * self._cached_feature.element_size()
            hashmap_size = 0

        elif cache_nids.shape[0] > 0:
            self.no_cached = False
            self._cached_feature = self._compressed_feature[
                cache_nids.cpu()].cuda()
            self._cache_hashmap = capi.BiFeatHashmaps(
                1, [cache_nids.int().cuda()])

            cache_size = self._cached_feature.numel(
            ) * self._cached_feature.element_size()
            hashmap_size = self._cache_hashmap.get_memory_usage()

        else:
            self.no_cached = True
            cache_size = 0
            hashmap_size = 0

        torch.cuda.synchronize()
        end = time.time()

        print(
            "GPU {} takes {:.3f} s to cache feature data, cached size = {:.3f} GB, cache rate = {:.3f}"
            .format(
                torch.cuda.current_device(), end - start,
                cache_size / 1024 / 1024 / 1024,
                cache_size / (self._compressed_feature.element_size() *
                              self._compressed_feature.numel())))
        print("GPU {} Hashmap size = {:.3f} GB".format(
            torch.cuda.current_device(), hashmap_size / 1024 / 1024 / 1024))

    def cache_core_feature(self, core_cache_nids):
        start = time.time()

        if core_cache_nids.shape[0] == self._core_compressed_feature.shape[0]:
            self.core_full_cached = True
            self.core_no_cached = False
            self._core_cached_feature = self._core_compressed_feature.cuda()
            cache_size = self._core_cached_feature.numel(
            ) * self._core_cached_feature.element_size()
            hashmap_size = 0

        elif core_cache_nids.shape[0] > 0:
            self.core_no_cached = False
            self._core_cached_feature = self._core_compressed_feature[
                core_cache_nids.cpu()].cuda()
            self._core_cache_hashmap = capi.BiFeatHashmaps(
                1, [core_cache_nids.int().cuda()])

            cache_size = self._core_cached_feature.numel(
            ) * self._core_cached_feature.element_size()
            hashmap_size = self._core_cache_hashmap.get_memory_usage()

        else:
            self.core_no_cached = True
            cache_size = 0
            hashmap_size = 0

        torch.cuda.synchronize()
        end = time.time()

        print(
            "GPU {} takes {:.3f} s to cache Core feature data, cached size = {:.3f} GB, cache rate = {:.3f}"
            .format(
                torch.cuda.current_device(), end - start,
                cache_size / 1024 / 1024 / 1024,
                cache_size / (self._core_compressed_feature.element_size() *
                              self._core_compressed_feature.numel())))
        print("GPU {} Core Hashmap size = {:.3f} GB".format(
            torch.cuda.current_device(), hashmap_size / 1024 / 1024 / 1024))

    def __getitem__(self, data):
        '''
        data = index, seeds_num
        - index is the frontier of a mini-batch, a gpu tensor
        - seeds_num: the number of seeds in this mini-batch
          * index[:seeds_num] = seeds
          * seeds_num can be 0 (for inference)
        - return fetched feature (a gpu tensor)
        '''
        index, seeds_num = data
        index = index.cuda().int()

        if self._count_hit:
            self.access_times += index.shape[0]

        if seeds_num > 0:
            searched_seeds_index = self._core_hash_map.query(
                index[:seeds_num], 0)

            if self.core_full_cached:
                if self._count_hit:
                    self.core_hit_times += seeds_num
                seeds_compressed_features = self._core_cached_feature[
                    searched_seeds_index.long()]
            elif self.core_no_cached:
                seeds_compressed_features = capi._CAPI_fetch_feature_data(
                    self._core_compressed_feature, searched_seeds_index)
            else:
                cache_nids = self._core_cache_hashmap.query(
                    searched_seeds_index, 0)
                seeds_compressed_features = capi._CAPI_fetch_feature_data_with_caching_v2(
                    self._core_compressed_feature, self._core_cached_feature,
                    searched_seeds_index, cache_nids)
                if self._count_hit:
                    self.core_hit_times += torch.sum(cache_nids >= 0).item()

        else:
            seeds_compressed_features = torch.empty(
                (0, self._core_compressed_feature.shape[1]),
                dtype=self._core_compressed_feature.dtype,
                device="cuda")

        if self.full_cached:
            if self._count_hit:
                self.other_hit_times += index.shape[0] - seeds_num
            frontier_compressed_features = self._cached_feature[
                index[seeds_num:].long()]
        elif self.no_cached:
            frontier_compressed_features = capi._CAPI_fetch_feature_data(
                self._compressed_feature, index[seeds_num:])
        else:
            local_nids = self._cache_hashmap.query(index[seeds_num:], 0)
            frontier_compressed_features = capi._CAPI_fetch_feature_data_with_caching_v2(
                self._compressed_feature, self._cached_feature,
                index[seeds_num:], local_nids)
            if self._count_hit:
                self.other_hit_times += torch.sum(local_nids >= 0).item()

        result = torch.empty((index.numel(), self._decompresser.feat_dim),
                             dtype=torch.float,
                             device='cuda')
        self._decompresser.decompress_v2(seeds_compressed_features,
                                         searched_seeds_index, 0, result, 0)
        self._decompresser.decompress_v2(frontier_compressed_features,
                                         index[seeds_num:], 1, result,
                                         seeds_num)

        return result

    def get_hit_rates(self):
        if self.access_times == 0:
            return (0, 0, 0.0, 0, 0.0)
        else:
            return (
                self.access_times,
                self.core_hit_times,
                self.core_hit_times / self.access_times,
                self.other_hit_times,
                self.other_hit_times / self.access_times,
            )

    def reset_hit_counts(self):
        self.access_times = 0
        self.core_hit_times = 0
        self.other_hit_times = 0
