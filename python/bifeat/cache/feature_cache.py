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
            self.hashmap_key, self.hashmap_value = capi._CAPI_create_hashmap(
                cache_nids.cuda())
            cache_size = self.cached_feature.numel(
            ) * self.cached_feature.element_size()

            hashmap_size = self.hashmap_key.numel(
            ) * self.hashmap_key.element_size()
            hashmap_size += self.hashmap_value.numel(
            ) * self.hashmap_value.element_size()

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
            if self._count_hit:
                self.access_times += index.shape[0]
                self.hit_times += capi._CAPI_count_cached_nids(
                    index, self.hashmap_key, self.hashmap_value)
            return capi._CAPI_fetch_feature_data_with_caching(
                self.feature,
                self.cached_feature,
                self.hashmap_key,
                self.hashmap_value,
                index,
            )

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


class FeatureLoadServer:

    def __init__(self, core_compressed_feature, core_idx, compressed_feature,
                 decompresser):
        self._core_compressed_feature = core_compressed_feature
        self._compressed_feature = compressed_feature
        self._core_idx = core_idx
        self._decompresser = decompresser
        self._feat_dim = decompresser.feat_dim

        capi._CAPI_pin_tensor(self._compressed_feature)
        capi._CAPI_pin_tensor(self._core_compressed_feature)

        self._full_cached = False
        self.no_cached = True

        self._core_hashmap_key, self._core_hashmap_value = capi._CAPI_create_hashmap(
            core_idx.cuda())
        core_idx_hashmap_size = self._core_hashmap_key.numel(
        ) * self._core_hashmap_key.element_size(
        ) + self._core_hashmap_value.numel(
        ) * self._core_hashmap_value.element_size()
        print("GPU {} Core index hashmap size = {:.3f} GB".format(
            torch.cuda.current_device(),
            core_idx_hashmap_size / 1024 / 1024 / 1024))

    def __del__(self):
        capi._CAPI_unpin_tensor(self._compressed_feature)
        capi._CAPI_unpin_tensor(self._core_compressed_feature)

    def cache_data(self, cache_nids):
        start = time.time()

        if cache_nids.shape[0] == self._compressed_feature.shape[0]:
            self._full_cached = True
            self.no_cached = False
            self._cached_feature = self._compressed_feature.cuda()
            cache_size = self._cached_feature.numel(
            ) * self._cached_feature.element_size()
            hashmap_size = 0

        elif cache_nids.shape[0] > 0:
            self.no_cached = False
            self._cached_feature = self._compressed_feature[
                cache_nids.cpu()].cuda()
            self._hashmap_key, self._hashmap_value = capi._CAPI_create_hashmap(
                cache_nids.cuda())
            cache_size = self._cached_feature.numel(
            ) * self._cached_feature.element_size()

            hashmap_size = self._hashmap_key.numel(
            ) * self._hashmap_key.element_size()
            hashmap_size += self._hashmap_value.numel(
            ) * self._hashmap_value.element_size()

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

        if seeds_num > 0:
            searched_seeds_index = capi._CAPI_search_hashmap(
                self._core_hashmap_key, self._core_hashmap_value,
                index[:seeds_num])
            searched_mask = searched_seeds_index != -1
            seeds_src_idx = searched_seeds_index[searched_mask]
            searched_seeds_num = seeds_src_idx.shape[0]

            if searched_seeds_num > 0:
                seeds_compressed_features = capi._CAPI_fetch_feature_data(
                    self._core_compressed_feature, seeds_src_idx)

                frontier_mask = torch.cat([
                    ~searched_mask,
                    torch.ones((index.shape[0] - seeds_num, ),
                               dtype=torch.bool,
                               device="cuda")
                ])
                frontier_src_idx = index[frontier_mask]
                if self._full_cached:
                    frontier_compressed_features = self._cached_feature[
                        frontier_src_idx]
                elif self.no_cached:
                    frontier_compressed_features = capi._CAPI_fetch_feature_data(
                        self._compressed_feature, frontier_src_idx)
                else:
                    frontier_compressed_features = capi._CAPI_fetch_feature_data_with_caching(
                        self._compressed_feature, self._cached_feature,
                        self._hashmap_key, self._hashmap_value,
                        frontier_src_idx)

                result = torch.zeros((index.shape[0], self._feat_dim),
                                     dtype=torch.float32,
                                     device="cuda")
                result[~frontier_mask] = self._decompresser.decompress(
                    seeds_compressed_features, seeds_src_idx, 0)
                result[frontier_mask] = self._decompresser.decompress(
                    frontier_compressed_features, frontier_src_idx, 1)
                return result

        if self._full_cached:
            compressed_features = self._cached_feature[index]
        elif self.no_cached:
            compressed_features = capi._CAPI_fetch_feature_data(
                self._compressed_feature, index)
        else:
            compressed_features = capi._CAPI_fetch_feature_data_with_caching(
                self._compressed_feature, self._cached_feature,
                self._hashmap_key, self._hashmap_value, index)
        return self._decompresser.decompress(compressed_features, index, 1)

    def clear_cache(self):
        self._cached_feature = None
        self._hashmap_key = None
        self._hashmap_value = None
        self._full_cached = False
        self.no_cached = True
