import torch
import time
import BiFeatLib as capi


class FeatureCacheServer:

    def __init__(self, core_feature, core_nids, other_feature, other_nids):
        self._core_feature = core_feature
        self._other_feature = other_feature

        self._num_nodes = core_nids.shape[0] + other_nids.shape[0]

        # to split nids into core & other
        self._core_mask = torch.zeros((self._num_nodes, ), dtype=torch.bool)
        self._core_mask[core_nids] = True
        self._core_mask = self._core_mask.cuda()

        # map nids to idx in core_feature or other_feature
        self._idx_map = torch.argsort(torch.cat([core_nids, other_nids]))
        self._idx_map[self._idx_map >=
                      core_nids.shape[0]] -= core_nids.shape[0]
        self._idx_map = self._idx_map.cuda()

        self._cached_core_feature = None
        self._cached_other_feature = None

        self._core_hashmap_key = None
        self._core_hashmap_value = None
        self._core_full_cached = False
        self._core_no_cached = False

        self._other_hashmap_key = None
        self._other_hashmap_value = None
        self._other_full_cached = False
        self._other_no_cached = False

    def __del__(self):
        if not self._core_full_cached:
            capi._CAPI_unpin_tensor(self._core_feature)
        if not self._other_full_cached:
            capi._CAPI_unpin_tensor(self._other_feature)

    def cache_data(self, core_cache_idx, other_cache_idx):
        tic = time.time()

        if core_cache_idx.shape[0] == self._core_feature.shape[0]:
            self._core_full_cached = True
            self._cached_core_feature = self._core_feature.cuda()
            core_mem_used = self._cached_core_feature.element_size(
            ) * self._cached_core_feature.numel()
        elif core_cache_idx.shape[0] == 0:
            self._core_no_cached = True
            capi._CAPI_pin_tensor(self._core_feature)
            core_mem_used = 0
        else:
            self._core_hashmap_key, self._core_hashmap_value = capi._CAPI_create_hashmap(
                core_cache_idx.cuda())
            self._cached_core_feature = self._core_feature[
                core_cache_idx].cuda()
            capi._CAPI_pin_tensor(self._core_feature)
            core_mem_used = self._cached_core_feature.element_size(
            ) * self._cached_core_feature.numel(
            ) + self._core_hashmap_key.element_size(
            ) * self._core_hashmap_key.numel(
            ) + self._core_hashmap_value.element_size(
            ) * self._core_hashmap_value.numel()

        if other_cache_idx.shape[0] == self._other_feature.shape[0]:
            self._other_full_cached = True
            self._cached_other_feature = self._other_feature.cuda()
            other_mem_used = self._cached_other_feature.element_size(
            ) * self._cached_other_feature.numel()
        elif other_cache_idx.shape[0] == 0:
            self._other_no_cached = True
            capi._CAPI_pin_tensor(self._other_feature)
            other_mem_used = 0
        else:
            self._other_hashmap_key, self._other_hashmap_value = capi._CAPI_create_hashmap(
                other_cache_idx.cuda())
            self._cached_other_feature = self._other_feature[
                other_cache_idx].cuda()
            capi._CAPI_pin_tensor(self._other_feature)
            other_mem_used = self._cached_other_feature.element_size(
            ) * self._cached_other_feature.numel(
            ) + self._other_hashmap_key.element_size(
            ) * self._other_hashmap_key.numel(
            ) + self._other_hashmap_value.element_size(
            ) * self._other_hashmap_value.numel()

        toc = time.time()

        info = "========================================\n"
        info += "Rank {} builds feature cache in GPU\n".format(
            torch.cuda.current_device())
        info += "Core feature cache num {}, used mem {:.3f} GB\n".format(
            core_cache_idx.shape[0], core_mem_used / 1024 / 1024 / 1024)
        info += "Other feature cache num {}, used mem {:.3f} GB\n".format(
            other_cache_idx.shape[0], other_mem_used / 1024 / 1024 / 1024)
        info += "Build cache time {:.3f} ms\n".format((toc - tic) * 1000)
        info += "========================================"
        print(info)

    def __getitem__(self, nids):
        '''
        nids is a gpu tensor
        and return fetched_core_feature, fetched_other_feature, core_mask
        '''
        core_mask = self._core_mask[nids]

        core_nids = nids[core_mask]
        core_idx = self._idx_map[core_nids]
        if self._core_full_cached:
            fetched_core_features = self._cached_core_feature[core_idx]
        elif self._core_no_cached:
            fetched_core_features = capi._CAPI_fetch_feature_data(
                self._core_feature, core_idx)
        else:
            fetched_core_features = capi._CAPI_fetch_feature_data_with_caching(
                self._core_feature,
                self._cached_core_feature,
                self._core_hashmap_key,
                self._core_hashmap_value,
                core_idx,
            )

        other_nids = nids[~core_mask]
        other_idx = self._idx_map[other_nids]
        if self._other_full_cached:
            fetched_other_features = self._cached_other_feature[other_idx]
        elif self._other_no_cached:
            fetched_other_features = capi._CAPI_fetch_feature_data(
                self._other_feature, other_idx)
        else:
            fetched_other_features = capi._CAPI_fetch_feature_data_with_caching(
                self._other_feature,
                self._cached_other_feature,
                self._other_hashmap_key,
                self._other_hashmap_value,
                other_idx,
            )

        return fetched_core_features, fetched_other_features, core_mask
