import torch
import time
import BiFeatLib as capi


class FeatureCache:

    def __init__(self, feature, enalbe_uva=True):
        self._feature = feature
        self._cached_feature = None
        self._full_cached = False
        self._no_cached = False
        self._num_nodes = feature.shape[0]
        self._num_cached = 0
        self._hit_num = 0
        self._lookup_num = 0
        self._enable_uva = enalbe_uva
        if self._enable_uva:
            capi._CAPI_pin_tensor(self._feature)

        self.cache_built = False

    def create_cache(self, alloca_mem_size):
        self.cache_built = True

        tic = time.time()

        mem_used = 0

        info = "========================================\n"
        info += "Rank {} builds feature cache in GPU\n".format(
            torch.cuda.current_device())
        info += "GPU alloca_mem_size {:.3f} GB\n".format(alloca_mem_size /
                                                         1024 / 1024 / 1024)

        item_size = self._feature.element_size() * self._feature.shape[1]
        cached_num = min(alloca_mem_size // item_size, self._num_nodes)

        info += "Total node num {}\n".format(self._num_nodes)
        info += "GPU cached node num {}\n".format(cached_num)

        self._num_cached = cached_num
        if cached_num <= 0:
            self._no_cached = True
        elif cached_num == self._num_nodes:
            self._full_cached = True
            if self._enable_uva:
                capi._CAPI_unpin_tensor(self._feature)
            self._cached_feature = self._feature.cuda()
        else:
            self._cached_feature = self._feature[:cached_num].cuda()
        mem_used = cached_num * item_size
        toc = time.time()

        self._hit_num = 0
        self._lookup_num = 0
        info += "GPU cache size {:.3f} GB\n".format(mem_used / 1024 / 1024 /
                                                    1024)
        info += "Build cache time {:.3f} ms\n".format((toc - tic) * 1000)
        info += "========================================"
        print(info)

        return mem_used

    def __getitem__(self, index):
        '''
        index is a gpu tensor
        and return gpu tensors (step, mem, power)
        '''
        if self._full_cached:
            return self._cached_feature[index]
        elif self._no_cached or not self.cache_built:
            if self._enable_uva:
                return capi._CAPI_fetch_feature_data(self._feature, index)
            else:
                return self._feature[index]
        else:
            if self._enable_uva:
                return capi._CAPI_fetch_feature_data_with_caching(
                    self._feature,
                    self._cached_feature,
                    index,
                    self._num_cached,
                )
            else:
                cached_mask = index < self._num_cached
                uncached_mask = ~cached_mask
                data = torch.zeros((index.shape[0], self._feature.shape[1]),
                                   dtype=self._feature.dtype,
                                   device="cuda")
                data[cached_mask] = self._cached_feature[index[cached_mask]]
                data[uncached_mask] = self._feature[index[uncached_mask]]
                return data

    def __del__(self):
        if self._enable_uva and not self._full_cached:
            capi._CAPI_unpin_tensor(self._feature)
