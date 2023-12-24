import torch
import time
import dgl
import BiFeatLib as capi


class StructureCacheServer:

    def __init__(self,
                 indptr,
                 indices,
                 fan_out,
                 count_hit=False,
                 pin_memory=True):
        self.indptr = indptr
        self.indices = indices
        self._pin_indptr = not indptr.is_pinned
        self._pin_indices = not indices.is_pinned
        self._pin_memory = pin_memory

        # if pin_memory is False, then indptr and indices must be pinned before.
        # Otherwise, cuda illegal memory access will occur during sampling.
        if pin_memory:
            capi._CAPI_pin_tensor(indptr)
            capi._CAPI_pin_tensor(indices)

        self.cached_indptr = None
        self.cached_indices = None

        self.device_id = torch.cuda.current_device()

        self._hashmap = None

        self.full_cached = False
        self.no_cached = True

        self._fan_out = fan_out
        self._count_hit = count_hit

        self.access_times = 0
        self.hit_times = 0

    def __del__(self):
        if self._pin_memory:
            capi._CAPI_unpin_tensor(self.indptr)
            capi._CAPI_unpin_tensor(self.indices)

    def cache_data(self, cache_nids):
        start = time.time()

        if cache_nids.shape[0] >= self.indptr.shape[0] - 1:
            self.full_cached = True
            self.no_cached = False

            self.cached_indptr = self.indptr.cuda(self.device_id)
            self.cached_indices = self.indices.cuda(self.device_id)

            indptr_cached_size = self.indptr.element_size(
            ) * self.indptr.numel()
            indices_cached_size = self.indices.element_size(
            ) * self.indices.numel()
            # hashmap_size = 0

        elif cache_nids.shape[0] <= 0:
            self.no_cached = True
            indptr_cached_size = 0
            indices_cached_size = 0
            # hashmap_size = 0
        else:
            self.no_cached = False
            cache_nids = cache_nids.cuda(self.device_id)
            self._hashmap = capi.CacheHashMap()
            self._hashmap.insert(cache_nids)

            self.cached_indptr = capi._CAPI_get_sub_indptr(
                cache_nids, self.indptr).cuda(self.device_id)

            self.cached_indices = capi._CAPI_get_sub_edge_data(
                cache_nids, self.indptr, self.cached_indptr,
                self.indices).cuda(self.device_id)

            indptr_cached_size = self.cached_indptr.element_size(
            ) * self.cached_indptr.numel()
            indices_cached_size = self.cached_indices.element_size(
            ) * self.cached_indices.numel()

            # hashmap_size = self.cached_nids_hashed.numel(
            # ) * self.cached_nids_hashed.element_size()
            # hashmap_size += self.cached_nids_in_gpu_hashed.numel(
            # ) * self.cached_nids_in_gpu_hashed.element_size()

        end = time.time()

        print("GPU {} takes {:.3f} s to cache structure data".format(
            self.device_id, end - start))
        print(
            "GPU {} Indptr cache size = {:.3f} GB, cache rate = {:.3f}".format(
                self.device_id, indptr_cached_size / 1024 / 1024 / 1024,
                indptr_cached_size /
                (self.indptr.element_size() * self.indptr.numel())))
        print("GPU {} Indices cache size = {:.3f} GB, cache rate = {:.3f}".
              format(
                  self.device_id, indices_cached_size / 1024 / 1024 / 1024,
                  indices_cached_size /
                  (self.indices.element_size() * self.indices.numel())))
        # print("GPU {} Hashmap size = {:.3f} GB".format(
        #     self.device_id, hashmap_size / 1024 / 1024 / 1024))

    def clear_cache(self):
        self.cached_indptr = None
        self.cached_indices = None

        self._hashmap = None

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

    def sample_neighbors(self, seeds_nids, replace=False):
        seeds = seeds_nids.cuda(self.device_id)
        blocks = []

        for num_picks in reversed(self._fan_out):

            if self.full_cached:
                if self._count_hit:
                    self.access_times += seeds.shape[0]
                    self.hit_times += seeds.shape[0]
                coo_row, coo_col = capi._CAPI_cuda_sample_neighbors(
                    seeds, self.cached_indptr, self.cached_indices, num_picks,
                    replace)

            elif self.no_cached:
                if self._count_hit:
                    self.access_times += seeds.shape[0]
                coo_row, coo_col = capi._CAPI_cuda_sample_neighbors(
                    seeds, self.indptr, self.indices, num_picks, replace)

            else:
                idx_in_cache = self._hashmap.find(seeds)
                if self._count_hit:
                    self.access_times += seeds.shape[0]
                    self.hit_times += torch.sum(idx_in_cache != -1).item()
                coo_row, coo_col = capi._CAPI_cuda_sample_neighbors_with_caching(
                    seeds, self.cached_indptr, self.indptr,
                    self.cached_indices, self.indices, idx_in_cache, num_picks,
                    replace)

            frontier, (coo_row, coo_col) = capi._CAPI_cuda_tensor_relabel(
                [seeds, coo_col], [coo_row, coo_col])

            block = dgl.create_block((coo_col, coo_row),
                                     num_src_nodes=frontier.numel(),
                                     num_dst_nodes=seeds.numel())
            block.srcdata[dgl.NID] = frontier
            block.dstdata[dgl.NID] = seeds
            blocks.insert(0, block)

            seeds = frontier

        return frontier, seeds_nids, blocks
