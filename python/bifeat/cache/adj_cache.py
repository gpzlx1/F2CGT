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

        # self.cached_nids_hashed = None
        # self.cached_nids_in_gpu_hashed = None
        self._hashmap = None

        self.full_cached = False
        self.no_cached = True

        self._fan_out = fan_out
        self._count_hit = count_hit

        self._num_layers = len(fan_out)

        self.access_times = [0 for i in range(self._num_layers)]
        self.hit_times = [0 for i in range(self._num_layers)]

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
            hashmap_size = 0

        elif cache_nids.shape[0] <= 0:
            self.no_cached = True
            indptr_cached_size = 0
            indices_cached_size = 0
            hashmap_size = 0
        else:
            self.no_cached = False
            cache_nids = cache_nids.cuda(self.device_id)
            # self.cached_nids_hashed, self.cached_nids_in_gpu_hashed = capi._CAPI_create_hashmap(
            #     cache_nids)
            self._hashmap = capi.BiFeatHashmaps(1, [cache_nids.int().cuda()])

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
            hashmap_size = self._hashmap.get_memory_usage()

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
        print("GPU {} Hashmap size = {:.3f} GB".format(
            self.device_id, hashmap_size / 1024 / 1024 / 1024))

    def clear_cache(self):
        self.cached_indptr = None
        self.cached_indices = None

        # self.cached_nids_hashed = None
        # self.cached_nids_in_gpu_hashed = None
        self._hashmap = None

        self.full_cached = False
        self.no_cached = False

        self.access_times = [0 for i in range(self._num_layers)]
        self.hit_times = [0 for i in range(self._num_layers)]

    def get_hit_rates(self):
        if self.access_times[-1] == 0:
            return ([0 for i in range(self._num_layers)
                     ], [0 for i in range(self._num_layers)],
                    [0.0 for i in range(self._num_layers)])
        else:
            hit_rates = [
                self.hit_times[i] / self.access_times[i]
                for i in range(self._num_layers)
            ]
            return (
                self.access_times,
                self.hit_times,
                hit_rates,
            )

    def reset_hit_counts(self):
        self.access_times = [0 for i in range(self._num_layers)]
        self.hit_times = [0 for i in range(self._num_layers)]

    def sample_neighbors(self, seeds_nids, replace=False):
        seeds = seeds_nids.cuda(self.device_id)
        blocks = []

        for i, num_picks in enumerate(reversed(self._fan_out)):

            if self.full_cached:
                if self._count_hit:
                    self.access_times[self._num_layers - 1 -
                                      i] += seeds.shape[0]
                    self.hit_times[self._num_layers - 1 - i] += seeds.shape[0]
                coo_row, coo_col = capi._CAPI_cuda_sample_neighbors(
                    seeds, self.cached_indptr, self.cached_indices, num_picks,
                    replace)

            elif self.no_cached:
                if self._count_hit:
                    self.access_times[self._num_layers - 1 -
                                      i] += seeds.shape[0]
                coo_row, coo_col = capi._CAPI_cuda_sample_neighbors(
                    seeds, self.indptr, self.indices, num_picks, replace)

            else:
                # local_nids = capi._CAPI_search_hashmap(
                #     self.cached_nids_hashed, self.cached_nids_in_gpu_hashed,
                #     seeds).int()
                local_nids = self._hashmap.query(seeds, 0)
                coo_row, coo_col = capi._CAPI_cuda_sample_neighbors_with_caching(
                    seeds, self.cached_indptr, self.indptr,
                    self.cached_indices, self.indices, local_nids, num_picks,
                    replace)
                if self._count_hit:
                    self.access_times[self._num_layers - 1 -
                                      i] += seeds.shape[0]
                    self.hit_times[self._num_layers - 1 -
                                   i] += torch.sum(local_nids >= 0).item()

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
