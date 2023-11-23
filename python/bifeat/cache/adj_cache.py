import torch
import time
import dgl
import BiFeatLib as capi


class StructureCacheServer:

    def __init__(self, indptr, indices):
        self.indptr = indptr
        self.indices = indices
        capi._CAPI_pin_tensor(indptr)
        capi._CAPI_pin_tensor(indices)

        self.cached_indptr = None
        self.cached_indices = None

        self.device_id = torch.cuda.current_device()

        self.cached_nids_hashed = None
        self.cached_nids_in_gpu_hashed = None

        self.full_cached = False
        self.no_cache = False

        self.access_times = 0
        self.hit_times = 0

    def __del__(self):
        capi._CAPI_unpin_tensor(self.indptr)
        capi._CAPI_unpin_tensor(self.indices)

    def cache_data(self, cache_nids):
        start = time.time()

        if cache_nids.shape[0] >= self.indptr.shape[0] - 1:
            self.full_cached = True

            self.cached_indptr = self.indptr.cuda(self.device_id)
            self.cached_indices = self.indices.cuda(self.device_id)

            indptr_cached_size = self.indptr.element_size(
            ) * self.indptr.numel()
            indices_cached_size = self.indices.element_size(
            ) * self.indices.numel()

        elif cache_nids.shape[0] <= 0:
            self.no_cache = True
            indptr_cached_size = 0
            indices_cached_size = 0
        else:
            cache_nids = cache_nids.cuda(self.device_id)
            self.cached_nids_hashed, self.cached_nids_in_gpu_hashed = capi._CAPI_create_hashmap(
                cache_nids)

            self.cached_indptr = capi._CAPI_get_sub_indptr(
                cache_nids, self.indptr).cuda(self.device_id)

            self.cached_indices = capi._CAPI_get_sub_edge_data(
                cache_nids, self.indptr, self.cached_indptr,
                self.indices).cuda(self.device_id)

            indptr_cached_size = self.cached_indptr.element_size(
            ) * self.cached_indptr.numel()
            indices_cached_size = self.cached_indices.element_size(
            ) * self.cached_indices.numel()

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

    def sample_neighbors(self, seeds_nids, fan_out, replace=False):
        seeds = seeds_nids.cuda(self.device_id)
        blocks = []

        for num_picks in reversed(fan_out):

            if self.full_cached:
                coo_row, coo_col = capi._CAPI_cuda_sample_neighbors(
                    seeds, self.cached_indptr, self.cached_indices, num_picks,
                    replace)

            elif self.no_cache:
                coo_row, coo_col = capi._CAPI_cuda_sample_neighbors(
                    seeds, self.indptr, self.indices, num_picks, replace)

            else:
                coo_row, coo_col = capi._CAPI_cuda_sample_neighbors_with_caching(
                    seeds, self.cached_indptr, self.indptr,
                    self.cached_indices, self.indices, self.cached_nids_hashed,
                    self.cached_nids_in_gpu_hashed, num_picks, replace)

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