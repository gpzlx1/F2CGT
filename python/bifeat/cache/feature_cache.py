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

        elif cache_nids.shape[0] > 0:
            self.cached_feature = self.feature[cache_nids].cuda()
            self.hashmap_key, self.hashmap_value = capi._CAPI_create_hashmap(
                cache_nids.cuda())
            cache_size = self.cached_feature.numel(
            ) * self.cached_feature.element_size()

        else:
            self.no_cached = True
            cache_size = 0

        torch.cuda.synchronize()
        end = time.time()

        print(
            "GPU {} takes {:.3f} s to cache feature data, cached size = {:.3f} GB, cache rate = {:.3f}"
            .format(
                torch.cuda.current_device(), end - start,
                cache_size / 1024 / 1024 / 1024, cache_size /
                (self.feature.element_size() * self.feature.numel())))

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


class CompressedFeatureCacheServer:

    def __init__(self, feature_partitions, decompresser):
        self.feature_partitions = feature_partitions
        self.decompresser = decompresser
        self.feat_dim = decompresser.feat_dim

        self.num_parts = len(feature_partitions)
        self.cached_feature_partitions = [None for _ in range(self.num_parts)]
        self.cached_part_size_list = [0 for _ in range(self.num_parts)]

        self.partition_range = torch.zeros(self.num_parts + 1,
                                           dtype=torch.long)
        self.partition_range[1:] = torch.cumsum(
            self.decompresser.part_size_list, dim=0)
        self.partition_range = self.partition_range.cuda()

        for feature in self.feature_partitions:
            capi._CAPI_pin_tensor(feature)

        self.no_cached = True

    def __del__(self):
        for feature in self.feature_partitions:
            capi._CAPI_unpin_tensor(feature)

    def cache_data(self, partition_local_cache_nids_list):
        assert len(partition_local_cache_nids_list) == self.num_parts

        tic = time.time()

        total_global_cache_nids = []

        for i in range(self.num_parts):

            cache_nids = partition_local_cache_nids_list[i]
            cache_num = cache_nids.shape[0]

            if cache_num == 0:
                cache_size = 0
            else:
                self.no_cached = False
                self.cached_feature_partitions[i] = self.feature_partitions[i][
                    cache_nids.cpu()].cuda()
                cache_size = self.cached_feature_partitions[i].numel(
                ) * self.cached_feature_partitions[i].element_size()

            total_global_cache_nids.append(cache_nids +
                                           self.partition_range[i])
            self.cached_part_size_list[i] = cache_num

            print(
                "GPU {} feature partition {}: cached size = {:.3f} GB, cache rate = {:.3f}"
                .format(
                    torch.cuda.current_device(), i,
                    cache_size / 1024 / 1024 / 1024,
                    cache_size / (self.feature_partitions[i].element_size() *
                                  self.feature_partitions[i].numel())))

        total_global_cache_nids = torch.cat(total_global_cache_nids).cuda()
        self.hash_key, self.hash_value = capi._CAPI_create_hashmap(
            total_global_cache_nids)

        self.cached_part_size_list = torch.tensor(
            self.cached_part_size_list).long()
        self.cached_partition_range = torch.zeros(self.num_parts + 1,
                                                  dtype=torch.long)
        self.cached_partition_range[1:] = torch.cumsum(
            self.cached_part_size_list, dim=0)
        self.cached_partition_range = self.partition_range.cuda()

        toc = time.time()
        print("GPU {} takes {:.3f} sec to cache all the feature partitions".
              format(torch.cuda.current_device(), toc - tic))

    def __getitem__(self, index):
        '''
        index is a gpu tensor
        return fetched decompressed feature (a gpu tensor)
        '''
        result = torch.zeros((index.shape[0], self.feat_dim),
                             dtype=torch.float32,
                             device="cuda")

        part_ids = torch.searchsorted(self.partition_range, index,
                                      right=True) - 1

        if not self.no_cached:
            searched_index = capi._CAPI_search_hashmap(self.hash_key,
                                                       self.hash_value, index)
            cached_mask = searched_index != -1

        for i in range(self.num_parts):
            local_mask = part_ids == i

            if not self.no_cached:
                local_cpu_src_index = index[
                    local_mask & (~cached_mask)] - self.partition_range[i]
                local_gpu_src_index = searched_index[
                    local_mask & cached_mask] - self.cached_partition_range[i]

                local_cached_mask = cached_mask[local_mask]
                local_cpu_dst_index = torch.nonzero(
                    (~local_cached_mask)).flatten()
                local_gpu_dst_index = torch.nonzero(
                    local_cached_mask).flatten()

                local_num = local_cached_mask.shape[0]
            else:
                local_cpu_src_index = index[local_mask] - self.partition_range[
                    i]
                local_cpu_dst_index = torch.arange(
                    0,
                    local_cpu_src_index.shape[0],
                    dtype=torch.int64,
                    device="cuda")
                local_num = local_cpu_src_index.shape[0]

            local_compressed_feature = torch.zeros(
                (local_num, self.feature_partitions[i].shape[1]),
                dtype=self.feature_partitions[i].dtype,
                device="cuda")

            if local_cpu_dst_index.shape[0] > 0:
                capi._CAPI_cuda_index_fetch(self.feature_partitions[i],
                                            local_cpu_src_index,
                                            local_compressed_feature,
                                            local_cpu_dst_index)

            if not self.no_cached and local_gpu_dst_index.shape[0] > 0:
                capi._CAPI_cuda_index_fetch(self.cached_feature_partitions[i],
                                            local_gpu_src_index,
                                            local_compressed_feature,
                                            local_gpu_dst_index)

            result[local_mask] = self.decompresser.decompress(
                local_compressed_feature,
                index[local_mask] - self.partition_range[i], i)

        return result

    def clear_cache(self):
        self.cached_feature_partitions = [None for _ in range(self.num_parts)]
        self.cached_part_size_list = [0 for _ in range(self.num_parts)]
        self.cached_partition_range = None
        self.hash_key, self.hash_value = None, None
        self.no_cached = True
