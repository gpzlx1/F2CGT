import torch
import torch.distributed as dist
import ShmTensorLib as capi
import weakref
import dgl
from dgl.heterograph import DGLBlock


def allocate(cap, ratios, limits):
    total_ratio = sum(ratios)
    limit_ratios = [limits[i] / cap for i in range(len(ratios))]

    allocated_ratios = []
    is_max = []
    for i in range(len(ratios)):
        if ratios[i] / total_ratio < limit_ratios[i]:
            allocated_ratios.append(ratios[i] / total_ratio)
            is_max.append(False)
        else:
            allocated_ratios.append(limit_ratios[i])
            is_max.append(True)

    if sum(allocated_ratios) >= 1 - 1e-5 or sum(is_max) == len(is_max):
        return [int(cap * i) + 4 for i in allocated_ratios]

    remain_ratios = 1 - sum(allocated_ratios)

    for i in range(10):
        if sum(allocated_ratios) >= 1 - 1e-5 or sum(is_max) == len(is_max):
            break

        for i in range(len(ratios)):
            if is_max[i]:
                continue

            append_ratio = ratios[i] / total_ratio * remain_ratios

            if append_ratio < limit_ratios[i] - allocated_ratios[i]:
                allocated_ratios[i] += append_ratio
            else:
                allocated_ratios[i] = limit_ratios[i]
                is_max[i] = True

        remain_ratios = 1 - sum(allocated_ratios)

    return [int(cap * i) + 4 for i in allocated_ratios]


def create_block_from_csc(indptr, indices, e_ids, num_src, num_dst):
    hgidx = dgl.heterograph_index.create_unitgraph_from_csr(
        2,
        num_src,
        num_dst,
        indptr,
        indices,
        e_ids,
        formats=['coo', 'csr', 'csc'],
        transpose=True)
    retg = DGLBlock(hgidx, (['_N'], ['_N']), ['_E'])
    return retg


def create_block_from_coo(row, col, num_src, num_dst):
    hgidx = dgl.heterograph_index.create_unitgraph_from_coo(
        2, num_src, num_dst, row, col, formats=['coo', 'csr', 'csc'])
    retg = DGLBlock(hgidx, (['_N'], ['_N']), ['_E'])
    return retg


class GPUSamplingDataloader:

    def __init__(self,
                 indptr,
                 indices,
                 seeds,
                 batchsize,
                 num_picks,
                 compression_manager,
                 replace=False,
                 use_ddp=False,
                 shuffle=True,
                 drop_last=False):
        self.indptr = indptr
        self.indices = indices
        self.seeds = seeds
        self.batchsize = batchsize
        self.num_picks = num_picks
        self.replace = replace
        self.use_ddp = use_ddp
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.compression_manager = compression_manager

        if self.indptr.device == torch.device('cpu'):
            capi.pin_memory(self.indptr)
            weakref.finalize(self, capi.unpin_memory, self.indptr)

        if self.indices.device == torch.device('cpu'):
            capi.pin_memory(self.indices)
            weakref.finalize(self, capi.unpin_memory, self.indices)

        if self.use_ddp:
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            numel = self.seeds.numel()
            partition_size = (numel + world_size - 1) // world_size
            self.seeds = self.seeds[rank * partition_size:(rank + 1) *
                                    partition_size]

        if self.shuffle:
            perm = torch.randperm(self.seeds.numel(), device=self.seeds.device)
            self.seeds = self.seeds[perm]

        if self.drop_last:
            self.len = self.seeds.numel() // self.batchsize
        else:
            self.len = (self.seeds.numel() + self.batchsize -
                        1) // self.batchsize
        self.curr = 0
        self.seeds = self.seeds.cuda()

    def __len__(self):
        return self.len

    def __iter__(self):
        if self.shuffle:
            perm = torch.randperm(self.seeds.numel(), device=self.seeds.device)
            self.seeds = self.seeds[perm]

        self.curr = 0
        return self

    def update_params(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

        if "seeds" in kwargs or "use_ddp" in kwargs:
            print("Updating seeds")
            if self.use_ddp:
                world_size = dist.get_world_size()
                rank = dist.get_rank()
                numel = self.seeds.numel()
                partition_size = (numel + world_size - 1) // world_size
                self.seeds = self.seeds[rank * partition_size:(rank + 1) *
                                        partition_size]
            self.seeds = self.seeds.cuda()

            if self.drop_last:
                self.len = self.seeds.numel() // self.batchsize
            else:
                self.len = (self.seeds.numel() + self.batchsize -
                            1) // self.batchsize

    def __next__(self):
        if self.curr >= self.len:
            raise StopIteration

        seeds = self.seeds[self.curr * self.batchsize:(self.curr + 1) *
                           self.batchsize]
        self.curr += 1

        seeds_feature = self.compression_manager.get_seeds_data(seeds)

        # begin sampling
        output_nodes = seeds
        result = []
        for num_pick in reversed(self.num_picks):
            if getattr(self, 'has_cache', None):
                sub_indptr, sub_indices = capi.csr_cache_sampling(
                    self.indptr, self.indices, self.gpu_indptr,
                    self.gpu_indices, self.hashmap, seeds, num_pick,
                    self.replace)
            else:
                sub_indptr, sub_indices = capi.csr_sampling(
                    self.indptr, self.indices, seeds, num_pick, self.replace)
            unique_tensor, (_, relabel_indices) = capi.tensor_relabel(
                [seeds, sub_indices])
            block = create_block_from_csc(sub_indptr, relabel_indices,
                                          torch.Tensor(),
                                          unique_tensor.numel(), seeds.numel())
            result.insert(0, block)
            seeds = unique_tensor

        intput_nodes = seeds

        all_features = self.compression_manager.get_all_data(intput_nodes)

        seeds_feature = self.compression_manager.reorganize_seeds_data(
            seeds_feature, all_features, result[0].number_of_dst_nodes())

        return intput_nodes, output_nodes, result, (seeds_feature,
                                                    all_features)

    def total_create_cache(self, cache_capacity):

        if dist.get_rank() == 0:
            print("max cache_capacity", cache_capacity / 1024 / 1024 / 1024)

        if dist.get_rank() == 0:
            print("Creating cache")
        sampling_hotness, feature_hotness, seeds_hotness = self.presampling()

        samp_full_size, samp_item_size = self.compute_weight_size(
            sampling_hotness)
        feat_full_size, feat_item_size = self.compression_manager.data.compute_weight_size(
            feature_hotness)
        seeds_full_size, seeds_item_size = self.compression_manager.seeds_data.compute_weight_size(
            seeds_hotness) if self.compression_manager.is_two_level else (0, 0)

        samp_capacity, feat_capacity, seeds_capacity = allocate(
            cache_capacity, [samp_item_size, feat_item_size, seeds_item_size],
            [samp_full_size, feat_full_size, seeds_full_size])

        self._create_cache(samp_capacity, sampling_hotness)
        self.compression_manager.data.create_cache(feat_capacity,
                                                   feature_hotness)

        if self.compression_manager.is_two_level:
            self.compression_manager.seeds_data.create_cache(
                seeds_capacity, seeds_hotness)

            if dist.get_rank() == 0:
                print("Seeds: {:6.3f} GB, Feats: {:6.3f} GB, samp: {:6.3f} GB".
                      format(seeds_capacity / 1024 / 1024 / 1024,
                             feat_capacity / 1024 / 1024 / 1024,
                             samp_capacity / 1024 / 1024 / 1024))

        else:
            if dist.get_rank() == 0:
                print("Feats: {:6.3f} GB, samp: {:6.3f} GB".format(
                    feat_capacity / 1024 / 1024 / 1024,
                    samp_capacity / 1024 / 1024 / 1024))

    def _create_cache(self, cache_capacity, hotness):
        if cache_capacity <= 0:
            return

        # test wheather full cache
        full_size = self.indptr.nbytes + self.indices.nbytes

        if full_size <= cache_capacity:
            self.indices = self.indices.cuda()
            self.indptr = self.indptr.cuda()
            print("Cache Ratio for GPU sampling 100.00%, size: {:.3f}".format(
                full_size / 1024 / 1024 / 1024))
            return

        degress = self.indptr[1:] - self.indptr[:-1]
        _, cache_candidates = torch.sort(hotness, descending=True)

        # compute size
        size = degress[cache_candidates] * self.indices.element_size(
        ) + self.indptr.element_size(
        ) + 1.25 * 2 * 4  # int type for key and value
        prefix_sum_size = torch.cumsum(size, dim=0)
        cache_size = torch.searchsorted(prefix_sum_size, cache_capacity).item()
        cache_candidates = cache_candidates[:cache_size].cuda()

        # binary search
        if cache_candidates.numel() > 0:
            self.gpu_indptr, self.gpu_indices = capi.create_subcsr(
                self.indptr, self.indices, cache_candidates)
            self.hashmap = capi.CUCOStaticHashmap(
                cache_candidates.int(),
                torch.arange(cache_candidates.numel(),
                             device='cuda',
                             dtype=torch.int32), 0.8)
            self.has_cache = True
            print("Cache Ratio for GPU sampling: {:.2f}%, size: {:.3f} GB".
                  format(prefix_sum_size[cache_size].item() / full_size * 100,
                         cache_capacity / 1024 / 1024 / 1024))
            print("create cache success")

    def presampling(self, device='cuda'):
        seeds_hotness = torch.zeros(self.indptr.numel() - 1, device=device)
        sampling_hotness = torch.zeros(self.indptr.numel() - 1, device=device)
        feature_hotness = torch.zeros(self.indptr.numel() - 1, device=device)

        seeds_hotness[self.seeds.to(device)] += 1

        for i in range(self.len):
            seeds = self.seeds[i * self.batchsize:(i + 1) * self.batchsize]

            for num_pick in reversed(self.num_picks):
                sampling_hotness[seeds.to(device)] += 1

                sub_indptr, sub_indices = capi.csr_sampling(
                    self.indptr, self.indices, seeds, num_pick, self.replace)
                unique_tensor, (_, relabel_indices) = capi.tensor_relabel(
                    [seeds, sub_indices])
                seeds = unique_tensor

            feature_hotness[unique_tensor.to(device)] += 1

        if self.use_ddp:
            sampling_hotness = sampling_hotness.cuda()
            feature_hotness = feature_hotness.cuda()
            seeds_hotness = seeds_hotness.cuda()

            torch.distributed.all_reduce(sampling_hotness)
            torch.distributed.all_reduce(feature_hotness)
            torch.distributed.all_reduce(seeds_hotness)

        return sampling_hotness.cpu(), feature_hotness.cpu(), seeds_hotness.cpu()

    def compute_weight_size(self, hotness):
        full_size = self.indptr.nbytes + self.indices.nbytes
        degree = (self.indptr[1:] - self.indptr[:-1]).clip(
            max=self.num_picks[0] * 1.3).float().cpu()
        item_size = (degree[hotness > 0] * self.indices.element_size() +
                     self.indptr.element_size()) * hotness[hotness > 0]
        item_size = item_size.mean().item()
        return full_size, item_size

        # full_size = self.indptr.nbytes + self.indices.nbytes
        # avg_degree = self.indices.shape[0] / (self.indptr.shape[0] - 1)
        # item_size = (avg_degree * self.indices.element_size() +
        #              self.indptr.element_size()) * torch.mean(
        #                  hotness[hotness > 0]).item()
        # return full_size, item_size
