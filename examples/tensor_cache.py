import torch
import shmtensor
import weakref
import torch.distributed as dist
from dataloader import allocate


def total_create_cache(cache_capacity, dataloader, featcache):

    if dist.get_rank() == 0:
        print("max cache_capacity", cache_capacity / 1024 / 1024 / 1024)

    if dist.get_rank() == 0:
        print("Creating cache")

    sampling_hotness, feature_hotness = dataloader.presampling()

    samp_full_size, samp_item_size = dataloader.compute_weight_size(
        sampling_hotness)
    feat_full_size, feat_item_size = featcache.compute_weight_size(
        feature_hotness)

    samp_capacity, feat_capacity = allocate(cache_capacity,
                                            [samp_item_size, feat_item_size],
                                            [samp_full_size, feat_full_size])

    dataloader.create_cache(samp_capacity, sampling_hotness)
    featcache.create_cache(feat_capacity, feature_hotness)

    if dist.get_rank() == 0:
        print("Feats: {:6.3f} GB, samp: {:6.3f} GB".format(
            feat_capacity / 1024 / 1024 / 1024,
            samp_capacity / 1024 / 1024 / 1024))


class TensorCache:

    def __init__(self, data):
        self.data = data

        self.gpu_data = None
        self.hashmap = None
        self.has_cache = False

        if self.data.device == torch.device('cpu'):
            shmtensor.capi.pin_memory(self.data)
            weakref.finalize(self, shmtensor.capi.unpin_memory, self.data)

    def __getitem__(self, index):
        '''
        index is a gpu tensor
        '''
        if self.has_cache:
            return shmtensor.capi.cache_fetch(self.data, self.gpu_data, index,
                                              self.hashmap)
        else:
            return shmtensor.capi.uva_fetch(self.data, index)

    def create_cache(self, cache_capacity, hotness):
        if cache_capacity <= 0:
            return

        # test wheather full cache
        full_size = self.data.nbytes

        if full_size <= cache_capacity:
            self.data = self.data.cuda()
            print("Cache Ratio for Feature: 100.00%, size: {:.3f} GB".format(
                full_size / 1024 / 1024 / 1024))
            return

        _, cache_candidates = torch.sort(hotness, descending=True)

        # compute size
        item_size = self.data.shape[1] * self.data.element_size(
        ) + 1.25 * 2 * 4  # int type for key and value
        cache_num = int(cache_capacity / item_size)
        cache_candidates = cache_candidates[:cache_num]

        if cache_candidates.numel() > 0:
            self.gpu_data = self.data[cache_candidates].cuda()
            cache_size = cache_num * item_size
            self.hashmap = shmtensor.capi.CUCOStaticHashmap(
                cache_candidates.int().cuda(),
                torch.arange(cache_candidates.numel(),
                             device='cuda',
                             dtype=torch.int32), 0.8)
            self.has_cache = True
            print("Cache Ratio for Feature: {:.2f}%, num: {}, size: {:.3f} GB".
                  format((cache_num * self.data.shape[1] *
                          self.data.element_size() / full_size) * 100,
                         cache_num, cache_size / 1024 / 1024 / 1024))
            print("create feature cache success")

    def clear_cache(self):
        self.has_cache = False
        del self.gpu_data
        del self.hashmap
        self.gpu_data = None
        self.hashmap = None

    def compute_weight_size(self, hotness):
        full_size = self.data.nbytes
        item_size = self.data.element_size() * self.data.shape[1] * torch.mean(
            hotness[hotness > 0]).item()
        return full_size, item_size
