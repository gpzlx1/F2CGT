import torch
import torch.distributed as dist
import F2CGTLib
import shmtensor
from tensor_cache import TensorCache


class CompressionManager:

    def __init__(self, compress_data, fusion=False) -> None:
        self.meta = compress_data
        self.methods = self.meta['methods']
        self.feat_dim = self.meta['feat_dim']
        self.configs = self.meta['configs']  # (column_slice, target_bits)
        self.codebooks = [i.cuda() for i in self.meta['codebooks']]

        self.data = TensorCache(compress_data['compression_data'].tensor_)

        self.seeds_data = None
        if len(self.methods) >= 2:
            self.seeds_data = TensorCache(
                compress_data['seeds_compression_data'].tensor_)

            keys = compress_data['seeds'].int().cuda()
            values = torch.arange(keys.numel()).int().cuda()
            self.hashmap = shmtensor.capi.CUCOStaticHashmap(keys, values, 0.8)

        if dist.get_rank() == 0:
            print('CompressionManager initialized')
            print("methods: {}, configs: {}".format(self.methods,
                                                    self.configs))

        self.is_two_level = True if len(self.methods) >= 2 else False
        self.is_fusion = fusion

        self.seeds_task_stream = torch.cuda.Stream()
        self.seeds_task_event = torch.cuda.Event()

    def get_seeds_data(self, seeds_key):

        if self.is_two_level:
            remap_seeds_keys = self.hashmap.query(seeds_key)
            with torch.cuda.stream(self.seeds_task_stream):
                seeds_data = self.seeds_data[remap_seeds_keys]
                
                if self.methods[0] == 'vq':
                    seeds_feature = F2CGTLib.vq_decompress(
                        seeds_data, self.codebooks[0], self.feat_dim)
                else:
                    seeds_feature = F2CGTLib.sq_decompress(
                        seeds_data, self.codebooks[0], self.configs[0][1],
                        self.configs[0][0], self.feat_dim)
                self.seeds_task_event.record()
            return seeds_feature
        else:
            return None

    def get_all_data(self, keys):
        data = self.data[keys]
        if self.is_fusion:
            return data

        all_index = 1 if self.is_two_level else 0
        if self.methods[all_index] == 'vq':
            output = F2CGTLib.vq_decompress(data, self.codebooks[all_index],
                                            self.feat_dim)
        else:
            output = F2CGTLib.sq_decompress(data, self.codebooks[all_index],
                                            self.configs[all_index][1],
                                            self.configs[all_index][0],
                                            self.feat_dim)
        return output

    def reorganize_seeds_data(self, seeds_feature, all_feature, num_dst):
        self.seeds_task_event.wait()

        if not self.is_fusion:

            if seeds_feature is None:
                return None

            num_seeds = seeds_feature.shape[0]
            return torch.cat([seeds_feature, all_feature[num_seeds:num_dst]],
                             dim=0)

        else:
            if self.is_two_level:
                num_seeds = seeds_feature.shape[0]
                part_data = all_feature[num_seeds:num_dst]
            else:
                part_data = all_feature[:num_dst]

            all_index = 1 if self.is_two_level else 0
            if self.methods[all_index] == 'vq':
                part_feature = F2CGTLib.vq_decompress(
                    part_data, self.codebooks[all_index], self.feat_dim)
            else:
                part_feature = F2CGTLib.sq_decompress(
                    part_data, self.codebooks[all_index],
                    self.configs[all_index][1], self.configs[all_index][0],
                    self.feat_dim)

            if self.is_two_level:
                return torch.cat([seeds_feature, part_feature], dim=0)
            else:
                return part_feature
