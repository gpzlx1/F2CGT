import torch
import dgl
import BiFeatLib as capi
import torch.distributed as dist
import math
from ..shm import ShmManager
from .utils import sq_compress, vq_compress, sq_decompress, vq_decompress


class CompressionManager(object):

    def __init__(self, ratios, methods, configs, cache_path,
                 shm_manager: ShmManager):
        assert len(ratios) == len(methods)
        assert len(ratios) == len(configs)
        assert sum(ratios) == 1.0
        for method, config in zip(methods, configs):
            if method == 'sq':
                assert 'target_bits' in config
            elif method == 'vq':
                assert 'length' in config
                assert 'width' in config
            else:
                raise ValueError

        self.ratios = ratios
        self.methods = methods
        self.configs = configs
        self.cache_path = cache_path
        self.shm_manager = shm_manager

    def register(self, indptr, indices, train_seeds, labels, features):
        self.indptr = indptr
        self.indices = indices
        self.train_seeds = train_seeds
        self.labels = labels
        self.features = features

        self.original_size = self.features.numel(
        ) * self.features.element_size()

    def presampling(self, fanouts):
        self.hotness = torch.zeros(self.indptr.numel() - 1,
                                   device='cuda',
                                   dtype=torch.float32)
        sampler = dgl.dataloading.NeighborSampler(fanouts)

        g = dgl.graph(('csc', (self.indptr, self.indices,
                               torch.empty(0, dtype=torch.int64))))

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        part_size = (self.train_seeds.numel() + world_size - 1) // world_size
        start = rank * part_size
        end = (rank + 1) * part_size

        local_train_seeds = self.train_seeds[start:end]
        dataloader = dgl.dataloading.DataLoader(
            g,
            local_train_seeds,
            sampler,
            batch_size=512,
            device='cuda',
            num_workers=0,
            use_uva=True,
            shuffle=True,
        )
        for input_nodes, output_nodes, blocks in dataloader:
            for block in blocks:
                self.hotness[block.dstdata[dgl.NID]] += 1
            self.hotness[output_nodes] += 1

        dist.all_reduce(self.hotness, op=dist.ReduceOp.SUM)

    def graph_reorder(self):
        if self.shm_manager._is_chief:
            assert self.hotness is not None

            max_hot = self.hotness.max()
            self.hotness[self.train_seeds] += max_hot

            self.hotness, dst2src = torch.sort(self.hotness, descending=True)
            self.dst2src = dst2src.cpu()

            self.src2dst = torch.empty_like(self.dst2src)
            self.src2dst[self.dst2src] = torch.arange(self.dst2src.numel())

            # create
            new_indptr = torch.zeros_like(self.indptr)
            new_indices = torch.empty_like(self.indices)

            degrees = self.indptr[1:] - self.indptr[:-1]
            new_indptr[1:] = degrees[self.dst2src].cumsum(dim=0)

            # create new_indices
            # for i in range(self.indptr.numel() - 1):
            #    dst = i
            #    src = self.dst2src[dst].item()
            #    new_indices[new_indptr[dst].item():new_indptr[dst + 1].item(
            #    )] = self.indices[self.indptr[src].item():self.indptr[src +
            #                                                          1].item()]
            capi.omp_reorder_indices(self.indptr, self.indices, new_indptr,
                                     new_indices, self.dst2src)

            new_indices = self.src2dst[new_indices]

            self.indptr[:] = new_indptr
            self.indices[:] = new_indices
            self.train_seeds[:] = self.src2dst[self.train_seeds]
            self.labels[:] = self.labels[self.dst2src]

            # recovery hotness
            self.hotness[self.train_seeds] -= max_hot

            # for broadcast
            dst2src = self.dst2src.cpu()
            hotness = self.hotness.cpu()
            package = [dst2src, hotness]
        else:
            package = [None, None]

        dist.barrier(self.shm_manager._local_group)

        # brocast hotness and dst2src
        dist.broadcast_object_list(package,
                                   0,
                                   group=self.shm_manager._local_group)
        self.dst2src = package[0]
        self.hotness = package[1]

    def compress(self):
        num_train_idx = self.train_seeds.numel()
        num_parts = len(self.ratios)
        num_items = self.features.shape[0]

        part_size_list = []

        if num_train_idx > num_items * self.ratios[0]:
            part_size_list.append(num_train_idx)
            part_size_list.append(num_train_idx)

            new_ratios = self.ratios[1:]
            new_ratios = new_ratios / sum(new_ratios)

            for i in range(num_parts - 1):
                part_size = int(num_items * new_ratios[i])
                part_size_list.append(part_size)
            part_size_list[-1] = num_items - sum(part_size_list[:-1])

        else:
            for i in range(num_parts):
                part_size = int(num_items * self.ratios[i])
                part_size_list.append(part_size)
            part_size_list[-1] = num_items - sum(part_size_list[:-1])

        print(part_size_list)

        # begin compress
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        codebooks = []
        compressed_features = []
        for i in range(num_parts):
            features_size = part_size_list[i]
            each_gpu_size = (features_size + world_size - 1) // world_size
            begin = rank * each_gpu_size
            end = (rank + 1) * each_gpu_size

            for j in range(i):
                begin += part_size_list[j]
                end += part_size_list[j]

            if self.methods[i] == 'sq':
                codebook, compressed_feature = sq_compress(
                    self.features[self.dst2src[begin:end]],
                    self.configs[i]['target_bits'], 'cuda')
            elif self.methods[i] == 'vq':
                codebook, compressed_feature = vq_compress(
                    self.features[self.dst2src[begin:end]],
                    self.configs[i]['width'], self.configs[i]['length'],
                    'cuda')
            else:
                raise ValueError

            codebooks.append(codebook)
            compressed_features.append(compressed_feature)

        for i in codebooks:
            print(i.shape)

        for i in compressed_features:
            print(i.shape)
