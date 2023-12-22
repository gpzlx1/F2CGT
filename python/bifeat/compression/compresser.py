import torch
import dgl
import BiFeatLib as capi
import torch.distributed as dist
import os
from ..shm import ShmManager
from ..cache import StructureCacheServer
from ..dataloading import SeedGenerator
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

    def register(self,
                 indptr,
                 indices,
                 train_seeds,
                 labels,
                 features,
                 valid_idx=None,
                 test_idx=None):
        self.indptr = indptr
        self.indices = indices
        self.train_seeds = train_seeds
        self.labels = labels
        self.features = features
        self.valid_idx = valid_idx
        self.test_idx = test_idx

        self.original_size = self.features.numel(
        ) * self.features.element_size()

    def presampling(self, fanouts, batch_size=512):
        self.adj_hotness = torch.zeros(self.indptr.numel() - 1,
                                       device='cuda',
                                       dtype=torch.float32)
        self.feat_hotness = torch.zeros(self.indptr.numel() - 1,
                                        device='cuda',
                                        dtype=torch.float32)

        sampler = StructureCacheServer(self.indptr,
                                       self.indices,
                                       fanouts,
                                       pin_memory=False)

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        part_size = (self.train_seeds.numel() + world_size - 1) // world_size
        start = rank * part_size
        end = (rank + 1) * part_size
        local_train_seeds = self.train_seeds[start:end]
        seeds_loader = SeedGenerator(local_train_seeds,
                                     batch_size,
                                     shuffle=True)

        for it, seeds in enumerate(seeds_loader):
            frontier, _, blocks = sampler.sample_neighbors(seeds)
            for block in blocks:
                self.adj_hotness[block.dstdata[dgl.NID]] += 1
            self.feat_hotness[frontier] += 1

        dist.all_reduce(self.adj_hotness, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.feat_hotness, op=dist.ReduceOp.SUM)
        self.hotness = self.adj_hotness + self.feat_hotness
        self.adj_hotness = self.adj_hotness.cpu()
        self.feat_hotness = self.feat_hotness.cpu()

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
            self.adj_hotness[:] = self.adj_hotness[self.dst2src]
            self.feat_hotness[:] = self.feat_hotness[self.dst2src]
            if self.valid_idx is not None:
                self.valid_idx[:] = self.src2dst[self.valid_idx]
            if self.test_idx is not None:
                self.test_idx[:] = self.src2dst[self.test_idx]

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

        # compute part size
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

        self.part_size_list = part_size_list

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
                compressed_feature, codebook = sq_compress(
                    self.features[self.dst2src[begin:end]],
                    self.configs[i]['target_bits'], 'cuda')
            elif self.methods[i] == 'vq':
                compressed_feature, codebook = vq_compress(
                    self.features[self.dst2src[begin:end]],
                    self.configs[i]['width'], self.configs[i]['length'],
                    'cuda')
            else:
                raise ValueError

            codebooks.append(codebook)
            compressed_features.append(compressed_feature)

        # root ranks gather results
        roots = [
            i for i in range(world_size)
            if i % dist.get_world_size(self.shm_manager._local_group) == 0
        ]

        for root in roots:
            if rank == root:
                self.codebooks = [None] * num_parts
                self.compressed_features = [None] * num_parts
            for i in range(num_parts):
                if rank == root:
                    output_codebooks = [None for _ in range(world_size)]
                    output_features = [None for _ in range(world_size)]
                dist.gather_object(codebooks[i],
                                   output_codebooks if rank == root else None,
                                   root)
                dist.gather_object(compressed_features[i],
                                   output_features if rank == root else None,
                                   root)
                if rank == root:
                    full_codebooks = torch.cat(
                        [value.unsqueeze(0) for value in output_codebooks],
                        dim=0)
                    self.codebooks[i] = full_codebooks
                    full_features = torch.cat(output_features, dim=0)
                    self.compressed_features[i] = full_features

        # compute compressed size and compression ratio
        if self.shm_manager._is_chief:
            self.compressed_size = 0
            for i in range(num_parts):
                self.compressed_size += self.compressed_features[i].numel(
                ) * self.compressed_features[i].element_size()

            print(
                "compressed size: {:.3f} GB, compression ratio: {:.1f}".format(
                    self.compressed_size / 1024 / 1024 / 1024,
                    self.original_size / self.compressed_size))

    def decompress_old(self, idx):
        output_tensor = torch.empty((idx.numel(), self.features.shape[1]),
                                    dtype=self.features.dtype,
                                    device='cuda')
        part_indices = torch.searchsorted(
            self.partition_range, idx, right=True) - 1
        local_part_indices = idx - self.partition_range[part_indices]
        local_codebook_indices = local_part_indices // self.chunk_size[
            part_indices]
        self.feat_dim = self.features.shape[1]

        for i in range(idx.numel()):
            part_index = part_indices[i]
            local_part_index = local_part_indices[i]
            local_codebook_index = local_codebook_indices[i]

            if self.methods[part_index] == 'sq':
                decompressed_feature = sq_decompress(
                    self.compressed_features[part_index][local_part_index],
                    self.feat_dim,
                    self.codebooks[part_index][local_codebook_index],
                )
            elif self.methods[part_index] == 'vq':
                decompressed_feature = vq_decompress(
                    self.compressed_features[part_index]
                    [local_part_index].unsqueeze(0), self.feat_dim,
                    self.codebooks[part_index][local_codebook_index])

            output_tensor[i] = decompressed_feature

        return output_tensor

    def decompress(self, idx):
        output_tensor = torch.empty((idx.numel(), self.features.shape[1]),
                                    dtype=self.features.dtype,
                                    device='cuda')
        part_indices = torch.searchsorted(
            self.partition_range, idx, right=True) - 1
        local_part_indices = idx - self.partition_range[part_indices]
        local_codebook_indices = local_part_indices // self.chunk_size[
            part_indices]
        self.feat_dim = self.features.shape[1]

        for i in range(len(self.part_size_list)):
            part_index = (part_indices == i).nonzero().flatten()
            if part_index.numel() > 0:
                local_part_index = local_part_indices[part_index]
                local_codebook_index = local_codebook_indices[part_index].cuda(
                )

                if self.methods[i] == 'sq':
                    decompressed_feature = capi._CAPI_sq_decompress(
                        local_codebook_index,
                        self.compressed_features[i][local_part_index].cuda(),
                        self.codebooks[i].cuda(), self.feat_dim)
                elif self.methods[i] == 'vq':
                    decompressed_feature = capi._CAPI_vq_decompress(
                        local_codebook_index,
                        self.compressed_features[i][local_part_index].cuda(),
                        self.codebooks[i].cuda(), self.feat_dim)

                output_tensor[part_index] = decompressed_feature

        return output_tensor

    def save_data(self):
        if self.shm_manager._is_chief:
            metadata = self.shm_manager.graph_meta_data
            metadata["part_size"] = self.part_size_list
            metadata["methods"] = self.methods
            torch.save(self.shm_manager.graph_meta_data,
                       os.path.join(self.cache_path, "metadata.pt"))
            torch.save(self.compressed_features,
                       os.path.join(self.cache_path, "compressed_features.pt"))
            torch.save(self.codebooks,
                       os.path.join(self.cache_path, "codebooks.pt"))
            torch.save(self.labels, os.path.join(self.cache_path, "labels.pt"))
            torch.save(self.indptr, os.path.join(self.cache_path, "indptr.pt"))
            torch.save(self.indices, os.path.join(self.cache_path,
                                                  "indices.pt"))
            torch.save(self.train_seeds,
                       os.path.join(self.cache_path, "train_idx.pt"))
            torch.save(self.adj_hotness,
                       os.path.join(self.cache_path, "adj_hotness.pt"))
            torch.save(self.feat_hotness,
                       os.path.join(self.cache_path, "feat_hotness.pt"))

            if self.valid_idx is not None:
                torch.save(self.valid_idx,
                           os.path.join(self.cache_path, "valid_idx.pt"))
            if self.test_idx is not None:
                torch.save(self.test_idx,
                           os.path.join(self.cache_path, "test_idx.pt"))
            print("Results saved to {}".format(self.cache_path))
