import torch
import dgl
import BiFeatLib as capi
import torch.distributed as dist
import os
from ..shm import ShmManager, dtype_sizeof
from ..cache import StructureCacheServer
from ..dataloading import SeedGenerator
from .utils import sq_compress, vq_compress


class CompressionManager(object):

    def __init__(self,
                 methods,
                 configs,
                 cache_path,
                 shm_manager: ShmManager,
                 sample_sizes=[100000, 100000],
                 compress_batch_sizes=[1000000, 1000000]):
        assert len(methods) == 2
        assert len(configs) == 2
        for method, config in zip(methods, configs):
            if method == 'sq':
                assert 'target_bits' in config
            elif method == 'vq':
                assert 'length' in config
                assert 'width' in config
            else:
                raise ValueError

        self.methods = methods
        self.configs = configs
        self.cache_path = cache_path
        self.shm_manager = shm_manager
        self.sample_sizes = sample_sizes
        self.compress_batch_sizes = compress_batch_sizes

    def register(self,
                 indptr,
                 indices,
                 train_seeds,
                 labels,
                 features,
                 core_idx,
                 valid_idx=None,
                 test_idx=None,
                 fake_feat_dim=None,
                 fake_feat_dtype=None):
        self.indptr = indptr
        self.indices = indices
        self.train_seeds = train_seeds
        self.labels = labels
        self.features = features
        self.valid_idx = valid_idx
        self.test_idx = test_idx
        self.core_idx = core_idx

        if self.features is None:
            assert fake_feat_dim is not None
            assert fake_feat_dtype is not None
            self.original_size = (
                indptr.shape[0] -
                1) * fake_feat_dim * dtype_sizeof(fake_feat_dtype)
            self.fake_feat_dim = fake_feat_dim
        else:
            self.original_size = self.features.numel(
            ) * self.features.element_size()

        self.original_graph_size = 0
        self.original_graph_size += indptr.numel() * indptr.element_size()
        self.original_graph_size += indices.numel() * indices.element_size()
        self.original_graph_size += train_seeds.numel(
        ) * train_seeds.element_size()
        self.original_graph_size += labels.numel() * labels.element_size()
        self.original_graph_size += self.original_size
        if valid_idx is not None:
            self.original_graph_size += valid_idx.numel(
            ) * valid_idx.element_size()
        if test_idx is not None:
            self.original_graph_size += test_idx.numel(
            ) * test_idx.element_size()

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

        if rank == 0:
            self.train_seeds[:] = self.train_seeds[torch.randperm(
                self.train_seeds.shape[0])]
        dist.barrier()
        print(self.train_seeds)

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

            self.hotness[self.core_idx] += max_hot

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

            print(torch.max(self.adj_hotness[self.train_seeds]))
            print(torch.min(self.adj_hotness[self.train_seeds]))

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
            self.core_idx[:] = self.src2dst[self.core_idx]

            print(torch.max(self.adj_hotness[self.train_seeds]))
            print(torch.min(self.adj_hotness[self.train_seeds]))

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
        features_size = self.indptr.shape[0] - 1
        world_size = dist.get_world_size() - 1
        rank = dist.get_rank()

        if rank != world_size:
            each_gpu_size = (features_size + world_size - 1) // world_size
            begin = rank * each_gpu_size
            end = min((rank + 1) * each_gpu_size, features_size)

            if self.methods[1] == 'sq':
                if self.features is not None:
                    compressed_feature, codebook = sq_compress(
                        self.features[self.dst2src[begin:end]],
                        self.configs[1]['target_bits'], 'cuda',
                        self.sample_sizes[1], self.compress_batch_sizes[1])
                else:
                    compressed_feature, codebook = sq_compress(
                        None,
                        self.configs[1]['target_bits'],
                        'cuda',
                        self.sample_sizes[1],
                        self.compress_batch_sizes[1],
                        fake_feat_items=end - begin,
                        fake_feat_dim=self.fake_feat_dim)
            elif self.methods[1] == 'vq':
                if self.features is not None:
                    compressed_feature, codebook = vq_compress(
                        self.features[self.dst2src[begin:end]],
                        self.configs[1]['width'], self.configs[1]['length'],
                        'cuda', self.sample_sizes[1],
                        self.compress_batch_sizes[1])
                else:
                    compressed_feature, codebook = vq_compress(
                        None,
                        self.configs[1]['width'],
                        self.configs[1]['length'],
                        'cuda',
                        self.sample_sizes[1],
                        self.compress_batch_sizes[1],
                        fake_feat_items=end - begin,
                        fake_feat_dim=self.fake_feat_dim)
            else:
                raise ValueError
        else:
            if self.methods[0] == 'sq':
                if self.features is not None:
                    self.compressed_core_feature, self.core_codebook = sq_compress(
                        self.features[self.core_idx],
                        self.configs[0]['target_bits'], 'cuda',
                        self.sample_sizes[0], self.compress_batch_sizes[0])
                else:
                    self.compressed_core_feature, self.core_codebook = sq_compress(
                        None,
                        self.configs[0]['target_bits'],
                        'cuda',
                        self.sample_sizes[0],
                        self.compress_batch_sizes[0],
                        fake_feat_items=self.core_idx.shape[0],
                        fake_feat_dim=self.fake_feat_dim)
            elif self.methods[0] == 'vq':
                if self.features is not None:
                    self.compressed_core_feature, self.core_codebook = vq_compress(
                        self.features[self.core_idx], self.configs[0]['width'],
                        self.configs[0]['length'], 'cuda',
                        self.sample_sizes[0], self.compress_batch_sizes[0])
                else:
                    self.compressed_core_feature, self.core_codebook = vq_compress(
                        None,
                        self.configs[0]['width'],
                        self.configs[0]['length'],
                        'cuda',
                        self.sample_sizes[0],
                        self.compress_batch_sizes[0],
                        fake_feat_items=self.core_idx.shape[0],
                        fake_feat_dim=self.fake_feat_dim)
            else:
                raise ValueError
            self.core_codebook = torch.unsqueeze(self.core_codebook, 0)
            compressed_feature, codebook = None, None

        root = world_size
        if rank == root:
            output_codebook = [None for _ in range(world_size + 1)]
            output_feature = [None for _ in range(world_size + 1)]
        dist.gather_object(codebook, output_codebook if rank == root else None,
                           root)
        dist.gather_object(compressed_feature,
                           output_feature if rank == root else None, root)
        if rank == root:
            self.codebooks = torch.cat(
                [value.unsqueeze(0) for value in output_codebook[:-1]], dim=0)
            self.compressed_features = torch.cat(output_feature[:-1], dim=0)

            self.compressed_size = self.compressed_features.numel(
            ) * self.compressed_features.element_size()
            print(
                "compressed size: {:.3f} GB, compression ratio: {:.1f}".format(
                    self.compressed_size / 1024 / 1024 / 1024,
                    self.original_size / self.compressed_size))

            self.core_compressed_size = self.compressed_core_feature.numel(
            ) * self.compressed_core_feature.element_size()
            self.core_original_size = self.original_size / features_size * self.core_idx.shape[
                0]
            print("core compressed size: {:.3f} GB, compression ratio: {:.1f}".
                  format(self.core_compressed_size / 1024 / 1024 / 1024,
                         self.core_original_size / self.core_compressed_size))

    def save_data(self):
        if dist.get_rank() == dist.get_world_size() - 1:
            torch.save(self.compressed_features,
                       os.path.join(self.cache_path, "compressed_features.pt"))
            torch.save(self.codebooks,
                       os.path.join(self.cache_path, "codebooks.pt"))
            torch.save(
                self.compressed_core_feature,
                os.path.join(self.cache_path, "compressed_core_features.pt"))
            torch.save(self.core_codebook,
                       os.path.join(self.cache_path, "core_codebooks.pt"))
            self.compressed_graph_size = 0
            self.compressed_graph_size += self.compressed_features.numel(
            ) * self.compressed_features.element_size()
            self.compressed_graph_size += self.codebooks.numel(
            ) * self.codebooks.element_size()
            self.compressed_graph_size += self.labels.numel(
            ) * self.labels.element_size()
            self.compressed_graph_size += self.indptr.numel(
            ) * self.indptr.element_size()
            self.compressed_graph_size += self.indices.numel(
            ) * self.indices.element_size()
            self.compressed_graph_size += self.train_seeds.numel(
            ) * self.train_seeds.element_size()
            self.compressed_graph_size += self.adj_hotness.numel(
            ) * self.adj_hotness.element_size()
            self.compressed_graph_size += self.feat_hotness.numel(
            ) * self.feat_hotness.element_size()
            self.compressed_graph_size += self.compressed_core_feature.numel(
            ) * self.compressed_core_feature.element_size()
            self.compressed_graph_size += self.core_codebook.numel(
            ) * self.core_codebook.element_size()
            self.compressed_graph_size += self.core_idx.numel(
            ) * self.core_idx.element_size()
            if self.valid_idx is not None:
                self.compressed_graph_size += self.valid_idx.numel(
                ) * self.valid_idx.element_size()
            if self.test_idx is not None:
                self.compressed_graph_size += self.test_idx.numel(
                ) * self.test_idx.element_size()

            print("Results saved to {}".format(self.cache_path))
            print(
                "Original graph size {:.3f} GB, compressed graph size {:.3f} GB"
                .format(self.original_graph_size / 1024 / 1024 / 1024,
                        self.compressed_graph_size / 1024 / 1024 / 1024))

        if self.shm_manager._is_chief:
            print(torch.max(self.adj_hotness[self.train_seeds]))
            print(torch.min(self.adj_hotness[self.train_seeds]))

            metadata = self.shm_manager.graph_meta_data
            metadata["methods"] = self.methods
            torch.save(self.shm_manager.graph_meta_data,
                       os.path.join(self.cache_path, "metadata.pt"))

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

            torch.save(self.core_idx,
                       os.path.join(self.cache_path, "core_idx.pt"))

            if self.valid_idx is not None:
                torch.save(self.valid_idx,
                           os.path.join(self.cache_path, "valid_idx.pt"))
            if self.test_idx is not None:
                torch.save(self.test_idx,
                           os.path.join(self.cache_path, "test_idx.pt"))

        dist.barrier(self.shm_manager._local_group)
