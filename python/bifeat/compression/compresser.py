import torch
import dgl
import BiFeatLib as capi
import torch.distributed as dist
from ..shm import ShmManager


class CompressionManager(object):

    def __init__(self, ratios, methods, configs, cache_path,
                 shm_manager: ShmManager):
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
