import torch
import dgl
import BiFeatLib as capi


class CompressionManager(object):

    def __init__(self, ratios, methods, configs, cache_path):
        self.ratios = ratios
        self.methods = methods
        self.configs = configs
        self.cache_path = cache_path

    def register(self, indptr, indices, features, train_seeds):
        self.indptr = indptr
        self.indices = indices
        self.features = features
        self.train_seeds = train_seeds

        self.original_size = self.features.numel(
        ) * self.features.element_size()

    def presampling(self, fanouts):
        self.hotness = torch.zeros(self.indptr.numel() - 1,
                                   device='cuda',
                                   dtype=torch.float32)
        sampler = dgl.dataloading.NeighborSampler(fanouts)

        g = dgl.graph(('csc', (self.indptr, self.indices,
                               torch.empty(0, dtype=torch.int64))))

        dataloader = dgl.dataloading.DataLoader(
            g,
            self.train_seeds,
            sampler,
            batch_size=512,
            device='cuda',
            num_workers=0,
            use_uva=True,
        )
        for input_nodes, output_nodes, blocks in dataloader:
            for block in blocks:
                self.hotness[block.dstdata[dgl.NID]] += 1
            self.hotness[output_nodes] += 1

        self.hotness[self.train_seeds] += 1000

    def graph_reorder(self):
        assert self.hotness is not None

        _, dst2src = torch.sort(self.hotness, descending=True)
        self.dst2src = dst2src.cpu()

        self.src2dst = torch.empty_like(self.dst2src)
        self.src2dst[self.dst2src] = torch.arange(self.dst2src.numel())

        new_indptr = torch.zeros_like(self.indptr)
        new_indices = torch.empty_like(self.indices)

        # create new_indptr
        degrees = self.indptr[1:] - self.indptr[:-1]
        new_indptr[1:] = degrees[self.dst2src].cumsum(dim=0)

        print(new_indptr)

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

        self.indptr = new_indptr
        self.indices = new_indices
        self.train_seeds = self.src2dst[self.train_seeds]
