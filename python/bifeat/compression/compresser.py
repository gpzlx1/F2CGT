import torch
import dgl


class CompressionManager(object):

    def __init__(self, ratios, methods, configs, cache_path):
        self.ratios = ratios
        self.methods = methods
        self.configs = configs
        self.cache_path = cache_path

    def register(self, g, features, train_seeds):
        self.g = g
        self.features = features
        self.train_seeds = train_seeds

        self.original_size = self.features.numel(
        ) * self.features.element_size()

    def presampling(self, fanouts):
        self.hotness = torch.zeros(self.g.number_of_nodes(),
                                   device='cuda',
                                   dtype=torch.float16)
        sampler = dgl.dataloading.NeighborSampler(fanouts)
        dataloader = dgl.dataloading.DataLoader(
            self.g,
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
