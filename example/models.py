import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import dgl.nn.pytorch as dglnn
import bifeat
import tqdm


class SAGE(nn.Module):

    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation,
                 dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        for i in range(0, n_layers):
            in_dim = in_feats if i == 0 else n_hidden
            out_dim = n_classes if i == n_layers - 1 else n_hidden
            self.layers.append(dglnn.SAGEConv(in_dim, out_dim, "mean"))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for i, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if i != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h


class GAT(nn.Module):

    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 n_heads,
                 activation=F.relu,
                 feat_dropout=0.6,
                 attn_dropout=0.6):
        assert len(n_heads) == n_layers
        assert n_heads[-1] == 1

        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_heads = n_heads

        self.layers = nn.ModuleList()
        for i in range(0, n_layers):
            in_dim = in_feats if i == 0 else n_hidden * n_heads[i - 1]
            out_dim = n_classes if i == n_layers - 1 else n_hidden
            layer_activation = None if i == n_layers - 1 else activation
            self.layers.append(
                dglnn.GATConv(in_dim,
                              out_dim,
                              n_heads[i],
                              feat_drop=feat_dropout,
                              attn_drop=attn_dropout,
                              activation=layer_activation,
                              allow_zero_in_degree=True))

    def forward(self, blocks, x):
        h = x
        for i, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if i == self.n_layers - 1:
                h = h.mean(1)
            else:
                h = h.flatten(1)
        return h


def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    labels = labels.long()
    return (torch.argmax(pred, dim=1) == labels).float().sum() / len(pred)


def evaluate(model, g, target_nids, feature, fan_out, batch_size):
    model.eval()
    acc = 0
    num_iters = 0
    sampler = bifeat.cache.StructureCacheServer(g["indptr"],
                                                g["indices"],
                                                fan_out,
                                                pin_memory=False)
    dataloader = bifeat.dataloading.SeedGenerator(target_nids,
                                                  batch_size,
                                                  shuffle=True)
    for seed_nids in tqdm.tqdm(dataloader):
        frontier, seeds, blocks = sampler.sample_neighbors(seed_nids)
        batch_inputs = feature[frontier, seed_nids.shape[0]]
        batch_labels = g["labels"][seeds.cpu()].long().cuda()
        batch_pred = model(blocks, batch_inputs)
        acc += compute_acc(batch_pred, batch_labels).item()
        num_iters += 1
    acc = acc / num_iters
    return acc
