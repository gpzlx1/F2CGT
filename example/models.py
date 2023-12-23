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

    def inference(self, g, feature, batch_size):
        """
        Conduct layer-wise inference to get all the node embeddings.
        Only support signle trainer now.
        """
        # full neighbor sampler
        sampler = bifeat.cache.StructureCacheServer(g["indptr"],
                                                    g["indices"], [0],
                                                    pin_memory=False)
        num_nodes = g["indptr"].shape[0] - 1
        local_nodes = torch.arange(0, num_nodes)
        dataloader = bifeat.dataloading.SeedGenerator(local_nodes,
                                                      batch_size,
                                                      shuffle=False)

        for l, layer in enumerate(self.layers):
            y = torch.empty(num_nodes,
                            self.n_hidden if l != len(self.layers) -
                            1 else self.n_classes,
                            dtype=torch.float32)
            for nodes in tqdm.tqdm(dataloader):
                src_nodes, dst_nodes, blocks = sampler.sample_neighbors(nodes)
                if l == 0:
                    x = feature[src_nodes]
                else:
                    x = feature[src_nodes.cpu()].cuda()
                h = layer(blocks[0], x)
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)
                y[dst_nodes] = h.to("cpu")
            feature = y
        return y


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

    def inference(self, g, feature, batch_size):
        """
        Conduct layer-wise inference to get all the node embeddings.
        Only support signle trainer now.
        """
        # full neighbor sampler
        sampler = bifeat.cache.StructureCacheServer(g["indptr"],
                                                    g["indices"], [0],
                                                    pin_memory=False)
        num_nodes = g["indptr"].shape[0] - 1
        local_nodes = torch.arange(0, num_nodes)
        dataloader = bifeat.dataloading.SeedGenerator(local_nodes,
                                                      batch_size,
                                                      shuffle=False)

        for l, layer in enumerate(self.layers):
            if l == len(self.layers) - 1:
                y = torch.empty(num_nodes,
                                self.n_classes * self.n_heads[l],
                                dtype=torch.float32)
            else:
                y = torch.empty(num_nodes,
                                self.n_hidden * self.n_heads[l],
                                dtype=torch.float32)
            for nodes in tqdm.tqdm(dataloader):
                src_nodes, dst_nodes, blocks = sampler.sample_neighbors(nodes)
                if l == 0:
                    x = feature[src_nodes]
                else:
                    x = feature[src_nodes.cpu()].cuda()
                h = layer(blocks[0], x)
                if l == self.n_layers - 1:
                    h = h.mean(1)
                else:
                    h = h.flatten(1)
                y[dst_nodes] = h.to("cpu")
            feature = y
        return y


def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    labels = labels.long()
    return (torch.argmax(pred, dim=1) == labels).float().sum() / len(pred)


def evaluate(model, g, feature_loader, labels, val_nid, test_nid, batch_size):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    feature_loader : The feature server
    labels : The labels of all the nodes.
    val_nid : the node Ids for validation.
    batch_size : Number of nodes to compute at the same time.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with torch.no_grad():
        pred = model.inference(g, feature_loader, batch_size)
    model.train()
    return compute_acc(pred[val_nid],
                       labels[val_nid]), compute_acc(pred[test_nid],
                                                     labels[test_nid])
