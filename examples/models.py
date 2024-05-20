import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import dgl.nn.pytorch as dglnn
import tqdm
from conv import MySAGEConv


class SAGE(nn.Module):

    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 cm=None):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        for i in range(0, n_layers):
            in_dim = in_feats if i == 0 else n_hidden
            out_dim = n_classes if i == n_layers - 1 else n_hidden
            if cm is not None and i == 0:
                self.layers.append(MySAGEConv(in_dim, out_dim, "mean", cm))
            else:
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
                 activation=F.elu,
                 dropout=0.5,
                 cm=None):
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
            if i == 0:
                if cm is not None:
                    self.layers.append(
                        MySAGEConv(in_dim, out_dim * n_heads[i], "mean", cm))
                else:
                    self.layers.append(
                        dglnn.SAGEConv(in_dim, out_dim * n_heads[i], "mean"))
            else:
                self.layers.append(
                    dglnn.GATConv(in_dim,
                                  out_dim,
                                  n_heads[i],
                                  residual=True,
                                  attn_drop=0.2,
                                  feat_drop=0.2,
                                  allow_zero_in_degree=True))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for i, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if i == self.n_layers - 1:
                h = h.mean(1)
            else:
                h = self.activation(h)
                h = self.dropout(h)
                h = h.flatten(1)
        return h


def nodewise_inference(model, dataloader, labels, device="cuda"):
    model.eval()
    with torch.no_grad():
        acc = 0
        length = 0
        for input_nodes, output_nodes, blocks, features in tqdm.tqdm(
                dataloader, disable=dist.get_rank() != 0):
            input_nodes = input_nodes.to(device)
            output_nodes = output_nodes.to(device)
            blocks = [block.to(device) for block in blocks]
            batch_inputs = features
            pred = model(blocks, batch_inputs).cpu()
            output_labels = labels[output_nodes.cpu()]
            acc += (torch.argmax(pred, dim=1) == output_labels).float().sum()
            length += output_nodes.numel()
        return acc / length


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
