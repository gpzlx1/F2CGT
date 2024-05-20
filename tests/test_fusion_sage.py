import torch
from torch import nn
from torch.nn import functional as F
from dgl.utils import check_eq_shape, expand_as_pair
import dgl.function as fn


class FusionSAGEConv(nn.Module):

    def __init__(
        self,
        in_feats,
        out_feats,
        aggregator_type,
        feat_drop=0.0,
        bias=True,
        norm=None,
        activation=None,
    ):
        super(FusionSAGEConv, self).__init__()
        valid_aggre_types = {"mean"}
        if aggregator_type not in valid_aggre_types:
            raise ValueError(
                "For fusion SAGEConv, aggregator_type must be'mean'.")

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation

        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=False)
        self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def forward(self, graph, feat):
        with graph.local_scope():
            if isinstance(feat, tuple):
                raise ValueError("FusionSAGEConv does not support multi-head.")
            else:
                feat_src = feat_dst = self.feat_drop(feat)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]

            msg_fn = fn.copy_u("h", "m")

            h_self = feat_dst

            # Handle the case of graphs without edges
            if graph.num_edges() == 0:
                graph.dstdata["neigh"] = torch.zeros(
                    feat_dst.shape[0], self._in_src_feats).to(feat_dst)

            # Determine whether to apply linear transformation before message passing A(XW)
            lin_before_mp = self._in_src_feats > self._out_feats
            print(lin_before_mp)

            # Message Passing
            if self._aggre_type == "mean":
                graph.srcdata["h"] = (self.fc_neigh(feat_src)
                                      if lin_before_mp else feat_src)
                graph.update_all(msg_fn, fn.mean("m", "neigh"))
                h_neigh = graph.dstdata["neigh"]
                if not lin_before_mp:
                    h_neigh = self.fc_neigh(h_neigh)
            else:
                raise KeyError("Aggregator type {} not recognized.".format(
                    self._aggre_type))

            # GraphSAGE GCN does not require fc_self.
            rst = self.fc_self(h_self) + h_neigh

            # activation
            if self.activation is not None:
                rst = self.activation(rst)
            # normalization
            if self.norm is not None:
                rst = self.norm(rst)
            return rst


if __name__ == "__main__":
    import dgl

    # generate dgl block
    indptr = torch.tensor([0, 1, 3, 6])
    indices = torch.tensor([0, 1, 1, 2, 2, 3])
    block = dgl.create_block(('csc', (indptr, indices, torch.Tensor())),
                             num_src_nodes=4,
                             num_dst_nodes=3)

    print(block)

    block = block.to('cuda')
    feature = torch.rand(4, 10).cuda()
    dgl_conv = dgl.nn.SAGEConv(10, 5, 'mean', feat_drop=0,
                               activation=F.relu).cuda()
    res1 = dgl_conv(block, feature)
    print(res1)

    my_conv = FusionSAGEConv(10, 5, 'mean', feat_drop=0,
                             activation=F.relu).cuda()
    my_conv.fc_neigh.weight = dgl_conv.fc_neigh.weight
    my_conv.fc_neigh.bias = dgl_conv.fc_neigh.bias
    my_conv.fc_self.weight = dgl_conv.fc_self.weight
    my_conv.fc_self.bias = dgl_conv.fc_self.bias
    res2 = my_conv(block, feature)
    print(res2)

    assert torch.equal(res1, res2)
