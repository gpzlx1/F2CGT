import torch
from torch import nn
from torch.nn import functional as F
from dgl.utils import check_eq_shape, expand_as_pair
import dgl.function as fn
import F2CGTLib as capi


class MySAGEConv(nn.Module):

    def __init__(
        self,
        in_feats,
        out_feats,
        aggregator_type,
        compression_manager,
        bias=True,
        norm=None,
        activation=None,
    ):
        super(MySAGEConv, self).__init__()
        valid_aggre_types = {"mean"}
        if aggregator_type not in valid_aggre_types:
            raise ValueError(
                "For fusion SAGEConv, aggregator_type must be'mean'.")

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.activation = activation

        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=False)
        self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=bias)
        self.reset_parameters()

        self.sm = compression_manager

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def forward(self, graph, feat):
        with graph.local_scope():
            h_self, h_neigh = feat

            if h_self is None:
                h_self = h_neigh[:graph.number_of_dst_nodes()]

            indptr, indices, _ = graph.adj_tensors('csc')

            if not self.sm.is_fusion:
                h_neigh = capi.spmm_csr(h_neigh, indptr, indices)
            else:
                all_index = 1 if self.sm.is_two_level else 0

                if self.sm.methods[all_index] == 'vq':
                    h_neigh = capi.vq_decompress_spmm_csr(
                        h_neigh, self.sm.codebooks[all_index], indptr, indices,
                        self.sm.feat_dim)
                else:
                    h_neigh = capi.sq_decompress_spmm_csr(
                        h_neigh, self.sm.codebooks[all_index], indptr, indices,
                        self.sm.configs[all_index][1],
                        self.sm.configs[all_index][0], self.sm.feat_dim)

            h_neigh = self.fc_neigh(h_neigh)
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
    dgl_conv = dgl.nn.SAGEConv(10, 5, 'mean', feat_drop=0).cuda()
    res1 = dgl_conv(block, feature)
    print(res1)

    my_conv = MySAGEConv(10, 5, 'mean').cuda()
    my_conv.fc_neigh.weight = dgl_conv.fc_neigh.weight
    my_conv.fc_neigh.bias = dgl_conv.fc_neigh.bias
    my_conv.fc_self.weight = dgl_conv.fc_self.weight
    my_conv.fc_self.bias = dgl_conv.fc_self.bias
    res2 = my_conv(block, feature)
    print(res2)

    max_diff = (res1 - res2).abs().max().item()
    print(max_diff)
