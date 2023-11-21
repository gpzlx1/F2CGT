import dgl
import torch as th


def load_reddit(self_loop=True):
    from dgl.data import RedditDataset

    # load reddit data
    data = RedditDataset(self_loop=self_loop)
    g = data[0]
    g.ndata["features"] = g.ndata.pop("feat")
    g.ndata["labels"] = g.ndata.pop("label")
    return g, data.num_classes


def load_ogb(name, root="dataset"):
    from ogb.nodeproppred import DglNodePropPredDataset

    print("load", name)
    data = DglNodePropPredDataset(name=name, root=root)
    print("finish loading", name)
    splitted_idx = data.get_idx_split()
    graph, labels = data[0]
    labels = labels[:, 0]
    labels[th.isnan(labels)] = 0

    graph.ndata["features"] = graph.ndata.pop("feat")
    graph.ndata["labels"] = labels
    in_feats = graph.ndata["features"].shape[1]
    num_labels = len(th.unique(labels[th.logical_not(th.isnan(labels))]))

    # Find the node IDs in the training, validation, and test set.
    train_nid, val_nid, test_nid = (
        splitted_idx["train"],
        splitted_idx["valid"],
        splitted_idx["test"],
    )
    train_mask = th.zeros((graph.num_nodes(), ), dtype=th.bool)
    train_mask[train_nid] = True
    val_mask = th.zeros((graph.num_nodes(), ), dtype=th.bool)
    val_mask[val_nid] = True
    test_mask = th.zeros((graph.num_nodes(), ), dtype=th.bool)
    test_mask[test_nid] = True
    graph.ndata["train_mask"] = train_mask
    graph.ndata["val_mask"] = val_mask
    graph.ndata["test_mask"] = test_mask
    print("finish constructing", name)
    return graph, num_labels


def to_bidirected_with_reverse_mapping(g):
    """Makes a graph bidirectional, and returns a mapping array ``mapping`` where ``mapping[i]``
    is the reverse edge of edge ID ``i``. Does not work with graphs that have self-loops.
    """
    g_simple, mapping = dgl.to_simple(dgl.add_reverse_edges(g),
                                      return_counts="count",
                                      writeback_mapping=True)
    c = g_simple.edata["count"]
    num_edges = g.num_edges()
    mapping_offset = th.zeros(g_simple.num_edges() + 1, dtype=g_simple.idtype)
    mapping_offset[1:] = c.cumsum(0)
    idx = mapping.argsort()
    idx_uniq = idx[mapping_offset[:-1]]
    reverse_idx = th.where(idx_uniq >= num_edges, idx_uniq - num_edges,
                           idx_uniq + num_edges)
    reverse_mapping = mapping[reverse_idx]
    # sanity check
    src1, dst1 = g_simple.edges()
    src2, dst2 = g_simple.find_edges(reverse_mapping)
    assert th.equal(src1, dst2)
    assert th.equal(src2, dst1)
    return g_simple, reverse_mapping


def load_ogb_link_pred(name, root="dataset"):
    from ogb.linkproppred import DglLinkPropPredDataset

    print("load", name)
    data = DglLinkPropPredDataset(name=name, root=root)
    print("finish loading", name)
    g = data[0]
    g, reverse_eids = to_bidirected_with_reverse_mapping(g)
    g.ndata["features"] = g.ndata.pop("feat").float()
    print("finish constructing", name)
    return g, reverse_eids


def inductive_split(g):
    """Split the graph into training graph, validation graph, and test graph by training
    and validation masks.  Suitable for inductive models."""
    train_g = g.subgraph(g.ndata["train_mask"])
    val_g = g.subgraph(g.ndata["train_mask"] | g.ndata["val_mask"])
    test_g = g
    return train_g, val_g, test_g
