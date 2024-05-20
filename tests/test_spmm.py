import torch
import dgl
import dgl.function as fn
import F2CGTLib as capi
import time


def dgl_spmm(block, features):

    block.srcdata["h"] = features
    msg_fn = fn.copy_u("h", "m")
    block.update_all(msg_fn, fn.sum("m", "neigh"))

    return block.dstdata["neigh"]


def test_capi_spmmcsr1():
    indptr = torch.tensor([0, 1, 3, 6]).long().cuda()
    indices = torch.tensor([0, 1, 1, 2, 2, 0]).long().cuda()
    features = torch.randn(6, 100).float().cuda()

    block = dgl.create_block(('csc', (indptr, indices, torch.Tensor())),
                             num_dst_nodes=indptr.numel() - 1,
                             num_src_nodes=features.shape[0])

    out1 = dgl_spmm(block, features)
    out2 = capi.spmm_csr(features, indptr, indices)

    assert torch.equal(out1, out2)


def test_capi_spmmcsr2():
    from dgl.data import RedditDataset
    graph = RedditDataset(self_loop=True)[0]

    sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 15, 15])
    dataloader = dgl.dataloading.DataLoader(
        graph,
        torch.arange(graph.number_of_nodes()),
        sampler,
        batch_size=1024,
        shuffle=True,
        drop_last=False,
    )
    for input_nodes, output_nodes, blocks in dataloader:
        block = blocks[0]
        break

    print(block)

    indptr, indices, _ = block.adj_tensors('csc')

    block = block.to('cuda')
    indptr = indptr.cuda()
    indices = indices.cuda()

    features = torch.randn(input_nodes.numel(), 128).float().cuda()
    features = (features * 10).int().float()
    out1 = dgl_spmm(block, features)
    out2 = capi.spmm_csr(features, indptr, indices)

    print(block.formats())

    for i in range(10):
        begin = time.time()
        out1 = dgl_spmm(block, features)
        torch.cuda.synchronize()
        print((time.time() - begin) * 1000)

    print()

    for i in range(10):
        begin = time.time()
        out2 = capi.spmm_csr(features, indptr, indices)
        torch.cuda.synchronize()
        print((time.time() - begin) * 1000)

    assert torch.equal(out1, out2)


if __name__ == "__main__":
    test_capi_spmmcsr1()
    test_capi_spmmcsr2()
