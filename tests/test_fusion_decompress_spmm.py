import torch
import dgl
import dgl.function as fn
import F2CGTLib as capi
import time
from test_sq import sample, capi_sq_compress, capi_sq_decompress


def compress(data, num_class):
    kmeans = capi.KMeans(num_class, data.shape[1], "cosine")
    kmeans.fit(data)
    cluster_centers = kmeans.get_centers()
    labels = kmeans.predict(data)
    cluster_centers = torch.tensor(cluster_centers, device='cuda')
    labels = torch.tensor(labels, device='cuda').to(torch.uint8)
    return cluster_centers, labels


def test_fusion_vq_spmmcsr1():
    indptr = torch.tensor([0, 1, 3, 6]).long().cuda()
    indices = torch.tensor([0, 1, 1, 2, 2, 0]).long().cuda()
    features = torch.randn(6, 100).float().cuda()
    cluster_centers, labels = compress(features, 256)
    compress_data = labels.reshape(*labels.shape, 1).cuda()
    codebook = cluster_centers.reshape(-1, *cluster_centers.shape).cuda()

    decompress_features = capi.vq_decompress(compress_data, codebook, 100)
    out1 = capi.spmm_csr(decompress_features, indptr, indices)

    out2 = capi.vq_decompress_spmm_csr(compress_data, codebook, indptr,
                                       indices, 100)

    assert torch.equal(out1, out2)


def test_fusion_vq_spmmcsr2():
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
    cluster_centers, labels = compress(features, 256)
    compress_data = labels.reshape(*labels.shape, 1).cuda()
    codebook = cluster_centers.reshape(-1, *cluster_centers.shape).cuda()

    decompress_features = capi.vq_decompress(compress_data, codebook, 100)
    out1 = capi.spmm_csr(decompress_features, indptr, indices)
    out2 = capi.vq_decompress_spmm_csr(compress_data, codebook, indptr,
                                       indices, 100)

    print(block.formats())

    for i in range(10):
        begin = time.time()
        decompress_features = capi.vq_decompress(compress_data, codebook, 100)
        out1 = capi.spmm_csr(decompress_features, indptr, indices)
        torch.cuda.synchronize()
        print((time.time() - begin) * 1000)

    print()

    for i in range(10):
        begin = time.time()
        out2 = capi.vq_decompress_spmm_csr(compress_data, codebook, indptr,
                                           indices, 100)
        torch.cuda.synchronize()
        print((time.time() - begin) * 1000)

    assert torch.equal(out1, out2)


def test_fusion_sq_spmmcsr1():
    indptr = torch.tensor([0, 1, 3, 6]).long().cuda()
    indices = torch.tensor([0, 1, 1, 2, 2, 0]).long().cuda()
    features = torch.randn(6, 100).float()

    for target_bits in [1, 2, 4, 8, 16]:
        (fmin, fmax, emin, emax, mean) = sample(features)
        codebooks = torch.tensor([fmin, fmax, emin, emax,
                                  mean]).float().reshape(1, 5).cuda()
        compress_data = capi_sq_compress(features.cuda(), target_bits,
                                         codebooks, features.shape[1])
        if target_bits < 8:
            # packbits
            compress_data = compress_data.to(torch.uint8)
            exp = capi.packbits(compress_data, target_bits)
        elif target_bits == 8:
            exp = compress_data.to(torch.int8)
        elif target_bits == 16:
            exp = compress_data.to(torch.int16)

        decompress_features = capi.sq_decompress(exp, codebooks, target_bits,
                                                 100, 100)
        out1 = capi.spmm_csr(decompress_features, indptr, indices)

        out2 = capi.sq_decompress_spmm_csr(exp, codebooks, indptr, indices,
                                           target_bits, 100, 100)

        assert torch.equal(out1, out2)


def test_fusion_sq_spmmcsr2():
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

    features = torch.randn(input_nodes.numel(), 128).float()
    features = (features * 10).int().float()

    for target_bits in [1, 2, 4, 8, 16]:
        (fmin, fmax, emin, emax, mean) = sample(features)
        codebooks = torch.tensor([fmin, fmax, emin, emax,
                                  mean]).float().reshape(1, 5).cuda()
        compress_data = capi_sq_compress(features.cuda(), target_bits,
                                         codebooks, features.shape[1])
        if target_bits < 8:
            # packbits
            compress_data = compress_data.to(torch.uint8)
            exp = capi.packbits(compress_data, target_bits)
        elif target_bits == 8:
            exp = compress_data.to(torch.int8)
        elif target_bits == 16:
            exp = compress_data.to(torch.int16)

        decompress_features = capi.sq_decompress(exp, codebooks, target_bits,
                                                 128, 128)
        out1 = capi.spmm_csr(decompress_features, indptr, indices)

        out2 = capi.sq_decompress_spmm_csr(exp, codebooks, indptr, indices,
                                           target_bits, 128, 128)

        print(block.formats())

        for i in range(10):
            begin = time.time()
            decompress_features = capi.sq_decompress(exp, codebooks,
                                                     target_bits, 128, 128)
            out1 = capi.spmm_csr(decompress_features, indptr, indices)
            torch.cuda.synchronize()
            print((time.time() - begin) * 1000)

        print()

        for i in range(10):
            begin = time.time()
            out2 = capi.sq_decompress_spmm_csr(exp, codebooks, indptr, indices,
                                               target_bits, 128, 128)
            torch.cuda.synchronize()
            print((time.time() - begin) * 1000)

        assert torch.equal(out1, out2)


if __name__ == "__main__":
    test_fusion_vq_spmmcsr1()
    test_fusion_vq_spmmcsr2()
    test_fusion_sq_spmmcsr1()
    test_fusion_sq_spmmcsr2()
