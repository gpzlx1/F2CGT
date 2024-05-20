import torch
import math
import F2CGTLib as capi
import time
import numpy as np


def vq_compress(data, num_class):
    kmeans = capi.KMeans(num_class, data.shape[1], "l2")
    kmeans.fit(data.cuda())
    cluster_centers = kmeans.get_centers()
    labels = kmeans.predict(data)
    labels = labels.to(torch.uint8)
    return cluster_centers, labels


def python_vq_decompress(cluster_centers, labels):
    labels = labels.to(torch.int64)
    return torch.index_select(cluster_centers, dim=0, index=labels)


def simple_test_compress():
    data = torch.randn(10000, 10).float().cuda()
    cluster_centers, labels = vq_compress(data, 256)
    print(cluster_centers)
    print(labels)


def simple_test_decompress():
    for dtype in [torch.float32]:
        feat_dim = 128
        data = torch.empty(100_0000, feat_dim).to(dtype).cuda().uniform_(-5, 5)
        cluster_centers, labels = vq_compress(data, 256)

        for i in range(10):
            begin = time.time()
            python_data = python_vq_decompress(cluster_centers, labels)
            torch.cuda.synchronize()
            end = time.time()
            print((end - begin) * 1000)

        cluster_centers = cluster_centers.reshape(-1, *cluster_centers.shape)

        for i in range(10):
            begin = time.time()
            capi_data = capi.vq_decompress(labels, cluster_centers, feat_dim)
            torch.cuda.synchronize()
            end = time.time()
            print((end - begin) * 1000)

        assert torch.equal(python_data, capi_data)


def complex_test_decompress():

    data1 = torch.empty(100_0000, 60).float().cuda().uniform_(-5, 5)
    cluster_centers1, labels1 = vq_compress(data1, 256)
    data2 = torch.empty(100_0000, 50).float().cuda().uniform_(-10, 10)
    cluster_centers2, labels2 = vq_compress(data2, 256)

    for i in range(10):
        begin = time.time()
        python_data1 = python_vq_decompress(cluster_centers1, labels1)
        python_data2 = python_vq_decompress(cluster_centers2, labels2)
        python_data = torch.cat([python_data1, python_data2], dim=1)
        torch.cuda.synchronize()
        end = time.time()
        print((end - begin) * 1000)

    capi_codebook1 = cluster_centers1.reshape(-1, *cluster_centers1.shape)
    capi_codebook2 = torch.zeros_like(capi_codebook1)
    capi_codebook2[:, :, :50] = cluster_centers2

    capi_codebooks = torch.cat([capi_codebook1, capi_codebook2], dim=0)
    labels1 = labels1.reshape(*labels1.shape, 1)
    labels2 = labels2.reshape(*labels2.shape, 1)
    index = torch.cat([labels1, labels2], dim=1)

    for i in range(10):
        begin = time.time()
        capi_data = capi.vq_decompress(index, capi_codebooks, 60 + 50)
        torch.cuda.synchronize()
        end = time.time()
        print((end - begin) * 1000)

    assert torch.equal(python_data, capi_data)


if __name__ == "__main__":
    simple_test_decompress()
    print()
    complex_test_decompress()
