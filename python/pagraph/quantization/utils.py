import torch
import math
import numpy as np
import tqdm
from .packbits import packbits, unpackbits
from .kmeans import get_centers, kmeans_predict


def sq_compress(tensor, target_bits, device):
    codebook = None
    num_items = tensor.shape[0]
    feat_dim = tensor.shape[1]

    if target_bits <= 8:
        dtype = torch.int8
    else:
        raise NotImplementedError

    if target_bits < 8:
        tfeat_dim = int(math.ceil(feat_dim / 8 * target_bits))
    else:
        tfeat_dim = feat_dim

    emin = 0
    emax = 0
    drange = 2**(target_bits - 1)
    epsilon = 1e-5

    perm = torch.randperm(num_items)
    sample = tensor[perm[:10_0000]]

    fmin = max(np.percentile(np.abs(sample), 0.5), epsilon)
    fmax = max(np.percentile(np.abs(sample), 99.5), 2 * epsilon)
    fmin = torch.tensor(fmin)
    fmax = torch.tensor(fmax)

    emin = torch.log2(fmin)
    emax = torch.log2(fmax).add(epsilon)
    mean = sample.float().abs().mean()
    if mean < 0.1:
        mean += 0.1

    codebook = torch.zeros(5)
    codebook[0] = emin
    codebook[1] = emax
    codebook[2] = mean
    codebook[3] = drange
    codebook[4] = target_bits

    emin = emin.to(device)
    emax = emax.to(device)
    mean = mean.to(device)
    fmin = fmin.to(device)
    fmax = fmax.to(device)

    print("scalar codebook: {}".format(codebook))

    print("start scalar compressing, precision={}, fmin={}, fmax={}".format(
        target_bits, fmin, fmax))

    compressed_tensor = torch.empty((num_items, tfeat_dim), dtype=dtype)

    compress_batch_size = 100_0000 if num_items > 100_0000 else num_items
    for start in tqdm.trange(0, num_items, compress_batch_size):
        end = min(num_items, start + compress_batch_size)

        tensor_ = tensor[start:end].to(device).to(torch.float32)
        sign = torch.sign(tensor_)
        if drange == 1:
            tensor_ = torch.where(sign <= 0, 0, 1)
        else:
            tensor_ = tensor_.abs()
            tensor_ = torch.clip(tensor_, fmin, fmax)
            exp = torch.log2(tensor_)

            exp = torch.floor((exp - emin) / (emax - emin) * drange)
            if target_bits < 8:
                tensor_ = torch.where(sign <= 0, drange - 1 - exp,
                                      exp + drange)
            else:
                tensor_ = torch.where(sign <= 0, -1 - exp, exp)

        if target_bits < 8:
            compressed_tensor[start:end].copy_(packbits(
                tensor_.to(torch.uint8), mask=(1 << target_bits) - 1),
                                               non_blocking=True)
        elif target_bits == 8:
            compressed_tensor[start:end].copy_(tensor_.to(dtype=torch.int16),
                                               non_blocking=True)
        else:
            raise NotImplementedError

    if tensor_.is_cuda:
        torch.cuda.synchronize()

    return compressed_tensor, codebook


def sq_decompress(compressed_tensor, feat_dim, codebook):
    emin, emax, mean, drange, target_bits = codebook

    exp = compressed_tensor
    if target_bits < 8:
        exp = unpackbits(exp,
                         mask=2 * drange - 1,
                         shape=[exp.shape[0], feat_dim],
                         dtype=torch.uint8)

    result = None
    if target_bits > 1:
        if target_bits < 8:
            exp = exp.to(torch.float32) - drange
        exp = exp + 0.5

        sign = torch.sign(exp)
        result = exp.abs_().mul_(
            (emax - emin) / drange).add_(emin).exp2_().mul_(sign)
    else:
        result = (exp.to(torch.float32).sub_(0.5)).mul_(2 * mean)

    return result


def vq_compress(tensor, width, length, device):
    num_items = tensor.shape[0]
    feat_dim = tensor.shape[1]

    codebooks = torch.zeros((math.ceil(feat_dim / width), length, width))

    if length <= 256:
        dtype = torch.uint8
    elif length <= 32767:
        dtype = torch.int16
    else:
        dtype = torch.int32

    perm = torch.randperm(num_items)
    sample = tensor[perm[:10_0000]]

    print("generate codebooks:")
    distance = "cosine"
    for i in tqdm.trange(0, math.ceil(feat_dim / width)):
        X = sample[:, i * width:(i + 1) * width]

        dist = X.norm(dim=1, p=2)
        rim = torch.quantile(dist, 0.8 / length)
        out = torch.ge(dist, rim)

        cluster_centers_o = get_centers(X=X[out],
                                        num_clusters=length,
                                        distance=distance,
                                        tol=5e-2 * length,
                                        device=device)
        codebooks[i, :, :feat_dim - i * width] = cluster_centers_o
        del X

    cluster_ids = torch.empty((num_items, math.ceil(feat_dim / width)),
                              dtype=dtype)
    compress_batch_size = 100_0000 if num_items > 100_0000 else num_items
    for i in tqdm.trange(0, num_items, compress_batch_size):
        start = i * compress_batch_size
        end = (i + 1) * compress_batch_size
        tensor_ = tensor[start:end].to(device).to(torch.float32)

        for j in range(math.ceil(feat_dim / width)):
            X = tensor_[:, j * width:(j + 1) * width]
            cluster_ids_x = kmeans_predict(X,
                                           codebooks[j, :, :feat_dim -
                                                     j * width],
                                           distance,
                                           device=device)

            cluster_ids[start:end, j] = cluster_ids_x

    return cluster_ids, codebooks


def vq_decompress(compressed_features, feat_dim, codebook):
    num_items = compressed_features.shape[0]
    num_parts = codebook.shape[0]
    length = codebook.shape[1]
    width = codebook.shape[2]

    result = torch.zeros((num_items, feat_dim),
                         dtype=torch.float32,
                         device=compressed_features.device)

    for i in range(num_parts - 1):
        begin = i * width
        end = (i + 1) * width

        result[:, begin:end] = torch.index_select(
            codebook[i], 0, compressed_features[:, i].flatten())

    result[:, (num_parts - 1) * width:] = torch.index_select(
        codebook[num_parts - 1, :, :self.feat_dim - (num_parts - 1) * width],
        0, compressed_features[:, num_parts - 1].flatten())

    return result
