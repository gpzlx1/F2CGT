import torch
import math
import numpy as np
import tqdm
from .packbits import packbits, unpackbits
from cuml import KMeans


def sq_compress(tensor,
                target_bits,
                device,
                sample_size=10_0000,
                compress_batch_size=100_0000,
                fake_feat_items=0,
                fake_feat_dim=0):
    codebook = None
    if tensor is None:
        assert fake_feat_items > 0
        assert fake_feat_dim > 0
        num_items = fake_feat_items
        feat_dim = fake_feat_dim
        fake_feat = True
    else:
        fake_feat = False
        num_items = tensor.shape[0]
        feat_dim = tensor.shape[1]

    if target_bits <= 8:
        dtype = torch.int8
    elif target_bits <= 16:
        dtype = torch.int16
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

    if fake_feat:
        sample = torch.randn(
            (sample_size, )).reshape(-1, 1).repeat(1, feat_dim).float()
    else:
        perm = torch.randperm(num_items)
        sample_size = num_items // 10 if num_items // 10 >= sample_size else sample_size
        sample = tensor[perm[:sample_size]]

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

    print("start scalar compressing, precision={}, fmin={}, fmax={}".format(
        target_bits, fmin, fmax))

    compressed_tensor = torch.empty((num_items, tfeat_dim), dtype=dtype)

    compress_batch_size = compress_batch_size if num_items > compress_batch_size else num_items
    for start in tqdm.trange(0, num_items, compress_batch_size):
        end = min(num_items, start + compress_batch_size)
        if fake_feat:
            tensor_ = torch.randn((min(compress_batch_size,
                                       end - start), )).reshape(-1, 1).repeat(
                                           1, feat_dim).float().cuda()
        else:
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
            compressed_tensor[start:end].copy_(tensor_.to(dtype=torch.int8),
                                               non_blocking=True)
        elif target_bits <= 16:
            compressed_tensor[start:end].copy_(tensor_.to(dtype=torch.int16),
                                               non_blocking=True)
        else:
            raise NotImplementedError

    if tensor_.is_cuda:
        torch.cuda.synchronize()

    return compressed_tensor, codebook


def sq_decompress(compressed_tensor, feat_dim, codebook):
    emin, emax, mean, drange, target_bits = codebook
    drange = drange.item()

    exp = compressed_tensor
    if target_bits < 8:
        exp = unpackbits(exp,
                         mask=2 * drange - 1,
                         shape=[exp.shape[0], feat_dim])

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


def vq_compress(tensor,
                width,
                length,
                device,
                sample_size=10_0000,
                compress_batch_size=100_0000,
                fake_feat_items=0,
                fake_feat_dim=0):
    if fake_feat_items > 0 and fake_feat_dim > 0:
        num_items = fake_feat_items
        feat_dim = fake_feat_dim
        fake_feat = True
    else:
        fake_feat = False
        num_items = tensor.shape[0]
        feat_dim = tensor.shape[1]
    num_parts = (feat_dim + width - 1) // width

    codebooks = torch.zeros((num_parts, length, width))

    if length <= 256:
        dtype = torch.uint8
    elif length <= 32767:
        dtype = torch.int16
    else:
        dtype = torch.int32

    if fake_feat:
        sample = torch.randn(
            (sample_size, )).reshape(-1, 1).repeat(1, feat_dim).float()
    else:
        perm = torch.randperm(num_items)
        sample_size = num_items // 10 if num_items // 10 >= sample_size else sample_size
        sample = tensor[perm[:sample_size]].to(device)

    print("generate codebooks:")
    kmeans_list = [KMeans(n_clusters=length) for _ in range(num_parts)]
    for i in tqdm.trange(0, num_parts):
        X = sample[:, i * width:(i + 1) * width]
        kmeans_list[i].fit(X)
        cluster_centers = kmeans_list[i].cluster_centers_
        codebooks[i, :, :feat_dim - i * width] = torch.tensor(cluster_centers)
        del X

    cluster_ids = torch.empty((num_items, num_parts), dtype=dtype)
    compress_batch_size = compress_batch_size if num_items > compress_batch_size else num_items
    for step in tqdm.trange(0, num_items, compress_batch_size):
        start = step
        end = min(step + compress_batch_size, num_items)
        if fake_feat:
            tensor_ = torch.randn((min(compress_batch_size,
                                       end - start), )).reshape(-1, 1).repeat(
                                           1, feat_dim).float().cuda()
        else:
            tensor_ = tensor[start:end].to(device).to(torch.float32)

        for j in range(num_parts):
            X = tensor_[:, j * width:(j + 1) * width]
            labels = kmeans_list[j].predict(X)
            cluster_ids[start:end, j] = torch.tensor(labels, dtype=dtype)

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
            codebook[i], 0, compressed_features[:, i].flatten().long())

    result[:, (num_parts - 1) * width:] = torch.index_select(
        codebook[num_parts - 1, :, :feat_dim - (num_parts - 1) * width], 0,
        compressed_features[:, num_parts - 1].flatten().long())

    return result
