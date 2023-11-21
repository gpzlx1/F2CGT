import torch
import math
import numpy as np
import tqdm
from .packbits import packbits, unpackbits


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
    sample = tensor[perm[:100000]]

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

    quantize_batch_size = 100_0000 if num_items > 100_0000 else num_items
    for start in tqdm.trange(0, num_items, quantize_batch_size):
        end = min(num_items, start + quantize_batch_size)

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
