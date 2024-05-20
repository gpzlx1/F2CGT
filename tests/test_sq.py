import torch
import math
import F2CGTLib as capi
import time
import numpy as np


def sample(data):
    epsilon = 1e-5
    sample = data.float().abs()
    fmin = max(np.percentile(np.abs(sample), 0.5), epsilon)
    fmax = max(np.percentile(np.abs(sample), 99.5), 2 * epsilon)

    fmin = torch.tensor(fmin)
    fmax = torch.tensor(fmax)
    emin = torch.log2(fmin)
    emax = torch.log2(fmax).add(epsilon)
    mean = sample.mean()
    if mean < 0.1:
        mean += 0.1
    return (fmin.cuda(), fmax.cuda(), emin.cuda(), emax.cuda(), mean.cuda())


def python_sq_compress(data, target_bits, fmin, fmax, emin, emax):
    begin = time.time()
    drange = 2**(target_bits - 1)

    sign = torch.sign(data)
    if drange == 1:
        data = torch.where(sign <= 0, 0, 1)
    else:
        data = torch.abs(data)
        data = torch.clip(data, fmin, fmax)
        exp = torch.log2(data)

        exp = torch.floor((exp - emin) / (emax - emin) * drange)
        if target_bits < 8:
            data = torch.where(sign <= 0, drange - 1 - exp, exp + drange)
        else:
            data = torch.where(sign <= 0, -1 - exp, exp)

    torch.cuda.synchronize()
    end = time.time()
    # print((end - begin) * 1000)
    return data


def capi_sq_compress(data, target_bits, codebooks, column_slice):
    begin = time.time()
    # codebooks = torch.tensor([fmin, fmax, emin, emax,
    #                          mean]).float().reshape(1, 5).cuda()
    # column_slice = data.shape[1]
    data = capi.sq_compress(data, codebooks, target_bits, column_slice)
    torch.cuda.synchronize()
    end = time.time()
    # print((end - begin) * 1000)
    return data


def simple_test_sq_compress():
    for target_bits in [1, 2, 4, 8, 16]:
        data = torch.empty((10000, 200)).float().uniform_(-1, 1)
        (fmin, fmax, emin, emax, mean) = sample(data)
        data = data.cuda()
        data1 = python_sq_compress(data, target_bits, fmin, fmax, emin, emax)

        codebooks = torch.tensor([fmin, fmax, emin, emax,
                                  mean]).float().reshape(1, 5).cuda()
        data2 = capi_sq_compress(data, target_bits, codebooks, data.shape[1])
        max_diff = (data1 - data2).abs().max()
        print(max_diff, data1.abs().max(), data1.abs().min())
        # assert torch.equal(data1, data2)


def python_sq_decompress(data, target_bits, fmin, fmax, emin, emax, mean):
    begin = time.time()

    drange = 2**(target_bits - 1)
    feat_dim = data.shape[1]

    compress_data = python_sq_compress(data, target_bits, fmin, fmax, emin,
                                       emax)

    if target_bits < 8:
        # packbits
        compress_data = compress_data.to(torch.uint8)
        exp = capi.packbits(compress_data, target_bits)
        exp = capi.unpackbits(exp, target_bits, feat_dim)

    elif target_bits == 8:
        exp = compress_data.to(torch.int8)

    elif target_bits == 16:
        exp = compress_data.to(torch.int16)

    exp = exp.to(torch.float32).cuda()
    if target_bits > 1:
        if target_bits < 8:
            exp = exp - drange
        exp = exp + 0.5

        sign = torch.sign(exp)
        decompress_data = exp.abs_().mul_(
            (emax - emin) / drange).add_(emin).exp2_().mul_(sign)
    else:
        decompress_data = (exp.sub_(0.5)).mul_(2 * mean)

    torch.cuda.synchronize()
    end = time.time()
    # print((end - begin) * 1000)
    return decompress_data


def capi_sq_decompress(data, target_bits, codebooks, column_slice):
    begin = time.time()
    feat_dim = data.shape[1]
    #codebooks = torch.tensor([fmin, fmax, emin, emax,
    #                          mean]).float().reshape(1, 5).cuda()
    #column_slice = data.shape[1]

    compress_data = capi_sq_compress(data, target_bits, codebooks,
                                     column_slice)

    if target_bits < 8:
        # packbits
        compress_data = compress_data.to(torch.uint8)
        exp = capi.packbits(compress_data, target_bits)

    elif target_bits == 8:
        exp = compress_data.to(torch.int8)

    elif target_bits == 16:
        exp = compress_data.to(torch.int16)

    #print("compression ratio: {}".format(data.numel() * data.element_size() /
    #                                     (exp.numel() * exp.element_size())))

    decompress_data = capi.sq_decompress(exp, codebooks, target_bits,
                                         column_slice, feat_dim)

    torch.cuda.synchronize()
    end = time.time()
    # print((end - begin) * 1000)
    return decompress_data


def simpe_test_sq_decompress():
    for target_bits in [1, 2, 4, 8, 16]:
        data = torch.empty((10000, 128)).float().uniform_(-1, 1)
        (fmin, fmax, emin, emax, mean) = sample(data)
        data = data.cuda()
        data1 = python_sq_decompress(data, target_bits, fmin, fmax, emin, emax,
                                     mean)
        codebooks = torch.tensor([fmin, fmax, emin, emax,
                                  mean]).float().reshape(1, 5).cuda()
        data2 = capi_sq_decompress(data, target_bits, codebooks, data.shape[1])
        max_diff = (data1 - data2).abs().max()
        mean_diff = (data1 - data2).abs().mean()
        print(max_diff, mean_diff, data1.abs().max(), data1.abs().min())


def complex_test_sq_decompress():
    column_slice = 60
    for target_bits in [1, 2, 4, 8, 16]:
        data1 = torch.empty((10000, column_slice)).float().uniform_(-1, 1)
        data2 = torch.empty(
            (10000, int(column_slice * 5 / 6))).float().uniform_(-1, 1) + 10
        codebook1 = sample(data1)
        codebook2 = sample(data2)

        data1 = data1.cuda()
        data2 = data2.cuda()
        python_data1 = python_sq_decompress(data1, target_bits, *codebook1)
        python_data2 = python_sq_decompress(data2, target_bits, *codebook2)
        python_data = torch.cat([python_data1, python_data2], dim=1)

        codebooks = torch.tensor([*codebook1,
                                  *codebook2]).float().reshape(2, 5).cuda()
        data = torch.cat([data1, data2], dim=1)
        capi_data = capi_sq_decompress(data, target_bits, codebooks,
                                       column_slice)

        max_diff = (python_data - capi_data).abs().max()
        mean_diff = (python_data - capi_data).abs().mean()
        print(max_diff, mean_diff, data1.abs().max(), data1.abs().min())


if __name__ == '__main__':
    simple_test_sq_compress()
    simpe_test_sq_decompress()
    complex_test_sq_decompress()
