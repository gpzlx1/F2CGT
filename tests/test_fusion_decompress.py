import torch
import math
import F2CGTLib as capi
import time
import numpy as np
from test_sq import capi_sq_compress, capi_sq_decompress, sample
from test_vq import vq_compress


def simpe_test_fusion_decompress():
    for target_bits in [1, 2, 4, 8, 16]:
        data = torch.empty((10000, 128)).float().uniform_(-1, 1)

        seeds_data = data[:2000]
        (fmin, fmax, emin, emax, mean) = sample(seeds_data)
        seeds_codebooks = torch.tensor([fmin, fmax, emin, emax,
                                        mean]).float().reshape(1, 5).cuda()
        seeds_data = capi_sq_compress(seeds_data.cuda(), target_bits,
                                      seeds_codebooks, 128)
        if target_bits < 8:
            seeds_data = seeds_data.to(torch.uint8)
            seeds_data = capi.packbits(seeds_data, target_bits)
        elif target_bits == 8:
            seeds_data = seeds_data.to(torch.int8)
        elif target_bits == 16:
            seeds_data = seeds_data.to(torch.int16)

        cluster_centers, labels = vq_compress(data[2000:].cuda(), 16)
        other_data = labels.reshape(*labels.shape, 1).cuda()
        other_codebook = cluster_centers.reshape(
            -1, *cluster_centers.shape).cuda()
        data1 = torch.cat([
            capi.sq_decompress(seeds_data, seeds_codebooks, target_bits, 128,
                               128),
            capi.vq_decompress(other_data.detach(), other_codebook.detach(),
                               128)
        ])
        data2 = capi.fusion_decompress(seeds_data, seeds_codebooks,
                                       target_bits, 128, other_data,
                                       other_codebook, 128)
        print(data1)
        print(data2)
        assert torch.equal(data1, data2)


if __name__ == '__main__':
    simpe_test_fusion_decompress()
