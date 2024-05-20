import torch
import os
import torch.distributed as dist
import time
from shmtensor import ShmTensor, GPUSamplingDataloader
import argparse
import numpy as np
import tqdm
import math
import F2CGTLib as capi
# from kmeans1 import kmeans1
from my_kmeans import KMeansTrainer
from F2CGTLib import KMeans as KMeansInfer


def normalize(data):
    data_norm = data.norm(dim=1, p=2)
    return data / data_norm.reshape(-1, 1) * data_norm.mean()


def sampling(sample_size, num_items, hotness):

    if hotness is None:
        sample_index = torch.randperm(num_items)[:sample_size]
    else:
        hotness = hotness.tensor_ / hotness.tensor_.sum()
        hotness[-1] = 1.0 - hotness[:-1].sum()
        sample_index = np.random.choice(num_items,
                                        sample_size,
                                        replace=False,
                                        p=hotness)
    sample_index = torch.tensor(sample_index)

    return sample_index.cpu()


def initialize(x, num_clusters):
    nonzero_idxs = x.norm(dim=1, p=0).nonzero().squeeze()
    num_samples = len(nonzero_idxs)
    indices = np.random.choice(num_samples, num_clusters, replace=False)
    initial_state = x[nonzero_idxs[indices]]
    return initial_state


def sq_metadata(features, slice_begin, slice_end, hotness):
    sample_index = None
    sample_size = 25_0000
    num_items = features.shape[0]

    sample_index = sampling(sample_size, num_items, hotness)

    sample_part_feature = features.tensor_[sample_index, slice_begin:slice_end]

    ## compute the mean and std, generate codebooks
    epsilon = 1e-5
    sample = sample_part_feature.float().abs()
    fmin = max(np.percentile(np.abs(sample), 0.5), epsilon)
    fmax = max(np.percentile(np.abs(sample), 99.5), 2 * epsilon)

    fmin = torch.tensor(fmin)
    fmax = torch.tensor(fmax)
    emin = torch.log2(fmin)
    emax = torch.log2(fmax).add(epsilon)
    mean = sample.mean()
    if mean < 0.1:
        mean += 0.1

    return fmin, fmax, emin, emax, mean


def sq_compress(features, target_bits, column_slice, hotness=None):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    feat_dim = features.shape[1]
    num_items = features.shape[0]

    if rank == 0:
        print('target_bits', target_bits)
        print('column_slice', column_slice)

    # first generate metadata
    slice_begin = column_slice * rank
    slice_end = min(column_slice * (rank + 1), feat_dim)

    codebooks = torch.zeros((world_size, 5), device='cuda')
    if slice_begin < slice_end:
        fmin, fmax, emin, emax, mean = sq_metadata(features, slice_begin,
                                                   slice_end, hotness)
        codebooks[rank, 0] = fmin
        codebooks[rank, 1] = fmax
        codebooks[rank, 2] = emin
        codebooks[rank, 3] = emax
        codebooks[rank, 4] = mean

    dist.all_gather_into_tensor(codebooks, codebooks[rank, :])

    # remove invalid codebooks
    num_codebooks = (feat_dim + column_slice - 1) // column_slice
    codebooks = codebooks[:num_codebooks, :]

    if rank == 0:
        print(codebooks.shape, codebooks)

    # begin compress
    row_slice = (num_items + world_size - 1) // world_size
    row_begin = row_slice * rank
    row_end = min(row_slice * (rank + 1), num_items)

    if target_bits <= 8:
        dtype = torch.int8
        tfeat_dim = int(math.ceil(feat_dim / 8 * target_bits))
    elif target_bits <= 16:
        dtype = torch.int16
        tfeat_dim = feat_dim
    else:
        raise NotImplementedError

    tfeatures = ShmTensor("tfeatures",
                          shape=((num_items, tfeat_dim)),
                          dtype=dtype,
                          local_rank=rank,
                          local_world_size=world_size)

    quantize_batch_size = 1000000
    for start in tqdm.trange(row_begin, row_end, quantize_batch_size):
        end = min(start + quantize_batch_size, row_end)
        data = features.tensor_[start:end, :].cuda().to(torch.float32)
        compress_data = capi.sq_compress(data, codebooks, target_bits,
                                         column_slice)

        if target_bits < 8:
            compress_data = compress_data.to(torch.uint8)
            exp = capi.packbits(compress_data, target_bits)

        elif target_bits == 8:
            exp = compress_data.to(torch.int8)

        elif target_bits == 16:
            exp = compress_data.to(torch.int16)

        tfeatures.tensor_[start:end, :].copy_(exp, non_blocking=True)

    dist.barrier()
    return tfeatures, codebooks


def vq_metadata(features,
                n_clusters,
                column_slice,
                slice_begin,
                slice_end,
                hotness=None):
    sample_index = None
    sample_size = 15_0000
    num_items = features.shape[0]
    # num_items = 244160499

    sample_index = sampling(sample_size, num_items, hotness)

    num_slice = (slice_end - slice_begin + column_slice - 1) // column_slice
    codebooks = torch.zeros((num_slice, n_clusters, column_slice),
                            device='cuda',
                            dtype=torch.float32)
    for i in tqdm.trange(num_slice, disable=dist.get_rank() != 0):
        slice_begin_ = slice_begin + i * column_slice
        slice_end_ = min(slice_begin_ + column_slice, slice_end)
        data = features.tensor_[sample_index,
                                slice_begin_:slice_end_].to(torch.float32)
        # data = torch.randn((sample_index.numel(), slice_end_ - slice_begin_),
        #                    dtype=torch.float32,
        #                    device='cuda')
        distance = data.norm(dim=1, p=2)
        rim = torch.quantile(distance, 0.00625 * (data.shape[1] / 8))
        data = data[torch.ge(distance, rim)]
        km = KMeansTrainer(n_clusters, metric="cosine", tol=5e-2, mode='fast')
        km.fit(data.cuda())
        cluster_centers = km.get_centers()
        length = cluster_centers.shape[1]
        cluster_centers = normalize(cluster_centers)
        codebooks[i, :, :length] = cluster_centers.cpu()

    return codebooks


def vq_compress(features, n_clusters, column_slice, hotness=None):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    feat_dim = features.shape[1]
    num_items = features.shape[0]
    # num_items = 244160499

    num_slice = (feat_dim + column_slice - 1) // column_slice
    slice_per_gpu = (num_slice + world_size - 1) // world_size

    slice_begin = column_slice * slice_per_gpu * rank
    slice_end = min(column_slice * slice_per_gpu * (rank + 1), feat_dim)

    if rank == 0:
        print("n_clusters, column_slice, slice_begin, slice_end: {} {} {} {}".
              format(n_clusters, column_slice, slice_begin, slice_end))

    if slice_begin < slice_end:
        part_codebooks = vq_metadata(features, n_clusters, column_slice,
                                     slice_begin, slice_end, hotness)
    else:
        part_codebooks = torch.zeros((0, n_clusters, column_slice),
                                     device='cuda',
                                     dtype=torch.float32)

    print("rank {}: {}".format(rank, part_codebooks.shape))

    # generate codebooks
    all_codebooks = []
    for i in range(world_size):
        _begin = column_slice * slice_per_gpu * i
        _end = min(column_slice * slice_per_gpu * (i + 1), feat_dim)
        num_codebooks = (_end - _begin + column_slice - 1) // column_slice
        num_codebooks = max(num_codebooks, 0)
        tensor = torch.zeros(
            (num_codebooks, part_codebooks.shape[1], part_codebooks.shape[2]),
            device='cuda',
            dtype=torch.float32)
        all_codebooks.append(tensor)

    dist.all_gather(all_codebooks, part_codebooks)

    codebooks = torch.cat(all_codebooks, dim=0)

    if rank == 0:
        print(codebooks.shape)

    if torch.isnan(codebooks).any():
        raise ValueError("codebooks has nan")

    # begin compress
    row_slice = (num_items + world_size - 1) // world_size
    row_begin = row_slice * rank
    row_end = min(row_slice * (rank + 1), num_items)

    if n_clusters <= 256:
        dtype = torch.uint8
    elif n_clusters <= 32767:
        dtype = torch.int16
    else:
        dtype = torch.int32

    tfeatures = ShmTensor("tfeatures",
                          shape=(num_items, num_slice),
                          dtype=dtype,
                          local_rank=rank,
                          local_world_size=world_size)

    km_list = []
    for i in range(num_slice):
        km_list.append(KMeansInfer(codebooks[i, :, :], "cosine"))

    quantize_batch_size = 1000000
    for start in tqdm.trange(row_begin, row_end, quantize_batch_size):
        end = min(start + quantize_batch_size, row_end)
        # data = features.tensor_[start:end, :].to(torch.float32)

        for i in range(num_slice):
            part_slice_begin = i * column_slice
            part_slice_end = min(part_slice_begin + column_slice, feat_dim)
            length = part_slice_end - part_slice_begin

            if length <= 0:
                continue
            elif length < column_slice:
                data = torch.zeros((end - start, column_slice),
                                   device='cpu',
                                   dtype=torch.float32)
                data[:, :length] = features.tensor_[
                    start:end,
                    part_slice_begin:part_slice_end].to(torch.float32)
                # data[:, :length] = torch.randn((end - start, length),
                #                                dtype=torch.float32,
                #                                device='cuda')
            else:
                data = features.tensor_[start:end,
                                        part_slice_begin:part_slice_end].to(
                                            torch.float32)
                # data = torch.randn((end - start, length),
                #                    dtype=torch.float32,
                #                    device='cuda')
            km = km_list[i]
            labels = km.predict(data.cuda(), normalize_weights=False)
            tfeatures.tensor_[start:end, i] = labels.cpu()

    dist.barrier()

    return tfeatures, codebooks


def run(args, data, target_compression_ratio=32):
    tic = time.time()
    # shape, dtype
    features, seeds, hotness = data

    feat_dim = features.tensor_.shape[1]
    feat_element_size = features.tensor_.element_size() * 8

    methods = []
    configs = []
    codebook_list = []
    data = {}
    compression_meta = {}

    if dist.get_rank() == 0:
        print(feat_dim, feat_element_size)

    if seeds is not None:
        seed_features = ShmTensor("seed_features",
                                  shape=(seeds.tensor_.numel(),
                                         features.tensor_.shape[1]),
                                  dtype=features.tensor_.dtype,
                                  local_rank=dist.get_rank(),
                                  local_world_size=dist.get_world_size())
        if dist.get_rank() == 0:
            seed_features.tensor_[:] = features.tensor_[seeds.tensor_, :]
            # seed_features.tensor_.uniform_(-5, 5)
        dist.barrier()

        print(seed_features.tensor_.shape)

        # sq compression for seeds_features
        seeds_target_bits = feat_element_size // 8
        column_slice = (feat_dim + dist.get_world_size() -
                        1) // dist.get_world_size()
        column_slice = (column_slice + 7) // 8 * 8
        seeds_tfeatures, seeds_codebooks = sq_compress(seed_features,
                                                       seeds_target_bits,
                                                       column_slice, None)

        methods.append('sq')
        configs.append((column_slice, seeds_target_bits))
        codebook_list.append(seeds_codebooks.cpu())
        data["seeds_compression_data"] = (seeds_tfeatures.tensor_.dtype,
                                          seeds_tfeatures.tensor_.shape)
        compression_meta['seeds'] = seeds.tensor_

        torch.save(seeds_tfeatures.tensor_,
                   os.path.join(args.root, 'seeds_compression_data.pt'))
        del seeds_tfeatures

        print("finish seeds compression")

    if target_compression_ratio <= feat_element_size:
        target_bits = int(feat_element_size / target_compression_ratio)
        column_slice = (feat_dim + dist.get_world_size() -
                        1) // dist.get_world_size()
        column_slice = (column_slice + 7) // 8 * 8
        tfeatures, codebooks = sq_compress(features, target_bits, column_slice,
                                           hotness)

        methods.append('sq')
        configs.append((column_slice, target_bits))
        codebook_list.append(codebooks.cpu())
        data["compression_data"] = (tfeatures.tensor_.dtype,
                                    tfeatures.tensor_.shape)

    else:
        n_clusters = args.n_clusters  # 1 bytes
        column_slice = target_compression_ratio // features.tensor_.element_size(
        )

        if n_clusters <= 256:
            column_slice = column_slice * 1
        elif n_clusters <= 32767:
            column_slice = column_slice * 2

        tfeatures, codebooks = vq_compress(features, n_clusters, column_slice,
                                           hotness)

        methods.append('vq')
        configs.append((column_slice, -1))
        codebook_list.append(codebooks.cpu())
        data["compression_data"] = (tfeatures.tensor_.dtype,
                                    tfeatures.tensor_.shape)

    toc = time.time()
    print("Compress time cost: {:.3f} s".format(toc - tic))

    if dist.get_rank() == 0:
        compression_meta['methods'] = methods
        compression_meta['codebooks'] = codebook_list
        compression_meta['configs'] = configs
        compression_meta['feat_dim'] = feat_dim
        compression_meta['data'] = data
        torch.save(compression_meta,
                   os.path.join(args.root, 'compression_meta.pt'))

        torch.save(tfeatures.tensor_,
                   os.path.join(args.root, 'compression_data.pt'))


def load_meta(args):
    meta = torch.load(os.path.join(args.root, 'meta.pt'))
    return meta


def load_shm_tensor(name, args, rank, world_size, meta):
    if name not in meta:
        raise Exception("Invalid name")

    shmtensor = ShmTensor(name,
                          shape=meta[name][1],
                          dtype=meta[name][0],
                          local_rank=rank,
                          local_world_size=world_size,
                          pin_memory=True)

    if rank == 0:
        tmp_tensor = torch.load(os.path.join(args.root, name + ".pt"),
                                mmap=True)
        assert tmp_tensor.shape == shmtensor.shape
        assert tmp_tensor.dtype == shmtensor.dtype
        shmtensor.tensor_[:] = tmp_tensor
        del tmp_tensor

    dist.barrier()
    return shmtensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-hotness", action="store_true", default=False)
    parser.add_argument("--with-seeds", action="store_true", default=False)
    parser.add_argument("--with-feature", action="store_true", default=False)
    # parser.add_argument("--graph", type=str, required=True)
    parser.add_argument("--n-clusters", type=int, default=128)
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--target-compression-ratio", type=int, default=32)
    args = parser.parse_args()
    print(args)

    dist.init_process_group(backend="nccl", init_method="env://")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"rank: {rank}, world_size: {world_size}")

    torch.cuda.set_device(rank)

    meta = load_meta(args)
    print(meta)
    features = load_shm_tensor("features", args, rank, world_size, meta)
    '''
    if args.with_feature:
        features = load_shm_tensor("features", args, rank, world_size, meta)
    else:
        # 244160499
        features = ShmTensor("features",
                             shape=(10, 768),
                             dtype=meta["features"][0],
                             local_rank=rank,
                             local_world_size=world_size,
                             pin_memory=False)
        dist.barrier()
    '''

    if args.with_hotness:
        hotness = load_shm_tensor("hotness", args, rank, world_size, meta)
    else:
        hotness = None

    if args.with_seeds:
        seeds = load_shm_tensor("seeds", args, rank, world_size, meta)
    else:
        seeds = None

    data = (features, seeds, hotness)
    run(args, data, args.target_compression_ratio)


if __name__ == "__main__":
    main()
