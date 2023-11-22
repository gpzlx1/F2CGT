import torch
import bifeat
import numpy as np
import os
from torch import distributed as dist
import time


def main(num_gpus=2):
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print("{} of {}".format(rank, world_size))

    omp_thread_num = os.cpu_count() // num_gpus
    torch.cuda.set_device(rank)
    torch.set_num_threads(omp_thread_num)

    print("Set device to {} and cpu threads num {}".format(
        rank, omp_thread_num))

    chief = rank % num_gpus == 0

    shm_manager = bifeat.shm.ShmManager(rank,
                                        num_gpus,
                                        './datasets',
                                        'ogbn-products',
                                        pin_memory=True)

    begin = time.time()
    graph_tensors, meta_data = shm_manager.load_dataset(with_feature=True)
    torch.cuda.synchronize()
    print("Load dataset time: {}".format(time.time() - begin))

    compression_manager = bifeat.CompressionManager(ratios=[0.1, 0.9],
                                                    methods=['sq', 'vq'],
                                                    configs=[{
                                                        'target_bits': 4
                                                    }, {
                                                        'width': 32,
                                                        'length': 256
                                                    }],
                                                    cache_path='./cache',
                                                    shm_manager=shm_manager)
    compression_manager.register(
        graph_tensors['indptr'],
        graph_tensors['indices'],
        graph_tensors['train_idx'],
        graph_tensors['labels'],
        graph_tensors['features'],
    )

    begin = time.time()
    compression_manager.presampling([25, 10])
    torch.cuda.synchronize()
    print("presampling time: {}".format(time.time() - begin))

    begin = time.time()
    compression_manager.graph_reorder()
    torch.cuda.synchronize()
    print("graph_reorder time: {}".format(time.time() - begin))

    begin = time.time()
    compression_manager.compress()
    torch.cuda.synchronize()
    print("compress time: {}".format(time.time() - begin))

    indices = 8_0000
    indices = torch.tensor([indices]).long()

    begin = time.time()
    out = compression_manager.decompress(indices)
    torch.cuda.synchronize()
    print("decompress time: {}".format(time.time() - begin))

    if chief:
        print(out[0][:10])
        print(graph_tensors['features'][compression_manager.dst2src[indices]]
              [:10])


if __name__ == '__main__':
    main()
