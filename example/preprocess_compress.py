import torch
import bifeat
import os
from torch import distributed as dist
import time
import argparse
"""
Usage: torchrun --nproc_per_node ${#gpus} example/preprocess_compress.py [args]
"""
if __name__ == "__main__":
    argparser = argparse.ArgumentParser("Partition builtin graphs")
    argparser.add_argument(
        "--dataset",
        type=str,
        default="ogbn-products",
        help="datasets: ogbn-products, ogbn-papers100M",
    )
    argparser.add_argument("--root",
                           type=str,
                           default="/data/ogbn_products/processed")
    argparser.add_argument("--num-gpus",
                           type=int,
                           default=2,
                           help="number of gpus participated in the compress")
    argparser.add_argument(
        "--methods",
        type=str,
        default="sq,vq",
        help=
        "compression methods, first one for core nodes (first level), second one for all nodes (second level)"
    )
    argparser.add_argument(
        "--configs",
        type=str,
        default="[{'target_bits':4},{'width':32,'length':256}]")
    argparser.add_argument("--save-path", type=str, default=".")
    argparser.add_argument("--fan-out",
                           type=str,
                           default="5,10,15",
                           help="fanout for presampling")
    argparser.add_argument("--batch-size",
                           type=int,
                           default=1024,
                           help="batch size for presampling")
    args = argparser.parse_args()
    print(args)

    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == args.num_gpus

    print("{} of {}".format(rank, world_size))

    omp_thread_num = os.cpu_count() // args.num_gpus
    torch.cuda.set_device(rank)
    torch.set_num_threads(omp_thread_num)

    print("Set device to {} and cpu threads num {}".format(
        rank, omp_thread_num))

    chief = rank % args.num_gpus == 0

    shm_manager = bifeat.shm.ShmManager(rank,
                                        args.num_gpus,
                                        args.root,
                                        args.dataset,
                                        pin_memory=True)

    begin = time.time()
    graph_tensors, meta_data = shm_manager.load_dataset(with_feature=True)
    torch.cuda.synchronize()
    print("Load dataset time: {:.3f} sec".format(time.time() - begin))

    methods = [method for method in args.methods.split(",")]
    compression_manager = bifeat.CompressionManager(methods=methods,
                                                    configs=eval(args.configs),
                                                    cache_path=args.save_path,
                                                    shm_manager=shm_manager)

    valid = graph_tensors["valid_idx"] if "valid_idx" in graph_tensors else None
    test = graph_tensors["test_idx"] if "test_idx" in graph_tensors else None
    compression_manager.register(
        graph_tensors['indptr'],
        graph_tensors['indices'],
        graph_tensors['train_idx'],
        graph_tensors['labels'],
        graph_tensors['features'],
        valid,
        test,
    )

    begin = time.time()
    fan_out = [int(fanout) for fanout in args.fan_out.split(",")]
    compression_manager.presampling(fan_out)
    torch.cuda.synchronize()
    print("presampling time: {:.3f} sec".format(time.time() - begin))

    begin = time.time()
    compression_manager.graph_reorder()
    torch.cuda.synchronize()
    print("graph_reorder time: {:.3f} sec".format(time.time() - begin))

    begin = time.time()
    compression_manager.compress()
    torch.cuda.synchronize()
    print("compress time: {:.3f} sec".format(time.time() - begin))

    begin = time.time()
    compression_manager.save_data()
    torch.cuda.synchronize()
    print("save data time: {:.3f} sec".format(time.time() - begin))
