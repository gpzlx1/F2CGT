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
    argparser.add_argument("--sample-sizes", type=str, default="100000,100000")
    argparser.add_argument("--compress-batch-sizes",
                           type=str,
                           default="1000000,1000000")
    args = argparser.parse_args()
    print(args)

    total_start = time.time()

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
    if args.dataset == "friendster":
        graph_tensors, meta_data = shm_manager.load_dataset(with_feature=True,
                                                            with_valid=False,
                                                            with_test=False)
        fake_feat_dim = None
        fake_feat_dtype = None
    elif args.dataset == "mag240m":
        graph_tensors, meta_data = shm_manager.load_dataset(with_feature=False,
                                                            with_valid=True,
                                                            with_test=True)
        fake_feat_dim = meta_data["feature_dim"]
        fake_feat_dtype = torch.float16
    else:
        graph_tensors, meta_data = shm_manager.load_dataset(with_feature=True,
                                                            with_valid=True,
                                                            with_test=True)
        fake_feat_dim = None
        fake_feat_dtype = None
    torch.cuda.synchronize()
    print("Load dataset time: {:.3f} sec".format(time.time() - begin))

    methods = [method for method in args.methods.split(",")]
    sample_sizes = [int(size) for size in args.sample_sizes.split(",")]
    compress_batch_sizes = [
        int(size) for size in args.compress_batch_sizes.split(",")
    ]
    compression_manager = bifeat.CompressionManager(
        methods=methods,
        configs=eval(args.configs),
        cache_path=args.save_path,
        shm_manager=shm_manager,
        sample_sizes=sample_sizes,
        compress_batch_sizes=compress_batch_sizes)

    valid = graph_tensors["valid_idx"] if "valid_idx" in graph_tensors else None
    test = graph_tensors["test_idx"] if "test_idx" in graph_tensors else None
    features = graph_tensors[
        "features"] if "features" in graph_tensors else None

    compression_manager.register(graph_tensors['indptr'],
                                 graph_tensors['indices'],
                                 graph_tensors['train_idx'],
                                 graph_tensors['labels'],
                                 features,
                                 graph_tensors['core_idx'],
                                 valid,
                                 test,
                                 fake_feat_dim=fake_feat_dim,
                                 fake_feat_dtype=fake_feat_dtype)

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

    total_end = time.time()
    print("total time: {:.3f} sec".format(total_end - total_start))

    begin = time.time()
    compression_manager.save_data()
    torch.cuda.synchronize()
    print("save data time: {:.3f} sec".format(time.time() - begin))
