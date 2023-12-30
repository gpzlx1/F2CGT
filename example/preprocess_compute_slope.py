import torch
import torch.distributed as dist
import time
import bifeat
import argparse
from bifeat.shm import dtype_sizeof
from load_dataset import load_compressed_dataset

torch.manual_seed(25)


def get_cache_nids(data, args, mem_capacity, save_memory_mode=False):
    g, metadata = data

    adj_space_tensor = bifeat.cache.compute_adj_space_tensor(
        g["indptr"], g["indptr"].dtype, g["indices"].dtype)

    features = [g["core_features"], g["features"]]
    feat_hotness = g["feat_hotness"]
    num_feat_parts = len(features)
    feat_part_size = torch.tensor(
        [g["core_idx"].shape[0], metadata["num_nodes"]], dtype=torch.long)
    feat_part_range = torch.zeros(num_feat_parts + 1, dtype=torch.long)
    feat_part_range[1:] = torch.cumsum(feat_part_size, dim=0)
    feat_hotness_list = [feat_hotness[g["core_idx"]], feat_hotness]
    feat_space_list = [0 for _ in range(num_feat_parts)]
    feat_slope_list = [0.0 for _ in range(num_feat_parts)]
    for i in range(num_feat_parts):
        feat_space_list[i] = bifeat.cache.compute_feat_sapce(
            features[i].shape[1], features[i].dtype)
        feat_slope_list[i] = args.feat_slope / 4 * dtype_sizeof(
            features[i].dtype) * features[i].shape[1]

    feature_cache_nids_list, adj_cache_nids = bifeat.cache.cache_idx_select(
        feat_hotness_list, g["adj_hotness"], feat_slope_list, args.adj_slope,
        feat_space_list, adj_space_tensor, mem_capacity)
    
    
    if save_memory_mode:
        mask = g["adj_hotness"][adj_cache_nids] > 0
        adj_cache_nids = adj_cache_nids[mask]
        feature_cache_nids_list = [nids[feat_hotness_list[i][nids] > 0] for i, nids in enumerate(feature_cache_nids_list)]
    
    
    adj_cache_nids = adj_cache_nids.cuda()
    feature_cache_nids_list = [nids.cuda() for nids in feature_cache_nids_list]
    
    

    return feature_cache_nids_list, adj_cache_nids


def run(rank, world_size, data, args):
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl',
                            'tcp://127.0.0.1:12347',
                            world_size=world_size,
                            rank=rank)

    g, train_nids = data

    tic = time.time()

    train_part_size = (train_nids.shape[0] + world_size - 1) // world_size
    seeds = train_nids[rank * train_part_size:(rank + 1) * train_part_size]
    fan_out = [int(fanout) for fanout in args.fan_out.split(",")]
    adj_slope = bifeat.cache.compute_adj_slope(g["indptr"],
                                               g["indices"],
                                               seeds,
                                               fan_out,
                                               args.batch_size,
                                               g["adj_hotness"],
                                               step=args.adj_step,
                                               num_epochs=args.adj_epochs)
    feat_slope = bifeat.cache.compute_feat_slope(g["features"],
                                                 g["feat_hotness"],
                                                 g["indptr"],
                                                 g["indices"],
                                                 seeds,
                                                 fan_out,
                                                 batch_size=args.batch_size,
                                                 step=args.feat_step,
                                                 num_epochs=args.feat_epochs)
    slope_tensor = torch.tensor([adj_slope, feat_slope], device="cuda")
    dist.all_reduce(slope_tensor, dist.ReduceOp.SUM)
    slope_tensor = slope_tensor / world_size

    adj_slope, feat_slope = slope_tensor[0].item(), slope_tensor[1].item()

    dist.barrier()
    toc = time.time()

    if rank == 0:
        print("====================================")
        print("Graph typo slope: {:.6f}".format(adj_slope))
        print("Feature slope: {:.6f}".format(feat_slope))
        print("Compute slopes time: {:.3f} sec".format(toc - tic))
        print("====================================")


def main(args):
    g, metadata, _ = load_compressed_dataset(args.root,
                                             args.dataset,
                                             with_feature=False,
                                             with_test=False,
                                             with_valid=False)
    train_nids = g.pop("train_idx")
    train_nids = train_nids[torch.randperm(train_nids.shape[0])]

    # compute slope with fake feature
    g["features"] = torch.zeros((metadata["num_nodes"], 128),
                                dtype=torch.float32)

    data = g, train_nids

    import torch.multiprocessing as mp
    mp.spawn(run,
             args=(args.num_trainers, data, args),
             nprocs=args.num_trainers)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        "Compute graph typo and feature slopes")
    argparser.add_argument(
        "--dataset",
        type=str,
        default="ogbn-products",
        help="datasets: ogbn-products, ogbn-papers100M",
    )
    argparser.add_argument("--root",
                           type=str,
                           default="/data/ogbn_products/compressed")
    argparser.add_argument(
        "--num-trainers",
        type=int,
        default=2,
        help=
        "number of trainers participated in the compress, no greater than available GPUs num"
    )
    argparser.add_argument("--batch-size", type=int, default=1000)
    argparser.add_argument("--fan-out", type=str, default="5,10,15")
    argparser.add_argument("--adj-step", default=0.05, type=float)
    argparser.add_argument("--adj-epochs", default=10, type=int)
    argparser.add_argument("--feat-step", default=0.2, type=float)
    argparser.add_argument("--feat-epochs", default=5, type=int)
    args = argparser.parse_args()
    print(args)
    main(args)
