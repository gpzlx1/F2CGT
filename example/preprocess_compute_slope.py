import torch
import torch.distributed as dist
import time
import bifeat
import argparse
from bifeat.shm import dtype_sizeof
from load_dataset import load_compressed_dataset

torch.manual_seed(25)


def get_cache_nids(data, args, mem_capacity):
    g, metadata = data
    adj_space_tensor = bifeat.cache.compute_adj_space_tensor(
        g["indptr"], g["indptr"].dtype, g["indices"].dtype)
    compressed_features = [g["features"]]
    feat_hotness = g["feat_hotness"]
    num_feat_parts = len(compressed_features)
    feat_part_size = torch.tensor([metadata["num_nodes"]], dtype=torch.long)
    feat_part_range = torch.zeros(num_feat_parts + 1, dtype=torch.long)
    feat_part_range[1:] = torch.cumsum(feat_part_size, dim=0)
    feat_hotness_list = [None for _ in range(num_feat_parts)]
    feat_space_list = [0 for _ in range(num_feat_parts)]
    feat_slope_list = [0.0 for _ in range(num_feat_parts)]
    for i in range(num_feat_parts):
        feat_hotness_list[i] = feat_hotness[
            feat_part_range[i]:feat_part_range[i + 1]]
        feat_space_list[i] = bifeat.cache.compute_feat_sapce(
            compressed_features[i].shape[1], compressed_features[i].dtype)
        feat_slope_list[i] = args.feat_slope / 4 * dtype_sizeof(
            compressed_features[i].dtype) * compressed_features[i].shape[1]

    feature_cache_nids_list, adj_cache_nids = bifeat.cache.cache_idx_select(
        feat_hotness_list, g["adj_hotness"], feat_slope_list, args.adj_slope,
        feat_space_list, adj_space_tensor, mem_capacity)
    feature_cache_nids_list = [nids.cuda() for nids in feature_cache_nids_list]
    adj_cache_nids = adj_cache_nids.cuda()

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
                                               step=0.05,
                                               num_epochs=10)
    feat_slope = bifeat.cache.compute_feat_slope(g["features"],
                                                 g["feat_hotness"],
                                                 g["indptr"],
                                                 g["indices"],
                                                 seeds,
                                                 fan_out,
                                                 batch_size=args.batch_size,
                                                 step=0.2,
                                                 num_epochs=5)
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
    argparser.add_argument("--batch-size", type=int, default=1024)
    argparser.add_argument("--fan-out", type=str, default="5,10,15")
    args = argparser.parse_args()
    print(args)
    main(args)
