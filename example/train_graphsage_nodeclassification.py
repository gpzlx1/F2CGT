import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import bifeat
from load_dataset import load_compressed_dataset
from models import SAGE
import argparse


def get_cache_nids(data, args, mem_capacity):
    g, seeds, metadata = data
    fan_out = [int(fanout) for fanout in args.fan_out.split(",")]

    # compute adj slope
    adj_slope = bifeat.cache.compute_adj_slope(g["indptr"],
                                               g["indices"],
                                               seeds,
                                               fan_out,
                                               args.batch_size,
                                               g["adj_hotness"],
                                               step=0.05,
                                               num_epochs=10,
                                               pin_adj=False)
    adj_space_tensor = bifeat.cache.compute_adj_space_tensor(
        g["indptr"], g["indptr"].dtype, g["indices"].dtype)

    # compute feature slope with a fake feature
    feat_slope = bifeat.cache.compute_feat_slope(torch.zeros(
        (metadata["num_nodes"], metadata["feature_dim"]), dtype=torch.float32),
                                                 g["feat_hotness"],
                                                 g["indptr"],
                                                 g["indices"],
                                                 seeds,
                                                 fan_out,
                                                 batch_size=args.batch_size,
                                                 step=0.2,
                                                 num_epochs=5,
                                                 pin_adj=False)
    compressed_features = g["features"]
    feat_hotness = g["feat_hotness"]
    num_feat_parts = len(compressed_features)
    feat_part_size = torch.tensor(metadata["part_size"], dtype=torch.long)
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
        feat_slope_list[i] = feat_slope * compressed_features[i].shape[1]

    feature_cache_nids_list, adj_cache_nids = bifeat.cache.cache_idx_select(
        feat_hotness_list, g["adj_hotness"], feat_slope_list, adj_slope,
        feat_space_list, adj_space_tensor, mem_capacity)
    feature_cache_nids_list = [nids.cuda() for nids in feature_cache_nids_list]
    adj_cache_nids = adj_cache_nids.cuda()

    return feature_cache_nids_list, adj_cache_nids


def run(rank, world_size, data, args):
    torch.cuda.set_device(rank)
    device = torch.device("cuda")
    dist.init_process_group('nccl',
                            'tcp://127.0.0.1:12347',
                            world_size=world_size,
                            rank=rank)

    tic = time.time()

    g, train_nids, metadata, codebooks = data

    train_part_size = (train_nids.shape[0] + world_size - 1) // world_size
    local_train_nids = train_nids[rank * train_part_size:(rank + 1) *
                                  train_part_size]
    fan_out = [int(fanout) for fanout in args.fan_out.split(",")]

    sampler = bifeat.cache.StructureCacheServer(g["indptr"], g["indices"],
                                                fan_out)
    feature_decompresser = bifeat.compression.Decompresser(
        metadata["feature_dim"], codebooks, metadata["methods"],
        metadata["part_size"])
    feature_server = bifeat.cache.CompressedFeatureCacheServer(
        g["features"], feature_decompresser)
    dataloader = bifeat.dataloading.SeedGenerator(local_train_nids,
                                                  args.batch_size,
                                                  shuffle=True)

    model = SAGE(metadata["feature_dim"], args.num_hidden,
                 metadata["num_classes"], len(fan_out), F.relu, args.dropout)
    model = model.to(device)
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[rank],
                                                output_device=rank)
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    epoch_time_log = []
    sample_time_log = []
    load_time_log = []
    forward_time_log = []
    backward_time_log = []
    update_time_log = []
    for epoch in range(args.num_epochs):
        sample_time = 0
        load_time = 0
        forward_time = 0
        backward_time = 0
        update_time = 0
        num_seeds = 0
        num_inputs = 0
        num_iters = 0

        epoch_tic = time.time()

        model.train()
        for it, seed_nids in enumerate(dataloader):

            num_iters += 1
            if args.breakdown:
                dist.barrier()
                torch.cuda.synchronize()

            tic = time.time()
            frontier, seeds, blocks = sampler.sample_neighbors(seed_nids)
            if args.breakdown:
                dist.barrier()
                torch.cuda.synchronize()
            sample_time += time.time() - tic

            num_seeds += seeds.shape[0]
            num_inputs += frontier.shape[0]
            tic = time.time()
            batch_inputs = feature_server[frontier]
            batch_labels = g["labels"][seeds.cpu()].cuda()
            if args.breakdown:
                dist.barrier()
                torch.cuda.synchronize()

            load_time += time.time() - tic
            tic = time.time()
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            if args.breakdown:
                dist.barrier()
                torch.cuda.synchronize()

            forward_time += time.time() - tic
            tic = time.time()
            optimizer.zero_grad()
            loss.backward()
            if args.breakdown:
                dist.barrier()
                torch.cuda.synchronize()

            backward_time += time.time() - tic
            tic = time.time()
            optimizer.step()
            if args.breakdown:
                dist.barrier()
                torch.cuda.synchronize()
            update_time += time.time() - tic

            if feature_server.no_cached and sampler.no_cached:
                mem_capacity = torch.cuda.mem_get_info(
                    torch.cuda.current_device(
                    ))[1] - torch.cuda.max_memory_allocated(
                    ) - args.reserved_mem * 1024 * 1024 * 1024
                print("Rank {} builds cache, GPU mem capacity = {:.3f} GB".
                      format(rank, mem_capacity / 1024 / 1024 / 1024))
                cache_tic = time.time()
                feature_cache_nids_list, adj_cache_nids = get_cache_nids(
                    (g, local_train_nids, metadata), args, mem_capacity)
                torch.cuda.empty_cache()
                feature_server.cache_data(feature_cache_nids_list)
                sampler.cache_data(adj_cache_nids)
                cache_toc = time.time()
                print("Rank {} builds cache time = {:.3f} sec".format(
                    rank, cache_toc - cache_tic))

        epoch_toc = time.time()

        for i in range(args.num_trainers):
            dist.barrier()
            if i == rank % args.num_trainers:
                timetable = ("=====================\n"
                             "Part {}, Epoch Time(s): {:.4f}\n"
                             "Sampling Time(s): {:.4f}\n"
                             "Loading Time(s): {:.4f}\n"
                             "Forward Time(s): {:.4f}\n"
                             "Backward Time(s): {:.4f}\n"
                             "Update Time(s): {:.4f}\n"
                             "#seeds: {}\n"
                             "#inputs: {}\n"
                             "#iterations: {}\n"
                             "=====================".format(
                                 rank,
                                 epoch_toc - epoch_tic,
                                 sample_time,
                                 load_time,
                                 forward_time,
                                 backward_time,
                                 update_time,
                                 num_seeds,
                                 num_inputs,
                                 num_iters,
                             ))
                print(timetable)
        sample_time_log.append(sample_time)
        load_time_log.append(load_time)
        forward_time_log.append(forward_time)
        backward_time_log.append(backward_time)
        update_time_log.append(update_time)
        epoch_time_log.append(epoch_toc - epoch_tic)

    avg_epoch_time = np.mean(epoch_time_log[2:])
    avg_sample_time = np.mean(sample_time_log[2:])
    avg_load_time = np.mean(load_time_log[2:])
    avg_forward_time = np.mean(forward_time_log[2:])
    avg_backward_time = np.mean(backward_time_log[2:])
    avg_update_time = np.mean(update_time_log[2:])

    for i in range(args.num_trainers):
        dist.barrier()
        if i == dist.get_rank() % args.num_trainers:
            timetable = ("=====================\n"
                         "Part {}, Avg Time:\n"
                         "Epoch Time(s): {:.4f}\n"
                         "Sampling Time(s): {:.4f}\n"
                         "Loading Time(s): {:.4f}\n"
                         "Forward Time(s): {:.4f}\n"
                         "Backward Time(s): {:.4f}\n"
                         "Update Time(s): {:.4f}\n"
                         "=====================".format(
                             dist.get_rank(),
                             avg_epoch_time,
                             avg_sample_time,
                             avg_load_time,
                             avg_forward_time,
                             avg_backward_time,
                             avg_update_time,
                         ))
            print(timetable)
    all_reduce_tensor = torch.tensor([0], device="cuda", dtype=torch.float32)

    all_reduce_tensor[0] = avg_epoch_time
    dist.all_reduce(all_reduce_tensor, dist.ReduceOp.SUM)
    all_reduce_epoch_time = all_reduce_tensor[0].item() / dist.get_world_size()

    all_reduce_tensor[0] = avg_sample_time
    dist.all_reduce(all_reduce_tensor, dist.ReduceOp.SUM)
    all_reduce_sample_time = all_reduce_tensor[0].item() / dist.get_world_size(
    )

    all_reduce_tensor[0] = avg_load_time
    dist.all_reduce(all_reduce_tensor, dist.ReduceOp.SUM)
    all_reduce_load_time = all_reduce_tensor[0].item() / dist.get_world_size()

    all_reduce_tensor[0] = avg_forward_time
    dist.all_reduce(all_reduce_tensor, dist.ReduceOp.SUM)
    all_reduce_forward_time = all_reduce_tensor[0].item(
    ) / dist.get_world_size()

    all_reduce_tensor[0] = avg_backward_time
    dist.all_reduce(all_reduce_tensor, dist.ReduceOp.SUM)
    all_reduce_backward_time = all_reduce_tensor[0].item(
    ) / dist.get_world_size()

    all_reduce_tensor[0] = avg_update_time
    dist.all_reduce(all_reduce_tensor, dist.ReduceOp.SUM)
    all_reduce_update_time = all_reduce_tensor[0].item() / dist.get_world_size(
    )

    if dist.get_rank() == 0:
        timetable = ("=====================\n"
                     "All reduce time:\n"
                     "Epoch Time(s): {:.4f}\n"
                     "Sampling Time(s): {:.4f}\n"
                     "Loading Time(s): {:.4f}\n"
                     "Forward Time(s): {:.4f}\n"
                     "Backward Time(s): {:.4f}\n"
                     "Update Time(s): {:.4f}\n"
                     "=====================".format(
                         all_reduce_epoch_time,
                         all_reduce_sample_time,
                         all_reduce_load_time,
                         all_reduce_forward_time,
                         all_reduce_backward_time,
                         all_reduce_update_time,
                     ))
        print(timetable)


def main(args):
    g, metadata, codebooks = load_compressed_dataset(args.root, args.dataset)
    train_nids = g.pop("train_idx")
    train_nids = train_nids[torch.randperm(train_nids.shape[0])]
    data = g, train_nids, metadata, codebooks

    import torch.multiprocessing as mp
    mp.spawn(run,
             args=(args.num_trainers, data, args),
             nprocs=args.num_trainers)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        "Train nodeclassification GraphSAGE model")
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
    argparser.add_argument("--lr", type=float, default=0.003)
    argparser.add_argument("--dropout", type=float, default=0.5)
    argparser.add_argument("--batch-size", type=int, default=512)
    argparser.add_argument("--batch-size-eval", type=int, default=100000)
    argparser.add_argument("--log-every", type=int, default=20)
    argparser.add_argument("--eval-every", type=int, default=5)
    argparser.add_argument("--fan-out", type=str, default="25,10")
    argparser.add_argument("--num-hidden", type=int, default=256)
    argparser.add_argument("--num-epochs", type=int, default=20)
    argparser.add_argument("--breakdown", action="store_true")
    argparser.add_argument("--reserved-mem",
                           type=float,
                           default=1.0,
                           help="reserverd GPU memory size, unit: GB")
    args = argparser.parse_args()
    print(args)
    main(args)
