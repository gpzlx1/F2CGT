import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import bifeat
from models import SAGE, compute_acc, evaluate
from preprocess_compute_slope import get_cache_nids
import argparse
import os

torch.manual_seed(25)


def run(rank, world_size, data, args):
    torch.cuda.set_device(rank % args.num_trainers)
    device = torch.cuda.current_device()
    tic = time.time()

    g, train_nids, metadata, codebooks = data

    train_part_size = (train_nids.shape[0] + world_size - 1) // world_size
    local_train_nids = train_nids[rank * train_part_size:(rank + 1) *
                                  train_part_size]
    fan_out = [int(fanout) for fanout in args.fan_out.split(",")]

    sampler = bifeat.cache.StructureCacheServer(g["indptr"],
                                                g["indices"],
                                                fan_out,
                                                count_hit=True)
    feature_decompresser = bifeat.compression.Decompresser(
        metadata["feature_dim"], codebooks, metadata["methods"],
        [g["core_idx"].shape[0], metadata["num_nodes"]])
    feature_server = bifeat.cache.FeatureLoadServer(g["core_features"],
                                                    g["core_idx"],
                                                    g["features"],
                                                    feature_decompresser,
                                                    count_hit=True)
    dataloader = bifeat.dataloading.SeedGenerator(local_train_nids,
                                                  args.batch_size,
                                                  shuffle=True)

    model = SAGE(metadata["feature_dim"], args.num_hidden,
                 metadata["num_classes"], len(fan_out), F.relu, args.dropout)
    model = model.to(device)
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[rank % args.num_trainers],
        output_device=rank % args.num_trainers)
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    epoch_time_log = []
    sample_time_log = []
    load_time_log = []
    forward_time_log = []
    backward_time_log = []
    update_time_log = []
    num_uncached_sample_seeds_log = []
    num_uncached_sample_neighbors_log = []
    num_uncached_feat_frontier_log = []
    num_uncached_feat_seeds_log = []

    for epoch in range(args.num_epochs):
        sample_time = 0
        load_time = 0
        forward_time = 0
        backward_time = 0
        update_time = 0
        num_seeds = 0
        num_inputs = 0
        num_iters = 0
        num_uncached_sample_seeds = 0
        num_uncached_sample_neighbors = 0
        num_uncached_feat_frontier = 0
        num_uncached_feat_seeds = 0

        torch.cuda.synchronize()
        epoch_tic = time.time()

        model.train()
        for it, seed_nids in enumerate(dataloader):
            num_iters += 1
            torch.cuda.synchronize()

            tic = time.time()
            frontier, seeds, blocks = sampler.sample_neighbors(seed_nids)
            torch.cuda.synchronize()
            sample_time += time.time() - tic

            num_seeds += seeds.shape[0]
            num_inputs += frontier.shape[0]
            tic = time.time()
            batch_inputs = feature_server[frontier, seeds.shape[0]]
            batch_labels = g["labels"][seeds.cpu()].long().cuda()
            torch.cuda.synchronize()

            load_time += time.time() - tic
            tic = time.time()
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            torch.cuda.synchronize()

            forward_time += time.time() - tic
            tic = time.time()
            optimizer.zero_grad()
            loss.backward()
            torch.cuda.synchronize()

            backward_time += time.time() - tic
            tic = time.time()
            optimizer.step()
            torch.cuda.synchronize()
            update_time += time.time() - tic

            if (it + 1) % args.log_every == 0:
                acc = compute_acc(batch_pred, batch_labels)
                gpu_mem_alloc = (torch.cuda.max_memory_allocated() /
                                 1000000 if torch.cuda.is_available() else 0)
                print("Part {} | Epoch {:05d} | Step {:05d} | Loss {:.4f} | "
                      "Train Acc {:.4f} | GPU {:.1f} MB".format(
                          rank, epoch, it + 1, loss.item(), acc.item(),
                          gpu_mem_alloc))
                train_acc_tensor = torch.tensor([acc.item()]).cuda()
                dist.all_reduce(train_acc_tensor, dist.ReduceOp.SUM)
                train_acc_tensor /= world_size
                if rank == 0:
                    print("Avg train acc {:.4f}".format(
                        train_acc_tensor[0].item()))

        epoch_toc = time.time()
        
        # build cache
        if not feature_server.cache_built and not sampler.cache_built:
            print("GPU memory usage",
                  torch.cuda.memory_allocated() / 1024 / 1024 / 1024,
                  torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024,
                  torch.cuda.memory_reserved() / 1024 / 1024 / 1024,
                  torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024)

            mem_capacity = torch.cuda.mem_get_info(torch.cuda.current_device(
            ))[1] - torch.cuda.max_memory_reserved(
            ) - args.reserved_mem * 1024 * 1024 * 1024 - g["core_idx"].shape[
                0] * 12 * 4
            mem_capacity = max(mem_capacity, 0)
            
            if mem_capacity > 0:
                print("Rank {} builds cache, GPU mem capacity = {:.3f} GB".format(
                    rank, mem_capacity / 1024 / 1024 / 1024))
                cache_tic = time.time()
                feature_cache_nids_list, adj_cache_nids = get_cache_nids(
                    (g, metadata), args, mem_capacity)
                feature_server.cache_core_feature(feature_cache_nids_list[0])
                feature_server.cache_feature(feature_cache_nids_list[1])
                sampler.cache_data(adj_cache_nids)
                cache_toc = time.time()
                print("Rank {} builds cache time = {:.3f} sec".format(
                    rank, cache_toc - cache_tic))
                
        sample_access_times, sample_hit_times, _ = sampler.get_hit_rates()
        sampler.reset_hit_counts()
        for l in range(len(fan_out)):
            num_layer_uncached_seeds = sample_access_times[
                l] - sample_hit_times[l]
            num_uncached_sample_seeds += num_layer_uncached_seeds
            num_uncached_sample_neighbors += num_layer_uncached_seeds * fan_out[
                l]
        (
            _,
            feat_seeds_hit_times,
            _,
            feat_frontier_hit_times,
            _,
        ) = feature_server.get_hit_rates()
        feature_server.reset_hit_counts()
        num_uncached_feat_frontier += num_inputs - num_seeds - feat_frontier_hit_times
        num_uncached_feat_seeds += num_seeds - feat_seeds_hit_times

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
                             "num_uncached_sample_seeds: {}\n"
                             "num_uncached_sample_neighbors: {}\n"
                             "num_uncached_feat_frontier: {}\n"
                             "num_uncached_feat_seeds: {}\n"
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
                                 num_uncached_sample_seeds,
                                 num_uncached_sample_neighbors,
                                 num_uncached_feat_frontier,
                                 num_uncached_feat_seeds,
                             ))
                print(timetable)
        sample_time_log.append(sample_time)
        load_time_log.append(load_time)
        forward_time_log.append(forward_time)
        backward_time_log.append(backward_time)
        update_time_log.append(update_time)
        epoch_time_log.append(epoch_toc - epoch_tic)
        num_uncached_sample_neighbors_log.append(num_uncached_sample_neighbors)
        num_uncached_sample_seeds_log.append(num_uncached_sample_seeds)
        num_uncached_feat_frontier_log.append(num_uncached_feat_frontier)
        num_uncached_feat_seeds_log.append(num_uncached_feat_seeds)

        if (epoch + 1) % args.eval_every == 0:
            tic = time.time()
            val_acc, test_acc = evaluate(
                model.module,
                g,
                feature_server,
                g["labels"],
                g["valid_idx"],
                g["test_idx"],
                args.batch_size_eval,
            )
            print("Part {}, Val Acc {:.4f}, Test Acc {:.4f}, time: {:.4f}".
                  format(rank, val_acc, test_acc,
                         time.time() - tic))
            acc_tensor = torch.tensor([val_acc, test_acc]).cuda()
            dist.all_reduce(acc_tensor, dist.ReduceOp.SUM)
            acc_tensor /= world_size
            if rank == 0:
                print("All parts avg val acc {:.4f}, test acc {:.4f}".format(
                    acc_tensor[0].item(), acc_tensor[1].item()))
            feature_server.reset_hit_counts()

    avg_epoch_time = np.mean(epoch_time_log[2:])
    avg_sample_time = np.mean(sample_time_log[2:])
    avg_load_time = np.mean(load_time_log[2:])
    avg_forward_time = np.mean(forward_time_log[2:])
    avg_backward_time = np.mean(backward_time_log[2:])
    avg_update_time = np.mean(update_time_log[2:])

    for i in range(args.num_trainers):
        dist.barrier()
        if i == rank % args.num_trainers:
            timetable = ("=====================\n"
                         "Part {}, Avg Time:\n"
                         "Epoch Time(s): {:.4f}\n"
                         "Sampling Time(s): {:.4f}\n"
                         "Loading Time(s): {:.4f}\n"
                         "Forward Time(s): {:.4f}\n"
                         "Backward Time(s): {:.4f}\n"
                         "Update Time(s): {:.4f}\n"
                         "num_uncached_sample_seeds: {}\n"
                         "num_uncached_sample_neighbors: {}\n"
                         "num_uncached_feat_frontier: {}\n"
                         "num_uncached_feat_seeds: {}\n"
                         "=====================".format(
                             rank,
                             avg_epoch_time,
                             avg_sample_time,
                             avg_load_time,
                             avg_forward_time,
                             avg_backward_time,
                             avg_update_time,
                             np.mean(num_uncached_sample_seeds_log[2:]),
                             np.mean(num_uncached_sample_neighbors_log[2:]),
                             np.mean(num_uncached_feat_frontier_log[2:]),
                             np.mean(num_uncached_feat_seeds_log[2:]),
                         ))
            print(timetable)
    all_reduce_tensor = torch.tensor([0], device="cuda", dtype=torch.float32)

    all_reduce_tensor[0] = avg_epoch_time
    dist.all_reduce(all_reduce_tensor, dist.ReduceOp.SUM)
    all_reduce_epoch_time = all_reduce_tensor[0].item() / world_size

    all_reduce_tensor[0] = avg_sample_time
    dist.all_reduce(all_reduce_tensor, dist.ReduceOp.SUM)
    all_reduce_sample_time = all_reduce_tensor[0].item() / world_size

    all_reduce_tensor[0] = avg_load_time
    dist.all_reduce(all_reduce_tensor, dist.ReduceOp.SUM)
    all_reduce_load_time = all_reduce_tensor[0].item() / world_size

    all_reduce_tensor[0] = avg_forward_time
    dist.all_reduce(all_reduce_tensor, dist.ReduceOp.SUM)
    all_reduce_forward_time = all_reduce_tensor[0].item() / world_size

    all_reduce_tensor[0] = avg_backward_time
    dist.all_reduce(all_reduce_tensor, dist.ReduceOp.SUM)
    all_reduce_backward_time = all_reduce_tensor[0].item() / world_size

    all_reduce_tensor[0] = avg_update_time
    dist.all_reduce(all_reduce_tensor, dist.ReduceOp.SUM)
    all_reduce_update_time = all_reduce_tensor[0].item() / world_size

    if rank == 0:
        timetable = ("=====================\n"
                     "All reduce time:\n"
                     "Throughput(seeds/sec): {:.4f}\n"
                     "Epoch Time(s): {:.4f}\n"
                     "Sampling Time(s): {:.4f}\n"
                     "Loading Time(s): {:.4f}\n"
                     "Forward Time(s): {:.4f}\n"
                     "Backward Time(s): {:.4f}\n"
                     "Update Time(s): {:.4f}\n"
                     "=====================".format(
                         train_nids.shape[0] / all_reduce_epoch_time,
                         all_reduce_epoch_time,
                         all_reduce_sample_time,
                         all_reduce_load_time,
                         all_reduce_forward_time,
                         all_reduce_backward_time,
                         all_reduce_update_time,
                     ))
        print(timetable)


def main(args):
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    num_machines = world_size // args.num_trainers
    # assert world_size == args.num_trainers
    omp_thread_num = os.cpu_count() // args.num_trainers
    torch.cuda.set_device(rank % args.num_trainers)
    torch.set_num_threads(omp_thread_num)
    print("Set device to {} and cpu threads num {}".format(
        rank, omp_thread_num))
    assert args.feat_slope is not None
    assert args.adj_slope is not None
    shm_manager = bifeat.shm.ShmManager(rank,
                                        args.num_trainers,
                                        args.root,
                                        args.dataset,
                                        pin_memory=False)
    if args.dataset == "friendster":
        g, metadata, codebooks = shm_manager.load_compressed_dataset(
            with_valid=False, with_test=False)
    else:
        g, metadata, codebooks = shm_manager.load_compressed_dataset(
            with_valid=True, with_test=True)

    cheif_ranks_list = [i * args.num_trainers for i in range(num_machines)]
    cheif_group = dist.new_group(cheif_ranks_list)

    train_nids = g.pop("train_idx")
    if rank % args.num_trainers == 0:
        train_nids_comm_buffer = torch.empty_like(train_nids)
    if rank == 0:
        train_nids_comm_buffer[:] = train_nids
    dist.barrier()
    if rank == 0:
        train_nids_comm_buffer = train_nids_comm_buffer[torch.randperm(
            train_nids_comm_buffer.shape[0])]
    if rank % args.num_trainers == 0:
        train_nids_comm_buffer = train_nids_comm_buffer.cuda()
        dist.broadcast(train_nids_comm_buffer, 0, group=cheif_group)
        train_nids_comm_buffer = train_nids_comm_buffer.cpu()
        torch.cuda.empty_cache()
    if rank % args.num_trainers == 0:
        train_nids[:] = train_nids_comm_buffer
        del train_nids_comm_buffer
    dist.barrier()
    print(train_nids)

    data = g, train_nids, metadata, codebooks

    if rank == 0:
        print("Core feature: dim {} dtype {}".format(
            g["core_features"].shape[1], g["core_features"].dtype))
        print("Frontier feature: dim {} dtype {}".format(
            g["features"].shape[1], g["features"].dtype))

    run(rank, world_size, data, args)

    dist.destroy_process_group(cheif_group)


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
    argparser.add_argument("--batch-size", type=int, default=1000)
    argparser.add_argument("--batch-size-eval", type=int, default=100000)
    argparser.add_argument("--log-every", type=int, default=20)
    argparser.add_argument("--eval-every", type=int, default=5)
    argparser.add_argument("--fan-out", type=str, default="5,10,15")
    argparser.add_argument("--num-hidden", type=int, default=256)
    argparser.add_argument("--num-epochs", type=int, default=20)
    argparser.add_argument("--breakdown", action="store_true")
    argparser.add_argument("--reserved-mem",
                           type=float,
                           default=1.0,
                           help="reserverd GPU memory size, unit: GB")
    argparser.add_argument("--feat-slope", type=float, default=None)
    argparser.add_argument("--adj-slope", type=float, default=None)
    args = argparser.parse_args()
    print(args)
    main(args)
