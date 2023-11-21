import argparse
import time

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl
from models import LinkPredSAGE
import torch.distributed as dist
import torch
from pagraph import FeatureCache
from load_graph import load_ogb_link_pred

torch.manual_seed(25)


def presampling(g, dataloader, train_data, num_epochs=1):
    presampling_heat = torch.zeros((g.num_nodes(), ), dtype=torch.float32)
    model, loss_fcn, optimizer = train_data
    tic = time.time()
    for epoch in range(num_epochs):
        with model.join():
            # run some epochs to count presampling heat and max allocated cuda memory
            for step, (input_nodes, pair_graph, neg_pair_graph,
                       blocks) in enumerate(dataloader):
                input_nodes = input_nodes.cpu()
                presampling_heat[input_nodes] += 1
                batch_inputs = g.ndata["features"][input_nodes].to("cuda")
                pos_score, neg_score = model(pair_graph, neg_pair_graph,
                                             blocks, batch_inputs)
                score = torch.cat([pos_score, neg_score])
                pos_label = torch.ones_like(pos_score)
                neg_label = torch.zeros_like(neg_score)
                labels = torch.cat([pos_label, neg_label])
                loss = loss_fcn(score, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    toc = time.time()

    presampling_heat_accessed = presampling_heat[presampling_heat > 0]
    mem_used = torch.cuda.max_memory_allocated()
    info = "========================================\n"
    info += "Rank {} presampling info:\n".format(torch.cuda.current_device())
    info += "Presampling done, max: {:.3f} min: {:.3f} avg: {:.3f}\n".format(
        torch.max(presampling_heat_accessed).item(),
        torch.min(presampling_heat_accessed).item(),
        torch.mean(presampling_heat_accessed).item())
    info += "Max allocated cuda mem: {:.3f} GB\n".format(mem_used / 1024 /
                                                         1024 / 1024)
    info += "Presampling time: {:.3f} s\n".format(toc - tic)
    info += "========================================"
    print(info)

    return presampling_heat, mem_used


def run(rank, world_size, data, args):
    torch.cuda.set_device(rank)
    device = torch.device("cuda")
    dist.init_process_group('nccl',
                            'tcp://127.0.0.1:12347',
                            world_size=world_size,
                            rank=rank)
    # Unpack data
    train_eid, g, reverse_eids = data
    shuffle = True
    sampler = dgl.dataloading.NeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(",")])
    sampler = dgl.dataloading.as_edge_prediction_sampler(
        sampler,
        exclude="reverse_id",
        reverse_eids=reverse_eids.cuda(),
        negative_sampler=dgl.dataloading.negative_sampler.Uniform(1),
    )
    dataloader = dgl.dataloading.DataLoader(g,
                                            train_eid,
                                            sampler,
                                            device=device,
                                            batch_size=args.batch_size,
                                            shuffle=shuffle,
                                            drop_last=False,
                                            num_workers=0,
                                            use_ddp=True,
                                            use_uva=True)
    # Define model and optimizer
    model = LinkPredSAGE(g.ndata["features"].shape[1], args.num_hidden,
                         args.num_layers, F.relu)
    model = model.to(device)
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[rank],
                                                output_device=rank)
    loss_fcn = F.binary_cross_entropy_with_logits
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    presampling_heat, cuda_mem_used = presampling(g, dataloader,
                                                  (model, loss_fcn, optimizer))
    reserved_mem = 1.0 * 1024 * 1024 * 1024
    gpu_capacity = int(
        torch.cuda.mem_get_info(torch.cuda.current_device())[1] -
        cuda_mem_used - reserved_mem)
    features = g.ndata.pop("features")
    feature_cache = FeatureCache(features)
    feature_cache.create_cache(gpu_capacity, presampling_heat)

    iter_tput = []
    epoch_time_log = []
    sample_time_log = []
    load_time_log = []
    forward_time_log = []
    backward_time_log = []
    update_time_log = []
    epoch = 0
    for epoch in range(args.num_epochs):
        tic = time.time()

        sample_time = 0
        load_time = 0
        forward_time = 0
        backward_time = 0
        update_time = 0
        num_seeds = 0
        num_inputs = 0

        with model.join():
            # Loop over the dataloader to sample the computation dependency
            # graph as a list of blocks.
            step_time = []
            if args.breakdown:
                dist.barrier()
                torch.cuda.synchronize()
            tic = time.time()
            tic_step = time.time()
            for step, (input_nodes, pair_graph, neg_pair_graph,
                       blocks) in enumerate(dataloader):
                if args.breakdown:
                    dist.barrier()
                    torch.cuda.synchronize()
                sample_time += time.time() - tic_step

                load_begin = time.time()
                batch_inputs = feature_cache[input_nodes]
                num_seeds += len(blocks[-1].dstdata[dgl.NID])
                num_inputs += len(blocks[0].srcdata[dgl.NID])
                if args.breakdown:
                    dist.barrier()
                    torch.cuda.synchronize()
                load_time += time.time() - load_begin

                forward_start = time.time()
                pos_score, neg_score = model(pair_graph, neg_pair_graph,
                                             blocks, batch_inputs)
                score = torch.cat([pos_score, neg_score])
                pos_label = torch.ones_like(pos_score)
                neg_label = torch.zeros_like(neg_score)
                labels = torch.cat([pos_label, neg_label])
                loss = loss_fcn(score, labels)
                if args.breakdown:
                    dist.barrier()
                    torch.cuda.synchronize()
                forward_time += time.time() - forward_start

                backward_begin = time.time()
                optimizer.zero_grad()
                loss.backward()
                if args.breakdown:
                    dist.barrier()
                    torch.cuda.synchronize()
                backward_time += time.time() - backward_begin

                update_start = time.time()
                optimizer.step()
                if args.breakdown:
                    dist.barrier()
                    torch.cuda.synchronize()
                update_time += time.time() - update_start

                step_t = time.time() - tic_step
                step_time.append(step_t)
                iter_tput.append(len(blocks[-1].dstdata[dgl.NID]) / step_t)
                tic_step = time.time()

        toc = time.time()
        epoch += 1

        for i in range(args.num_gpus):
            dist.barrier()
            if i == dist.get_rank() % args.num_gpus:
                timetable = ("=====================\n"
                             "Part {}, Epoch Time(s): {:.4f}\n"
                             "Sampling Time(s): {:.4f}\n"
                             "Loading Time(s): {:.4f}\n"
                             "Forward Time(s): {:.4f}\n"
                             "Backward Time(s): {:.4f}\n"
                             "Update Time(s): {:.4f}\n"
                             "#seeds: {}\n"
                             "#inputs: {}\n"
                             "=====================".format(
                                 dist.get_rank(),
                                 toc - tic,
                                 sample_time,
                                 load_time,
                                 forward_time,
                                 backward_time,
                                 update_time,
                                 num_seeds,
                                 num_inputs,
                             ))
                print(timetable)
        sample_time_log.append(sample_time)
        load_time_log.append(load_time)
        forward_time_log.append(forward_time)
        backward_time_log.append(backward_time)
        update_time_log.append(update_time)
        epoch_time_log.append(toc - tic)

    avg_epoch_time = np.mean(epoch_time_log[2:])
    avg_sample_time = np.mean(sample_time_log[2:])
    avg_load_time = np.mean(load_time_log[2:])
    avg_forward_time = np.mean(forward_time_log[2:])
    avg_backward_time = np.mean(backward_time_log[2:])
    avg_update_time = np.mean(update_time_log[2:])

    for i in range(args.num_gpus):
        th.distributed.barrier()
        if i == th.distributed.get_rank() % args.num_gpus:
            timetable = ("=====================\n"
                         "Part {}, Avg Time:\n"
                         "Epoch Time(s): {:.4f}\n"
                         "Sampling Time(s): {:.4f}\n"
                         "Loading Time(s): {:.4f}\n"
                         "Forward Time(s): {:.4f}\n"
                         "Backward Time(s): {:.4f}\n"
                         "Update Time(s): {:.4f}\n"
                         "=====================".format(
                             th.distributed.get_rank(),
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
    g, reverse_eids = load_ogb_link_pred(args.dataset, args.root)
    num_edges = g.num_nodes()
    num_train = int(num_edges * args.seeds_rate)
    train_eid = torch.randperm(num_edges)[:num_train]
    print("Train: {}".format(train_eid.numel()))

    data = train_eid, g, reverse_eids

    import torch.multiprocessing as mp
    mp.spawn(run, args=(args.num_gpus, data, args), nprocs=args.num_gpus)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCN")
    parser.add_argument("--n_classes",
                        type=int,
                        default=0,
                        help="the number of classes")
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=8,
        help="the number of GPU device. Use -1 for CPU training",
    )
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--num_hidden", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--fan_out", type=str, default="5,10,15")
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--batch_size_eval", type=int, default=100000)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument(
        "--dataset",
        type=str,
        default="ogbl-ppa",
        help="datasets: ogbl-citation2",
    )
    parser.add_argument("--root", type=str, default="/data")
    parser.add_argument("--breakdown", action="store_true")
    parser.add_argument("--seeds_rate", default=0.1, type=float)
    args = parser.parse_args()

    print(args)
    main(args)