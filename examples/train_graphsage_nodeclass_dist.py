import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import shmtensor
from load_dataset import create_local_group, dist_load_compress_data
from models import SAGE, compute_acc, GAT
from dataloader import allocate
import argparse
import dgl
from compress import CompressionManager
from process_compress import load_shm_tensor, load_meta

torch.manual_seed(25)


def presampling(dataloader, num_nodes, num_epochs=1):
    feature_hotness = torch.zeros((num_nodes, ), dtype=torch.float32)
    sampling_hotness = torch.zeros((num_nodes, ), dtype=torch.float32)
    seeds_hotness = torch.zeros((num_nodes, ), dtype=torch.float32)
    for epoch in range(num_epochs):
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            for block in blocks:
                layer_seeds = block.ndata[dgl.NID]["_N"][block.dstnodes()]
                sampling_hotness[layer_seeds] += 1
            feature_hotness[input_nodes] += 1
            seeds_hotness[seeds] += 1
    return sampling_hotness, feature_hotness, seeds_hotness


def run(args, data, compress_data, g, seeds):

    # unpack data
    labels = data['labels']

    train_nids, val_nids, test_nids = seeds

    shmtensor.capi.pin_memory(labels)

    cm = CompressionManager(compress_data, args.fusion)

    # create model
    if args.model == "sage":
        model = SAGE(compress_data['feat_dim'], args.num_hidden,
                     data['num_classes'], len(args.fan_out), F.relu,
                     args.dropout, cm)
    elif args.model == "gat":
        heads = [args.num_heads for _ in range(len(args.fan_out) - 1)]
        heads.append(1)
        num_hidden = args.num_hidden // args.num_heads
        model = GAT(compress_data['feat_dim'], num_hidden, data['num_classes'],
                    len(args.fan_out), heads, F.elu, args.dropout, cm)

    model = model.cuda()

    if dist.get_world_size() > 1:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_rank() % args.num_gpus],
        )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.978)

    sampler = dgl.dataloading.NeighborSampler(args.fan_out)
    train_dataloader = dgl.dataloading.DistNodeDataLoader(
        g,
        train_nids,
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    # val_dataloader = dgl.dataloading.DistNodeDataLoader(
    #     g,
    #     val_nids,
    #     sampler,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     drop_last=False,
    # )
    # test_dataloader = dgl.dataloading.DistNodeDataLoader(
    #     g,
    #     test_nids,
    #     sampler,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     drop_last=False,
    # )

    # sampling_hotness, feature_hotness = dataloader.presampling()
    test_accs = []
    valid_accs = []
    epoch_time_log = []
    samp_time_log = []
    load_time_log = []
    train_time_log = []
    remote_seeds_log = []
    remote_neighbors_log = []

    create_cache = args.create_cache
    if args.create_cache:
        sampling_hotness, feature_hotness, seeds_hotness = presampling(
            train_dataloader, g.num_nodes())

    pb = g.get_partition_book()
    partition_range = [0]
    for i in range(pb.num_partitions()):
        partition_range.append(pb._max_node_ids[i])
    part_id = pb.partid

    for i in range(args.num_epochs):
        samp_time = 0
        load_time = 0
        train_time = 0
        remote_seeds = 0
        remote_neighbors = 0
        begin = time.time()
        model.train()
        epoch_begin = time.time()

        samp_begin = time.time()
        for batch_idx, (input_nodes, output_nodes,
                        blocks) in enumerate(train_dataloader):
            for l, block in enumerate(blocks):
                layer_seeds = block.dstdata[dgl.NID]
                remote_layer_seeds_num = torch.sum(
                    layer_seeds >= partition_range[
                        part_id + 1]).item() + torch.sum(
                            layer_seeds < partition_range[part_id]).item()
                remote_seeds += remote_layer_seeds_num
                remote_neighbors += remote_layer_seeds_num * args.fan_out[l]

            input_nodes = input_nodes.cuda()
            output_nodes = output_nodes.cuda()
            blocks = [block.to("cuda") for block in blocks]
            samp_time += time.time() - samp_begin

            load_begin = time.time()
            output_labels = shmtensor.capi.uva_fetch(labels,
                                                     output_nodes).long()
            seeds_feature = cm.get_seeds_data(output_nodes)
            all_features = cm.get_all_data(input_nodes)
            seeds_feature = cm.reorganize_seeds_data(
                seeds_feature, all_features, blocks[0].number_of_dst_nodes())
            load_time += time.time() - load_begin

            train_begin = time.time()
            output_pred = model(blocks, (seeds_feature, all_features))
            loss = F.cross_entropy(output_pred, output_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_time += time.time() - train_begin

            if batch_idx % args.log_every == 0 and dist.get_rank() == 0:
                acc = compute_acc(output_pred, output_labels)
                print(
                    "Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Time(s) {:.4f}"
                    .format(i, batch_idx, loss.item(), acc,
                            time.time() - begin))

            if create_cache and batch_idx >= 10:
                # get max GPU memory
                max_device_mem = torch.cuda.mem_get_info()[1]
                max_allocated_mem = torch.cuda.max_memory_allocated()
                cache_capacity = max_device_mem - max_allocated_mem - args.reserved_mem * 1024 * 1024 * 1024
                print("Cache capacity: {:.3f} GB".format(cache_capacity /
                                                         1024 / 1024 / 1024))
                feat_full_size, feat_item_size = cm.data.compute_weight_size(
                    feature_hotness)
                seeds_full_size, seeds_item_size = cm.seeds_data.compute_weight_size(
                    seeds_hotness) if cm.is_two_level else (0, 0)
                feat_capacity, seeds_capacity = allocate(
                    cache_capacity, [feat_item_size, seeds_item_size],
                    [feat_full_size, seeds_full_size])
                cm.data.create_cache(feat_capacity, feature_hotness)
                if cm.is_two_level:
                    cm.seeds_data.create_cache(seeds_capacity, seeds_hotness)
                create_cache = False

            samp_begin = time.time()

        scheduler.step()
        epoch_end = time.time()

        epoch_time_log.append(epoch_end - epoch_begin)
        samp_time_log.append(samp_time)
        load_time_log.append(load_time)
        train_time_log.append(train_time)
        remote_seeds_log.append(remote_seeds)
        remote_neighbors_log.append(remote_neighbors)

    avg_remote_seeds = np.mean(remote_seeds_log[1:])
    avg_remote_neighbors = np.mean(remote_neighbors_log[1:])
    all_reduce_tensor = torch.tensor([0], device="cuda", dtype=torch.float32)
    all_reduce_tensor[0] = avg_remote_seeds
    dist.all_reduce(all_reduce_tensor, dist.ReduceOp.SUM)
    avg_remote_seeds = all_reduce_tensor[0].item()
    all_reduce_tensor[0] = avg_remote_neighbors
    dist.all_reduce(all_reduce_tensor, dist.ReduceOp.SUM)
    avg_remote_neighbors = all_reduce_tensor[0].item()

    if dist.get_rank() == 0:
        print(
            "avg epoch time: {:.3f} sec\navg samp time: {:.3f} sec\navg load time: {:.3f} sec\navg train time: {:.3f} sec\navg remote seeds: {}\navg remote neighbors: {}"
            .format(np.mean(epoch_time_log[1:]), np.mean(samp_time_log[1:]),
                    np.mean(load_time_log[1:]), np.mean(train_time_log[1:]),
                    avg_remote_seeds, avg_remote_neighbors))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        "Train nodeclassification GraphSAGE model")
    argparser.add_argument("--graph-name", type=str, help="graph name")
    argparser.add_argument("--id", type=int, help="the partition id")
    argparser.add_argument("--ip-config",
                           type=str,
                           help="The file for IP configuration")
    argparser.add_argument("--part_config",
                           type=str,
                           help="The path to the partition config file")
    argparser.add_argument(
        "--dataset",
        type=str,
        default="ogbn-products",
        help="datasets: ogbn-products, ogbn-papers100M",
    )
    argparser.add_argument("--root", type=str, default="./datasets/products")
    argparser.add_argument("--compress-root",
                           type=str,
                           default="./datasets/products")
    argparser.add_argument("--num-gpus",
                           type=int,
                           default=2,
                           help="number of gpus per machine")
    argparser.add_argument("--lr", type=float, default=0.003)
    argparser.add_argument("--dropout", type=float, default=0.5)
    argparser.add_argument("--batch-size", type=int, default=1000)
    argparser.add_argument("--batch-size-eval", type=int, default=100000)
    argparser.add_argument("--log-every", type=int, default=20)
    argparser.add_argument("--eval-every", type=int, default=5)
    argparser.add_argument("--fan-out", type=str, default="12,12,12")
    argparser.add_argument("--num-hidden", type=int, default=256)
    argparser.add_argument("--num-epochs", type=int, default=11)
    argparser.add_argument("--breakdown", action="store_true")
    argparser.add_argument("--fusion", action="store_true", default=False)
    argparser.add_argument("--create-cache",
                           action="store_true",
                           default=False)
    argparser.add_argument("--reserved-mem",
                           type=float,
                           default=1.0,
                           help="reserverd GPU memory size, unit: GB")
    argparser.add_argument("--num-heads", default=8, type=int)
    argparser.add_argument("--model",
                           default="sage",
                           type=str,
                           choices=["sage", "gat"])
    args = argparser.parse_args()
    print(args)

    args.fan_out = [int(x) for x in args.fan_out.split(",")]

    dgl.distributed.initialize(args.ip_config)
    dist.init_process_group(backend="nccl")
    g = dgl.distributed.DistGraph(args.graph_name,
                                  part_config=args.part_config)

    # if dist.get_rank() == 0:
    #     train_mask = g.ndata["train_mask"][np.arange(0, g.num_nodes())]
    #     val_mask = g.ndata["val_mask"][np.arange(0, g.num_nodes())]
    #     test_mask = g.ndata["test_mask"][np.arange(0, g.num_nodes())]
    #     train_nids = torch.nonzero(train_mask).flatten()
    #     print(train_nids)
    #     print(train_nids.shape[0])
    #     val_nids = torch.nonzero(val_mask).flatten()
    #     print(val_nids)
    #     print(val_nids.shape[0])
    #     test_nids = torch.nonzero(test_mask).flatten()
    #     print(test_nids)
    #     print(test_nids.shape[0])
    #     seeds = torch.cat([train_nids, val_nids, test_nids])
    #     torch.save(seeds, 'seeds.pt')

    pb = g.get_partition_book()
    train_nid = dgl.distributed.node_split(g.ndata["train_mask"],
                                           pb,
                                           force_even=True)
    val_nid = dgl.distributed.node_split(g.ndata["val_mask"],
                                         pb,
                                         force_even=True)
    test_nid = dgl.distributed.node_split(g.ndata["test_mask"],
                                          pb,
                                          force_even=True)
    local_nid = pb.partid2nids(pb.partid).detach().numpy()
    print("part {}, train: {} (local: {}), val: {} (local: {}), test: {} "
          "(local: {})".format(
              dist.get_rank(),
              len(train_nid),
              len(np.intersect1d(train_nid.numpy(), local_nid)),
              len(val_nid),
              len(np.intersect1d(val_nid.numpy(), local_nid)),
              len(test_nid),
              len(np.intersect1d(test_nid.numpy(), local_nid)),
          ))

    rank = dist.get_rank()
    torch.cuda.set_device(rank % args.num_gpus)

    meta = load_meta(args)
    local_rank, local_world_size, local_group = create_local_group(
        args.num_gpus)
    # labels = load_shm_tensor("labels", args, local_rank, local_world_size,
    #                          meta)
    # labels.tensor_[torch.isnan(labels.tensor_)] = 0
    # num_classes = int(labels.tensor_.max() + 1)
    # print("Num classes:", num_classes)
    # data = {"labels": labels, "num_classes": num_classes}
    labels = g.ndata["labels"][np.arange(0, g.num_nodes())]
    num_classes = int(labels[~torch.isnan(labels)].max().item() + 1)
    print("Num classes:", num_classes)
    data = {"labels": labels, "num_classes": num_classes}
    compress_data = dist_load_compress_data(args.compress_root, args.num_gpus)
    seeds = train_nid, val_nid, test_nid

    if dist.get_rank() == 0:
        # print(data.keys())
        print(compress_data.keys())

    run(args, data, compress_data, g, seeds)
