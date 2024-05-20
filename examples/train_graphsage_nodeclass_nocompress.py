import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import shmtensor
from shmtensor import GPUSamplingDataloader
from load_dataset import dist_load_tensor
from models import SAGE, compute_acc, nodewise_inference, GAT
from tensor_cache import total_create_cache, TensorCache
import argparse

torch.manual_seed(25)


def run(args, data):

    # unpack data
    labels = data['labels'].tensor_
    train_nids = data['train_nids'].tensor_
    test_nids = data['test_nids'].tensor_
    valid_nids = data['valid_nids'].tensor_

    shmtensor.capi.pin_memory(labels)

    # create model
    if args.model == "sage":
        model = SAGE(data['features'].tensor_.shape[1], args.num_hidden,
                     data['num_classes'], len(args.fan_out), F.relu,
                     args.dropout)
    elif args.model == "gat":
        heads = [args.num_heads for _ in range(len(args.fan_out) - 1)]
        heads.append(1)
        num_hidden = args.num_hidden // args.num_heads
        model = GAT(data['features'].tensor_.shape[1], num_hidden,
                    data['num_classes'], len(args.fan_out), heads, F.elu,
                    args.dropout)

    model = model.cuda()

    if dist.get_world_size() > 1:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_rank() % args.num_gpus],
        )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.978)

    dataloader = GPUSamplingDataloader(data['csc_indptr'].tensor_,
                                       data['csc_indices'].tensor_,
                                       train_nids,
                                       args.batch_size,
                                       args.fan_out,
                                       shuffle=True,
                                       drop_last=False,
                                       use_ddp=True)
    feature_cache = TensorCache(data['features'].tensor_)

    test_accs = []
    valid_accs = []
    epoch_time_log = []
    samp_time_log = []
    load_time_log = []
    train_time_log = []

    create_cache = args.create_cache

    for i in range(args.num_epochs):
        samp_time = 0
        load_time = 0
        train_time = 0

        begin = time.time()
        model.train()
        dataloader.update_params(seeds=train_nids)
        epoch_begin = time.time()

        samp_begin = time.time()
        for batch_idx, (
                input_nodes,
                output_nodes,
                blocks,
        ) in enumerate(dataloader):
            samp_time += time.time() - samp_begin

            load_begin = time.time()
            feature = feature_cache[input_nodes]
            output_labels = shmtensor.capi.uva_fetch(labels,
                                                     output_nodes).long()
            load_time += time.time() - load_begin

            train_begin = time.time()
            output_pred = model(blocks, feature)
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
                total_create_cache(cache_capacity, dataloader, feature_cache)
                create_cache = False

            samp_begin = time.time()

        scheduler.step()
        epoch_end = time.time()

        if dist.get_rank() == 0:
            print("Memory allocated: {}, Memory reserverd: {}".format(
                torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024,
                torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024))

        if (i + 1) % args.eval_every == 0:
            dataloader.update_params(seeds=valid_nids)
            valid_acc = nodewise_inference(model, dataloader, labels).cuda()
            dist.all_reduce(valid_acc)
            valid_acc = valid_acc / dist.get_world_size()

            dataloader.update_params(seeds=test_nids)
            test_acc = nodewise_inference(model, dataloader, labels).cuda()
            dist.all_reduce(test_acc)
            test_acc = test_acc / dist.get_world_size()

            if dist.get_rank() == 0:
                print("Valid Acc {:.4f}, Test Acc {:.4f}, epoch {:.4f}".format(
                    valid_acc, test_acc, epoch_end - epoch_begin))

            test_accs.append(test_acc.item())
            valid_accs.append(valid_acc.item())
        epoch_time_log.append(epoch_end - epoch_begin)
        samp_time_log.append(samp_time)
        load_time_log.append(load_time)
        train_time_log.append(train_time)

    if dist.get_rank() == 0:
        print(
            "final test acc: {}, valid acc: {}\navg epoch time: {:.3f} sec\navg samp time: {:.3f} sec\navg load time: {:.3f} sec\navg train time: {:.3f} sec"
            .format(np.mean(test_accs[-5:]), np.mean(valid_accs[-5:]),
                    np.mean(epoch_time_log[1:]), np.mean(samp_time_log[1:]),
                    np.mean(load_time_log[1:]), np.mean(train_time_log[1:])))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        "Train nodeclassification GraphSAGE model")
    argparser.add_argument(
        "--dataset",
        type=str,
        default="ogbn-products",
        help="datasets: ogbn-products, ogbn-papers100M",
    )
    argparser.add_argument("--root", type=str, default="./datasets/products")
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

    dist.init_process_group('nccl', init_method='env://')
    rank = dist.get_rank()
    torch.cuda.set_device(rank % args.num_gpus)

    data = dist_load_tensor(args.root, args.num_gpus, with_feature=True)

    if dist.get_rank() == 0:
        print(data.keys())

    run(args, data)
