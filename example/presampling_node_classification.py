import argparse
import time
import os
import dgl
import torch.distributed as dist
import torch
from load_dataset import load_dataset

torch.manual_seed(25)


def run(rank, world_size, data, args):
    torch.cuda.set_device(rank)
    device = torch.device("cuda")
    dist.init_process_group('nccl',
                            'tcp://127.0.0.1:12347',
                            world_size=world_size,
                            rank=rank)
    # Unpack data
    train_nid, g = data
    shuffle = True
    sampler = dgl.dataloading.NeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(",")])
    dataloader = dgl.dataloading.DataLoader(g,
                                            train_nid,
                                            sampler,
                                            device=device,
                                            batch_size=args.batch_size,
                                            shuffle=shuffle,
                                            drop_last=False,
                                            num_workers=0,
                                            use_ddp=True,
                                            use_uva=True)

    presampling_heat = torch.zeros((g.num_nodes(), ), dtype=torch.float32)

    tic = time.time()
    for epoch in range(args.num_epochs):
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            presampling_heat[input_nodes.cpu()] += 1

    if torch.distributed.get_backend() == "nccl":
        presampling_heat = presampling_heat.cuda()
        torch.distributed.all_reduce(presampling_heat,
                                     torch.distributed.ReduceOp.SUM)
        presampling_heat = presampling_heat.cpu()
    else:
        torch.distributed.all_gather(presampling_heat,
                                     torch.distributed.ReduceOp.SUM)
    toc = time.time()

    presampling_heat_accessed = presampling_heat[presampling_heat > 0]
    info = "========================================\n"
    info += "Rank {} presampling info:\n".format(torch.cuda.current_device())
    info += "Presampling done, max: {:.3f} min: {:.3f} avg: {:.3f}\n".format(
        torch.max(presampling_heat_accessed).item(),
        torch.min(presampling_heat_accessed).item(),
        torch.mean(presampling_heat_accessed).item())
    info += "Presampling time: {:.3f} s\n".format(toc - tic)
    info += "========================================"
    print(info)

    if rank == 0:
        save_path = os.path.join(
            args.save_path, args.dataset + "_" + str(args.fan_out) +
            "_presampling_hotness.pkl")
        torch.save(presampling_heat, save_path)
        print("Hotness saved to {}\n".format(save_path))


def main(args):
    g, _ = load_dataset(args.root, args.dataset, with_feature=False)
    dgl_g = dgl.graph(("csc", (g["indptr"], g["indices"], torch.tensor([]))))

    if args.seeds_rate > 0:
        num_nodes = dgl_g.num_nodes()
        num_train = int(num_nodes * args.seeds_rate)
        train_nid = torch.randperm(num_nodes)[:num_train]
    else:
        train_nid = g["train_idx"]
    print("Train: {}".format(train_nid.numel()))

    data = train_nid, dgl_g
    import torch.multiprocessing as mp
    mp.spawn(run, args=(args.num_gpus, data, args), nprocs=args.num_gpus)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCN")
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=8,
        help="the number of GPU device. Use -1 for CPU training",
    )
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--fan_out", type=str, default="5,10,15")
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument(
        "--dataset",
        type=str,
        default="ogbn-products",
        help="datasets: ogbn-products, ogbn-papers100M",
    )
    parser.add_argument("--root", type=str, default="/data")
    parser.add_argument("--seeds_rate", default=0.0, type=float)
    parser.add_argument("--save_path", type=str, default=".")
    args = parser.parse_args()

    print(args)
    main(args)
