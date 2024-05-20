import dgl
import torch
import os
import pandas as pd
import argparse
import numpy as np


def process_reddit(save_path, generate_seeds=False, generate_hotness=False):
    data = dgl.data.RedditDataset(self_loop=True)
    dgl_graph = data[0]

    csc = dgl_graph.adj_tensors('csc')
    indptr, indices, _ = csc
    train_nids = torch.nonzero(dgl_graph.ndata['train_mask']).squeeze(1)
    valid_nids = torch.nonzero(dgl_graph.ndata['val_mask']).squeeze(1)
    test_nids = torch.nonzero(dgl_graph.ndata['test_mask']).squeeze(1)
    labels = dgl_graph.ndata['label']

    features = dgl_graph.ndata['feat']

    torch.save(indptr, os.path.join(save_path, 'csc_indptr.pt'))
    torch.save(indices, os.path.join(save_path, 'csc_indices.pt'))
    torch.save(train_nids, os.path.join(save_path, 'train_nids.pt'))
    torch.save(valid_nids, os.path.join(save_path, 'valid_nids.pt'))
    torch.save(test_nids, os.path.join(save_path, 'test_nids.pt'))
    torch.save(labels, os.path.join(save_path, 'labels.pt'))

    torch.save(features, os.path.join(save_path, 'features.pt'))

    meta = {
        "csc_indptr": (indptr.dtype, indptr.shape),
        "csc_indices": (indices.dtype, indices.shape),
        "train_nids": (train_nids.dtype, train_nids.shape),
        "valid_nids": (valid_nids.dtype, valid_nids.shape),
        "test_nids": (test_nids.dtype, test_nids.shape),
        "labels": (labels.dtype, labels.shape),
        "features": (features.dtype, features.shape),
    }

    if generate_hotness:
        hotness = indptr[1:] - indptr[:-1]
        torch.save(hotness, os.path.join(save_path, 'hotness.pt'))
        meta["hotness"] = (hotness.dtype, hotness.shape)

    if generate_seeds:
        seeds = torch.cat([train_nids, valid_nids, test_nids])
        torch.save(seeds, os.path.join(save_path, 'seeds.pt'))
        meta["seeds"] = (seeds.dtype, seeds.shape)

    torch.save(meta, os.path.join(save_path, 'meta.pt'))


def process_ogbn(name,
                 root,
                 save_path,
                 generate_seeds=False,
                 generate_hotness=False,
                 bidirected=True):
    from ogb.nodeproppred import DglNodePropPredDataset
    data = DglNodePropPredDataset(name=name, root=root)
    dgl_graph, labels = data[0]

    splitted_idx = data.get_idx_split()

    train_nids = splitted_idx["train"]
    valid_nids = splitted_idx["valid"]
    test_nids = splitted_idx["test"]
    labels = labels[:, 0]
    features = dgl_graph.ndata.pop('feat')

    if bidirected:
        dgl_graph = dgl.to_bidirected(dgl_graph)
    csc = dgl_graph.adj_tensors('csc')
    indptr, indices, _ = csc

    torch.save(indptr, os.path.join(save_path, 'csc_indptr.pt'))
    torch.save(indices, os.path.join(save_path, 'csc_indices.pt'))
    torch.save(train_nids, os.path.join(save_path, 'train_nids.pt'))
    torch.save(valid_nids, os.path.join(save_path, 'valid_nids.pt'))
    torch.save(test_nids, os.path.join(save_path, 'test_nids.pt'))
    torch.save(labels, os.path.join(save_path, 'labels.pt'))
    torch.save(features, os.path.join(save_path, 'features.pt'))

    meta = {
        "csc_indptr": (indptr.dtype, indptr.shape),
        "csc_indices": (indices.dtype, indices.shape),
        "train_nids": (train_nids.dtype, train_nids.shape),
        "valid_nids": (valid_nids.dtype, valid_nids.shape),
        "test_nids": (test_nids.dtype, test_nids.shape),
        "labels": (labels.dtype, labels.shape),
        "features": (features.dtype, features.shape)
    }

    if generate_hotness:
        hotness = indptr[1:] - indptr[:-1]
        torch.save(hotness, os.path.join(save_path, 'hotness.pt'))
        meta["hotness"] = (hotness.dtype, hotness.shape)

    if generate_seeds:
        seeds = torch.cat([train_nids, valid_nids, test_nids])
        torch.save(seeds, os.path.join(save_path, 'seeds.pt'))
        meta["seeds"] = (seeds.dtype, seeds.shape)

    torch.save(meta, os.path.join(save_path, 'meta.pt'))


def process_friendster(dataset_path,
                       save_path,
                       generate_seeds=False,
                       generate_hotness=False,
                       bidirected=True):

    def _download(url, path, filename):
        import requests

        fn = os.path.join(path, filename)
        if os.path.exists(fn):
            return
        print("Download friendster.")
        os.makedirs(path, exist_ok=True)
        f_remote = requests.get(url, stream=True)
        sz = f_remote.headers.get('content-length')
        assert f_remote.status_code == 200, 'fail to open {}'.format(url)
        with open(fn, 'wb') as writer:
            for chunk in f_remote.iter_content(chunk_size=1024 * 1024):
                writer.write(chunk)
        print('Download finished.')

    _download(
        'https://dgl-asv-data.s3-us-west-2.amazonaws.com/dataset/friendster/com-friendster.ungraph.txt.gz',
        dataset_path, 'com-friendster.ungraph.txt.gz')

    df = pd.read_csv(os.path.join(dataset_path,
                                  'com-friendster.ungraph.txt.gz'),
                     sep='\t',
                     skiprows=4,
                     header=None,
                     names=['src', 'dst'],
                     compression='gzip')
    src = df['src'].values
    dst = df['dst'].values

    print('construct the graph...')
    g = dgl.graph((src, dst))
    if bidirected:
        g = dgl.to_bidirected(g)
    g = g.formats("csc")
    g.create_formats_()
    indptr, indices, _ = g.adj_tensors("csc")
    num_nodes = g.num_nodes()
    num_edges = g.num_edges()
    del g

    train_nids = torch.randperm(num_nodes)[:int(num_nodes * 0.01)]
    valid_nids = torch.randperm(num_nodes)[int(num_nodes *
                                               0.01):int(num_nodes * 0.02)]
    test_nids = torch.randperm(num_nodes)[int(num_nodes * 0.02):int(num_nodes *
                                                                    0.03)]

    num_classes = 42
    feature_dim = 256
    labels = torch.randint(0, num_classes, (num_nodes, )).long()
    features = torch.randn((num_nodes, feature_dim)).float()

    print("Save data...")
    torch.save(indptr, os.path.join(save_path, 'csc_indptr.pt'))
    torch.save(indices, os.path.join(save_path, 'csc_indices.pt'))
    torch.save(train_nids, os.path.join(save_path, 'train_nids.pt'))
    torch.save(valid_nids, os.path.join(save_path, 'valid_nids.pt'))
    torch.save(test_nids, os.path.join(save_path, 'test_nids.pt'))
    torch.save(labels, os.path.join(save_path, 'labels.pt'))
    torch.save(features, os.path.join(save_path, 'features.pt'))

    meta = {
        "csc_indptr": (indptr.dtype, indptr.shape),
        "csc_indices": (indices.dtype, indices.shape),
        "train_nids": (train_nids.dtype, train_nids.shape),
        "valid_nids": (valid_nids.dtype, valid_nids.shape),
        "test_nids": (test_nids.dtype, test_nids.shape),
        "labels": (labels.dtype, labels.shape),
        "features": (features.dtype, features.shape)
    }

    if generate_hotness:
        hotness = indptr[1:] - indptr[:-1]
        torch.save(hotness, os.path.join(save_path, 'hotness.pt'))
        meta["hotness"] = (hotness.dtype, hotness.shape)

    if generate_seeds:
        seeds = torch.cat([train_nids, valid_nids, test_nids])
        torch.save(seeds, os.path.join(save_path, 'seeds.pt'))
        meta["seeds"] = (seeds.dtype, seeds.shape)

    torch.save(meta, os.path.join(save_path, 'meta.pt'))


def process_mag240m(dataset_path,
                    save_path,
                    generate_seeds=False,
                    generate_hotness=False,
                    bidirected=True):
    from ogb.lsc import MAG240MDataset
    dataset = MAG240MDataset(dataset_path)

    ei_writes = dataset.edge_index("author", "writes", "paper")
    ei_cites = dataset.edge_index("paper", "paper")
    ei_affiliated = dataset.edge_index("author", "institution")

    # We sort the nodes starting with the authors, insts, papers
    author_offset = 0
    inst_offset = author_offset + dataset.num_authors
    paper_offset = inst_offset + dataset.num_institutions
    print("Author offset", author_offset)
    print("Inst offset", inst_offset)
    print("Paper offset", paper_offset)

    print("Construct graph...")
    g = dgl.heterograph({
        ("author", "write", "paper"): (ei_writes[0], ei_writes[1]),
        ("paper", "write-by", "author"): (ei_writes[1], ei_writes[0]),
        ("author", "affiliate-with", "institution"): (
            ei_affiliated[0],
            ei_affiliated[1],
        ),
        ("institution", "affiliate", "author"): (
            ei_affiliated[1],
            ei_affiliated[0],
        ),
        ("paper", "cite", "paper"): (
            np.concatenate([ei_cites[0], ei_cites[1]]),
            np.concatenate([ei_cites[1], ei_cites[0]]),
        ),
    })
    print("#Nodes", g.num_nodes())
    print("#Edges", g.num_edges())

    print("Convert to homogeneous...")
    g = dgl.to_homogeneous(g)
    assert torch.equal(
        g.ndata[dgl.NTYPE],
        torch.cat([
            torch.full((dataset.num_authors, ), 0),
            torch.full((dataset.num_institutions, ), 1),
            torch.full((dataset.num_papers, ), 2),
        ]),
    )
    assert torch.equal(
        g.ndata[dgl.NID],
        torch.cat([
            torch.arange(dataset.num_authors),
            torch.arange(dataset.num_institutions),
            torch.arange(dataset.num_papers),
        ]),
    )
    del g.edata[dgl.ETYPE]
    del g.ndata[dgl.NTYPE]
    del g.ndata[dgl.NID]
    if bidirected:
        g = dgl.to_bidirected(g)
    indptr, indices, _ = g.adj_tensors("csc")
    num_nodes = g.num_nodes()
    num_edges = g.num_edges()
    del g

    train_nids = torch.tensor(dataset.get_idx_split("train")).long()
    train_nids += paper_offset
    print("#Train", train_nids.shape[0])
    valid_nids = torch.tensor(dataset.get_idx_split("valid")).long()
    valid_nids += paper_offset
    print("#Valid", valid_nids.shape[0])
    test_nids = torch.tensor(dataset.get_idx_split("test-whole")).long()
    test_nids += paper_offset
    print("#Test", test_nids.shape[0])

    paper_labels = torch.tensor(dataset.paper_label)
    other_labels = torch.full((num_nodes - paper_labels.shape[0], ),
                              float('nan'),
                              dtype=torch.float)
    labels = torch.cat([other_labels, paper_labels])

    num_classes = torch.max(paper_labels).item() + 1

    print("Save data...")
    torch.save(indptr, os.path.join(save_path, 'csc_indptr.pt'))
    torch.save(indices, os.path.join(save_path, 'csc_indices.pt'))
    torch.save(train_nids, os.path.join(save_path, 'train_nids.pt'))
    torch.save(valid_nids, os.path.join(save_path, 'valid_nids.pt'))
    torch.save(test_nids, os.path.join(save_path, 'test_nids.pt'))
    torch.save(labels, os.path.join(save_path, 'labels.pt'))

    meta = {
        "csc_indptr": (indptr.dtype, indptr.shape),
        "csc_indices": (indices.dtype, indices.shape),
        "train_nids": (train_nids.dtype, train_nids.shape),
        "valid_nids": (valid_nids.dtype, valid_nids.shape),
        "test_nids": (test_nids.dtype, test_nids.shape),
        "labels": (labels.dtype, labels.shape),
        "features": (torch.float16, (num_nodes, 768))
    }

    if generate_hotness:
        hotness = indptr[1:] - indptr[:-1]
        torch.save(hotness, os.path.join(save_path, 'hotness.pt'))
        meta["hotness"] = (hotness.dtype, hotness.shape)

    if generate_seeds:
        seeds = torch.cat([train_nids, valid_nids, test_nids])
        torch.save(seeds, os.path.join(save_path, 'seeds.pt'))
        meta["seeds"] = (seeds.dtype, seeds.shape)

    torch.save(meta, os.path.join(save_path, 'meta.pt'))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset", type=str, default="ogbn-products")
    argparser.add_argument("--root",
                           type=str,
                           default="/home/ubuntu/workspace/datasets")
    argparser.add_argument(
        "--out-dir",
        type=str,
        default="/home/ubuntu/f2cgt_workspace/v2_dataset/products")
    args = argparser.parse_args()
    print(args)

    if args.dataset == "ogbn-products" or args.dataset == "ogbn-papers100M":
        process_ogbn(args.dataset,
                     args.root,
                     args.out_dir,
                     generate_seeds=True,
                     generate_hotness=True,
                     bidirected=True)
    elif args.dataset == "friendster":
        process_friendster(args.root,
                           args.out_dir,
                           generate_seeds=True,
                           generate_hotness=True,
                           bidirected=True)
    elif args.dataset == "mag240m":
        process_mag240m(args.root,
                        args.out_dir,
                        generate_seeds=True,
                        generate_hotness=True,
                        bidirected=True)
