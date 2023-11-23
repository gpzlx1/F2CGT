import argparse
import numpy as np
import pandas as pd
import torch
import os
from scipy.sparse import coo_matrix
from ogb.lsc import MAG240MDataset
import dgl
import dgl.function as fn
import tqdm


def process_products(dataset_path, save_path):
    print("Process ogbn-products...")

    print("Read raw data...")
    edges = pd.read_csv(os.path.join(dataset_path, "raw/edge.csv.gz"),
                        compression="gzip",
                        header=None).values.T
    features = pd.read_csv(os.path.join(dataset_path, "raw/node-feat.csv.gz"),
                           compression="gzip",
                           header=None).values
    labels = pd.read_csv(os.path.join(dataset_path, "raw/node-label.csv.gz"),
                         compression="gzip",
                         header=None).values.T[0]
    train_idx = pd.read_csv(os.path.join(dataset_path,
                                         "split/sales_ranking/train.csv.gz"),
                            compression="gzip",
                            header=None).values.T[0]
    valid_idx = pd.read_csv(os.path.join(dataset_path,
                                         "split/sales_ranking/valid.csv.gz"),
                            compression="gzip",
                            header=None).values.T[0]
    test_idx = pd.read_csv(os.path.join(dataset_path,
                                        "split/sales_ranking/test.csv.gz"),
                           compression="gzip",
                           header=None).values.T[0]

    print("Process data...")
    num_nodes = features.shape[0]
    src = np.concatenate((edges[0], edges[1]))
    dst = np.concatenate((edges[1], edges[0]))
    data = np.zeros(src.shape)
    coo = coo_matrix((data, (dst, src)),
                     shape=(num_nodes, num_nodes),
                     dtype=np.int64)
    csc = coo.tocsr()
    indptr = csc.indptr
    indices = csc.indices
    num_edges = indices.shape[0]

    print("Save data...")
    torch.save(
        torch.from_numpy(features).float(),
        os.path.join(save_path, "features.pt"))
    torch.save(
        torch.from_numpy(labels).long(), os.path.join(save_path, "labels.pt"))
    torch.save(
        torch.from_numpy(indptr).long(), os.path.join(save_path, "indptr.pt"))
    torch.save(
        torch.from_numpy(indices).long(), os.path.join(save_path,
                                                       "indices.pt"))
    torch.save(
        torch.from_numpy(train_idx).long(),
        os.path.join(save_path, "train_idx.pt"))
    torch.save(
        torch.from_numpy(valid_idx).long(),
        os.path.join(save_path, "valid_idx.pt"))
    torch.save(
        torch.from_numpy(test_idx).long(),
        os.path.join(save_path, "test_idx.pt"))

    print("Generate meta data...")
    num_classes = np.unique(labels[~np.isnan(labels)]).shape[0]
    feature_dim = features.shape[1]
    num_train_nodes = train_idx.shape[0]
    num_valid_nodes = valid_idx.shape[0]
    num_test_nodes = test_idx.shape[0]

    print("Save meta data...")
    meta_data = {
        "dataset": "ogbn-products",
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "num_classes": num_classes,
        "feature_dim": feature_dim,
        "num_train_nodes": num_train_nodes,
        "num_valid_nodes": num_valid_nodes,
        "num_test_nodes": num_test_nodes
    }
    torch.save(meta_data, os.path.join(save_path, "metadata.pt"))


def process_papers100M(dataset_path, save_path):
    print("Process ogbn-papers100M...")

    print("Read raw data...")
    data_file = np.load(os.path.join(dataset_path, "raw/data.npz"))
    label_file = np.load(os.path.join(dataset_path, "raw/node-label.npz"))
    features = data_file["node_feat"]
    labels = label_file["node_label"]
    edge_index = data_file["edge_index"]
    train_idx = pd.read_csv(os.path.join(dataset_path,
                                         'split/time/train.csv.gz'),
                            compression='gzip',
                            header=None).values.T[0]
    valid_idx = pd.read_csv(os.path.join(dataset_path,
                                         'split/time/valid.csv.gz'),
                            compression='gzip',
                            header=None).values.T[0]
    test_idx = pd.read_csv(os.path.join(dataset_path,
                                        'split/time/test.csv.gz'),
                           compression='gzip',
                           header=None).values.T[0]

    print("Process data...")
    num_nodes = features.shape[0]
    src = edge_index[0]
    dst = edge_index[1]
    data = np.zeros(src.shape)
    coo = coo_matrix((data, (dst, src)),
                     shape=(num_nodes, num_nodes),
                     dtype=np.int64)
    csc = coo.tocsr()
    indptr = csc.indptr
    indices = csc.indices
    num_edges = indices.shape[0]

    print("Save data...")
    torch.save(
        torch.from_numpy(features).float(),
        os.path.join(save_path, "features.pt"))
    torch.save(
        torch.from_numpy(labels).float().squeeze(1),
        os.path.join(save_path, "labels.pt"))
    torch.save(
        torch.from_numpy(indptr).long(), os.path.join(save_path, "indptr.pt"))
    torch.save(
        torch.from_numpy(indices).long(), os.path.join(save_path,
                                                       "indices.pt"))
    torch.save(
        torch.from_numpy(train_idx).long(),
        os.path.join(save_path, "train_idx.pt"))
    torch.save(
        torch.from_numpy(valid_idx).long(),
        os.path.join(save_path, "valid_idx.pt"))
    torch.save(
        torch.from_numpy(test_idx).long(),
        os.path.join(save_path, "test_idx.pt"))

    print("Generate meta data...")
    num_classes = np.unique(labels[~np.isnan(labels)]).shape[0]
    feature_dim = features.shape[1]
    num_train_nodes = train_idx.shape[0]
    num_valid_nodes = valid_idx.shape[0]
    num_test_nodes = test_idx.shape[0]

    print("Save meta data...")
    meta_data = {
        "dataset": "ogbn-papers100M",
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "num_classes": num_classes,
        "feature_dim": feature_dim,
        "num_train_nodes": num_train_nodes,
        "num_valid_nodes": num_valid_nodes,
        "num_test_nodes": num_test_nodes
    }
    torch.save(meta_data, os.path.join(save_path, "metadata.pt"))


def process_mag240m(dataset_path, save_path, gen_feat=False):
    dataset = MAG240MDataset(dataset_path)

    ei_writes = dataset.edge_index("author", "writes", "paper")
    ei_cites = dataset.edge_index("paper", "paper")
    ei_affiliated = dataset.edge_index("author", "institution")

    # We sort the nodes starting with the authors, insts, papers
    author_offset = 0
    inst_offset = author_offset + dataset.num_authors
    paper_offset = inst_offset + dataset.num_institutions

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

    if gen_feat:
        print("process author & inst features...")
        paper_feat = dataset.paper_feat
        author_feat = np.memmap(
            os.path.join(save_path, "author_features.npy"),
            mode="w+",
            dtype="float16",
            shape=(dataset.num_authors, dataset.num_paper_features),
        )
        inst_feat = np.memmap(
            os.path.join(save_path, "inst_features.npy"),
            mode="w+",
            dtype="float16",
            shape=(dataset.num_institutions, dataset.num_paper_features),
        )

        # Iteratively process author features along the feature dimension.
        BLOCK_COLS = 16
        with tqdm.trange(0, dataset.num_paper_features, BLOCK_COLS) as tq:
            for start in tq:
                tq.set_postfix_str("Reading paper features...")
                g.nodes["paper"].data["x"] = torch.FloatTensor(
                    paper_feat[:, start:start + BLOCK_COLS].astype("float32"))
                # Compute author features...
                tq.set_postfix_str("Computing author features...")
                g.update_all(fn.copy_u("x", "m"),
                             fn.mean("m", "x"),
                             etype="write-by")
                # Then institution features...
                tq.set_postfix_str("Computing institution features...")
                g.update_all(fn.copy_u("x", "m"),
                             fn.mean("m", "x"),
                             etype="affiliate-with")
                tq.set_postfix_str("Writing author features...")
                author_feat[:, start:start + BLOCK_COLS] = (
                    g.nodes["author"].data["x"].numpy().astype("float16"))
                tq.set_postfix_str("Writing institution features...")
                inst_feat[:, start:start + BLOCK_COLS] = (
                    g.nodes["institution"].data["x"].numpy().astype("float16"))
                del g.nodes["paper"].data["x"]
                del g.nodes["author"].data["x"]
                del g.nodes["institution"].data["x"]
        author_feat.flush()
        inst_feat.flush()

    print("Convert to homogeneous...")
    g = dgl.to_homogeneous(g)
    # DGL ensures that nodes with the same type are put together with the order preserved.
    # DGL also ensures that the node types are sorted in ascending order.
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
    g.edata["etype"] = g.edata[dgl.ETYPE].byte()
    del g.edata[dgl.ETYPE]
    del g.ndata[dgl.NTYPE]
    del g.ndata[dgl.NID]

    g = g.formats("csc")
    g.create_formats_()
    indptr, indices, _ = g.adj_tensors("csc")
    num_nodes = g.num_nodes()
    num_edges = g.num_edges()
    del g

    train_idx = torch.LongTensor(dataset.get_idx_split("train")) + paper_offset
    valid_idx = torch.LongTensor(dataset.get_idx_split("valid")) + paper_offset
    test_idx = torch.LongTensor(
        dataset.get_idx_split("test-whole")) + paper_offset

    meta_data = {
        "dataset": "mag240m",
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "num_classes": dataset.num_classes,
        "feature_dim": dataset.num_paper_features,
        "num_train_nodes": train_idx.shape[0],
        "num_valid_nodes": valid_idx.shape[0],
        "num_test_nodes": test_idx.shape[0],
    }
    print(meta_data)

    # random label for authors and insts
    paper_labels = torch.LongTensor(dataset.paper_label)
    other_labels = torch.randint(0, dataset.num_classes,
                                 (num_nodes - paper_labels.shape[0], )).long()
    full_labels = torch.cat([other_labels, paper_labels])

    print("Save data...")
    torch.save(full_labels, os.path.join(save_path, "labels.pt"))
    torch.save(indptr.long(), os.path.join(save_path, "indptr.pt"))
    torch.save(indices.long(), os.path.join(save_path, "indices.pt"))
    torch.save(train_idx, os.path.join(save_path, "train_idx.pt"))
    torch.save(valid_idx, os.path.join(save_path, "valid_idx.pt"))
    torch.save(test_idx, os.path.join(save_path, "test_idx.pt"))
    torch.save(meta_data, os.path.join(save_path, "metadata.pt"))

    if gen_feat:
        # Process feature
        print("Write full features...")
        full_feat = np.memmap(
            os.path.join(save_path, "features.npy"),
            mode="w+",
            dtype="float16",
            shape=(
                dataset.num_authors + dataset.num_institutions +
                dataset.num_papers,
                dataset.num_paper_features,
            ),
        )
        BLOCK_ROWS = 100000
        for start in tqdm.trange(0, dataset.num_authors, BLOCK_ROWS):
            end = min(dataset.num_authors, start + BLOCK_ROWS)
            full_feat[author_offset + start:author_offset +
                      end] = author_feat[start:end]
        for start in tqdm.trange(0, dataset.num_institutions, BLOCK_ROWS):
            end = min(dataset.num_institutions, start + BLOCK_ROWS)
            full_feat[inst_offset + start:inst_offset +
                      end] = inst_feat[start:end]
        for start in tqdm.trange(0, dataset.num_papers, BLOCK_ROWS):
            end = min(dataset.num_papers, start + BLOCK_ROWS)
            full_feat[paper_offset + start:paper_offset +
                      end] = paper_feat[start:end]
        full_feat.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="ogbn-papers100M",
        choices=["ogbn-products", "ogbn-papers100M", "mag240m"])
    parser.add_argument("--root", help="Path of the dataset.")
    parser.add_argument("--save-path", help="Path to save the processed data.")
    args = parser.parse_args()
    print(args)

    if args.dataset == "ogbn-papers100M":
        process_papers100M(args.root, args.save_path)
    elif args.dataset == "ogbn-products":
        process_products(args.root, args.save_path)
    elif args.dataset == "mag240m":
        process_mag240m(args.root, args.save_path)
