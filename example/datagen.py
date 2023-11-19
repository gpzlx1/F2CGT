import argparse
import numpy as np
import pandas as pd
import torch
import os
from scipy.sparse import coo_matrix


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        default="ogbn-papers100M",
                        choices=["ogbn-products", "ogbn-papers100M"])
    parser.add_argument("--root", help="Path of the dataset.")
    parser.add_argument("--save-path", help="Path to save the processed data.")
    args = parser.parse_args()
    print(args)

    if args.dataset == "ogbn-papers100M":
        process_papers100M(args.root, args.save_path)
    elif args.dataset == "ogbn-products":
        process_products(args.root, args.save_path)