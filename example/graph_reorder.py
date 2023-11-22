import torch
import os
import dgl
import argparse


def graph_reorder(path, hotness, save_path):
    # support homogeneous graph like products & papers100M
    labels = torch.load(os.path.join(path, "labels.pt"))
    indptr = torch.load(os.path.join(path, "indptr.pt"))
    indices = torch.load(os.path.join(path, "indices.pt"))
    train_idx = torch.load(os.path.join(path, "train_idx.pt"))
    val_idx = torch.load(os.path.join(path, "valid_idx.pt"))
    test_idx = torch.load(os.path.join(path, "test_idx.pt"))
    features = torch.load(os.path.join(path, "features.pt"))

    meta_data = torch.load(os.path.join(path, "metadata.pt"))

    csc_g = dgl.graph(("csc", (indptr, indices, torch.tensor([]))))
    coo_g = csc_g.formats("coo")
    coo_g.create_formats_()
    coo_row, coo_col = coo_g.adj_tensors("coo")
    del coo_g

    sort_idx = torch.argsort(hotness, descending=True)

    reorder_map = torch.argsort(sort_idx)

    reorder_row = reorder_map[coo_row]
    reorder_col = reorder_map[coo_col]
    coo_sort_idx = torch.argsort(reorder_row)
    reorder_row = reorder_row[coo_sort_idx]
    reorder_col = reorder_col[coo_sort_idx]
    coo_g = dgl.graph(("coo", (reorder_row, reorder_col)))
    csc_g = coo_g.formats("csc")
    csc_g.create_formats_()
    reorder_indptr, reorder_indices, _ = csc_g.adj_tensors("csc")

    reorder_train_idx = reorder_map[train_idx]
    reorder_val_idx = reorder_map[val_idx]
    reorder_test_idx = reorder_map[test_idx]

    reorder_labels = labels[sort_idx]
    reorder_features = features[sort_idx]

    torch.save(reorder_features, os.path.join(save_path, "features.pt"))
    torch.save(reorder_labels, os.path.join(save_path, "labels.pt"))
    torch.save(reorder_indptr, os.path.join(save_path, "indptr.pt"))
    torch.save(reorder_indices, os.path.join(save_path, "indices.pt"))
    torch.save(reorder_train_idx, os.path.join(save_path, "train_idx.pt"))
    torch.save(reorder_val_idx, os.path.join(save_path, "valid_idx.pt"))
    torch.save(reorder_test_idx, os.path.join(save_path, "test_idx.pt"))
    torch.save(meta_data, os.path.join(save_path, "metadata.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/data")
    parser.add_argument("--hotness_path", type=str)
    parser.add_argument("--save_path", type=str, default="/data")
    args = parser.parse_args()

    hotness = torch.load(args.hotness_path)
    graph_reorder(args.root, hotness, args.save_path)
