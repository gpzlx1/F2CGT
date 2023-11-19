import torch
import os


def load_dataset(path, dataset_name, with_feature=True):
    print("load {}...".format(dataset_name))

    meta_data = torch.load(os.path.join(path, "metadata.pt"))
    assert meta_data["dataset"] == dataset_name

    labels = torch.load(os.path.join(path, "labels.pt"))
    indptr = torch.load(os.path.join(path, "indptr.pt"))
    indices = torch.load(os.path.join(path, "indices.pt"))
    train_idx = torch.load(os.path.join(path, "train_idx.pt"))

    graph_tensors = {
        "labels": labels,
        "indptr": indptr,
        "indices": indices,
        "train_idx": train_idx
    }

    if with_feature:
        features = torch.load(os.path.join(path, "features.pt"))
        graph_tensors["features"]  = features
        graph_tensors["features"] = features
    print("finish loading {}...".format(dataset_name))

    return graph_tensors, meta_data["num_classes"]