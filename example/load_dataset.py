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
        graph_tensors["features"] = features
    print("finish loading {}...".format(dataset_name))

    return graph_tensors, meta_data["num_classes"]


def load_compressed_dataset(path,
                            dataset_name,
                            with_feature=True,
                            with_valid=True,
                            with_test=True):
    print("load {}...".format(dataset_name))

    meta_data = torch.load(os.path.join(path, "metadata.pt"))
    assert meta_data["dataset"] == dataset_name

    labels = torch.load(os.path.join(path, "labels.pt"))
    indptr = torch.load(os.path.join(path, "indptr.pt"))
    indices = torch.load(os.path.join(path, "indices.pt"))
    train_idx = torch.load(os.path.join(path, "train_idx.pt"))

    adj_hotness = torch.load(os.path.join(path, "adj_hotness.pt"))
    feat_hotness = torch.load(os.path.join(path, "feat_hotness.pt"))

    codebooks = torch.load(os.path.join(path, "codebooks.pt"))
    core_codebooks = torch.load(os.path.join(path, "core_codebooks.pt"))

    if with_feature:
        features = torch.load(os.path.join(path, "compressed_features.pt"))
        core_features = torch.load(
            os.path.join(path, "compressed_core_features.pt"))

    if with_valid:
        valid_idx = torch.load(os.path.join(path, "valid_idx.pt"))

    if with_test:
        test_idx = torch.load(os.path.join(path, "test_idx.pt"))

    if not (with_valid or with_test):
        core_idx = train_idx
    else:
        core_idx = torch.load(os.path.join(path, "core_idx.pt"))

    graph_tensors = {
        "labels": labels,
        "indptr": indptr,
        "indices": indices,
        "train_idx": train_idx,
        "adj_hotness": adj_hotness,
        "feat_hotness": feat_hotness,
        "core_idx": core_idx
    }

    if with_feature:
        graph_tensors["features"] = features
        graph_tensors["core_features"] = core_features
    if with_valid:
        graph_tensors["valid_idx"] = valid_idx
    if with_test:
        graph_tensors["test_idx"] = test_idx

    print("finish loading {}...".format(dataset_name))

    return graph_tensors, meta_data, [core_codebooks, codebooks]
