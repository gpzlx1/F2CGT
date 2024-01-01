import torch
import time
from bifeat.compression import Decompresser
from bifeat.cache import FeatureLoadServer
from load_dataset import load_compressed_dataset

g, metadata, codebooks = load_compressed_dataset(
    "/data/ogbn_products/compressed-2level/", "ogbn-products")
decompresser = Decompresser(metadata["feature_dim"], codebooks,
                            metadata["methods"],
                            [g["core_idx"].shape[0], metadata["num_nodes"]])
feature_server = FeatureLoadServer(g["core_features"], g["core_idx"],
                                   g["features"], decompresser)
num_feat_parts = len(g["features"])

idx = torch.randint(0, metadata["num_nodes"], (1000000, ), device="cuda")

# no cache
cache_ids = torch.tensor([], dtype=torch.int64, device="cuda")
feature_server.clear_cache()
feature_server.cache_data(cache_ids)
for _ in range(10):
    tic = time.time()
    feat1 = feature_server[idx, 1000]
    toc = time.time()
    print(toc - tic)
print()

# full cache
cache_ids = torch.arange(0,
                         metadata["num_nodes"],
                         dtype=torch.int64,
                         device="cuda")
feature_server.clear_cache()
feature_server.cache_data(cache_ids)
for _ in range(10):
    tic = time.time()
    feat2 = feature_server[idx, 1000]
    toc = time.time()
    print(toc - tic)
print()

# part cache
cache_ids = torch.arange(0,
                         int(metadata["num_nodes"] * 0.5),
                         dtype=torch.int64,
                         device="cuda")
feature_server.clear_cache()
feature_server.cache_data(cache_ids)
for _ in range(10):
    tic = time.time()
    feat3 = feature_server[idx, 1000]
    toc = time.time()
    print(toc - tic)
print()

assert torch.equal(feat1, feat2)
assert torch.equal(feat1, feat3)
