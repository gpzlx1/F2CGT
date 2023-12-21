import torch
import time
from bifeat.compression import Decompresser
from bifeat.cache import CompressedFeatureCacheServer
from load_dataset import load_compressed_dataset

g, metadata, codebooks = load_compressed_dataset(
    "/data/ogbn_products/compressed/", "ogbn-products")
decompresser = Decompresser(metadata["feature_dim"], codebooks,
                            metadata["methods"], metadata["part_size"])
feature_server = CompressedFeatureCacheServer(g["features"], decompresser)
num_feat_parts = len(g["features"])

idx = torch.randint(0, metadata["num_nodes"], (1000000, ), device="cuda")

# no cache
cache_ids = []
for i in range(num_feat_parts):
    cache_ids.append(torch.tensor([], dtype=torch.int64, device="cuda"))
feature_server.clear_cache()
feature_server.cache_data(cache_ids)
for _ in range(10):
    tic = time.time()
    feat1 = feature_server[idx]
    toc = time.time()
    print(toc - tic)
print()

# full cache
cache_ids = []
for i in range(num_feat_parts):
    cache_ids.append(
        torch.arange(0,
                     g["features"][i].shape[0],
                     dtype=torch.int64,
                     device="cuda"))
feature_server.clear_cache()
feature_server.cache_data(cache_ids)
for _ in range(10):
    tic = time.time()
    feat2 = feature_server[idx]
    toc = time.time()
    print(toc - tic)
print()

# part cache
cache_ids = []
for i in range(num_feat_parts):
    cache_ids.append(
        torch.arange(0,
                     int(g["features"][i].shape[0] * 0.5),
                     dtype=torch.int64,
                     device="cuda"))
feature_server.clear_cache()
feature_server.cache_data(cache_ids)
for _ in range(10):
    tic = time.time()
    feat3 = feature_server[idx]
    toc = time.time()
    print(toc - tic)
print()

assert torch.equal(feat1, feat2)
assert torch.equal(feat1, feat3)
