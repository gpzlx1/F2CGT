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

idx = torch.randint(0, metadata["num_nodes"], (100000, ), device="cuda")

# no cache
cache_ids = []
for i in range(num_feat_parts):
    cache_ids.append(torch.tensor([], dtype=torch.int64, device="cuda"))
feature_server.clear_cache()
feature_server.cache_data(cache_ids)
feat1 = feature_server[idx]
print(feat1)
print(feat1.shape)

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
feat2 = feature_server[idx]
print(feat2)
print(feat2.shape)

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
feat3 = feature_server[idx]
print(feat3)
print(feat3.shape)

assert torch.equal(feat1, feat2)
assert torch.equal(feat1, feat3)
