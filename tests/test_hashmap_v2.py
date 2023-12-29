import torch
import BiFeatLib

key1 = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long).cuda()
key2 = torch.tensor([9, 8, 7, 6], dtype=torch.long).cuda()

hashmap_v2 = BiFeatLib.BiFeatHashmaps(2, [key1, key2])

torch.cuda.synchronize()

query = torch.tensor([1, 2, 3, 4, 5, 10, 6, 7, 8, 9, 10]).cuda().int()
print(hashmap_v2.query(query, 0))
print(hashmap_v2.query(query, 10))
print(hashmap_v2.query(query, 5))

del hashmap_v2

# hashmap v2 benchmark
keys = torch.randint(0, 20_000_000, (20_000_000, ), dtype=torch.long).cuda()
keys = torch.unique(keys)
# keys = torch.arange(0, 1000_000, dtype=torch.long).cuda()
hashmap_v2 = BiFeatLib.BiFeatHashmaps(1, [keys])

query = torch.randint(0, 25_000_000, (10_000_000, ), dtype=torch.long).cuda()
result_v2 = hashmap_v2.query(query, query.numel())
print(result_v2)

# hashmap v1 benchmark
hashmap_key_v1, hashmap_value_v1 = BiFeatLib._CAPI_create_hashmap(keys)

result_v1 = BiFeatLib._CAPI_search_hashmap(hashmap_key_v1, hashmap_value_v1,
                                           query)
print(result_v1)

print(result_v1.eq(result_v2).all())

## benchmark
import time

query_v2 = query.int()
for i in range(10):
    start = time.time()
    result_v2 = hashmap_v2.query(query_v2, query_v2.numel())
    torch.cuda.synchronize()
    end = time.time()
    print("hashmap v2 takes {:.3f} ms".format((end - start) * 1000))

print()

for i in range(10):
    start = time.time()
    result_v1 = BiFeatLib._CAPI_search_hashmap(hashmap_key_v1,
                                               hashmap_value_v1, query)
    torch.cuda.synchronize()
    end = time.time()
    print("hashmap v1 takes {:.3f} ms".format((end - start) * 1000))
