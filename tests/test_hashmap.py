import BiFeatLib
import torch

hashmap = BiFeatLib.CacheHashMap()
key = torch.arange(0, 200, 2).cuda()
hashmap.insert(key)

search_key = torch.randint(0, 400, (10, )).cuda()
val = hashmap.find(search_key)

print(search_key)
print(val)
