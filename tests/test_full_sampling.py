import torch
import bifeat

indptr = torch.tensor([0, 4, 7, 10, 12, 12, 12])
indices = torch.tensor([1, 2, 3, 5, 0, 2, 4, 0, 1, 4, 4, 5])
full_sampler = bifeat.StructureCacheServer(indptr, indices, [0])
frontier, seeds, blocks = full_sampler.sample_neighbors(torch.tensor([0, 1]))
print(frontier)
print(seeds)
print(blocks)
