import torch
from bifeat import CompressionManager
import dgl
import time

CM = CompressionManager(0, 0, 0, 0)

reddit = dgl.data.RedditDataset(self_loop=True)
g = reddit._graph
features = g.ndata.pop('feat')
train_seeds = torch.nonzero(g.ndata['train_mask'], as_tuple=False).squeeze()
g.ndata.clear()
g.edata.clear()

print(g)
print(features)
print(train_seeds)

indptr, indices, _ = g.adj_tensors('csc')

CM.register(indptr, indices, features, train_seeds)
CM.presampling([25, 10])

print(CM.hotness)
print(CM.hotness.long().sum())

begin = time.time()
CM.graph_reorder()
end = time.time()
print(end - begin)

# CM.presampling([25, 10])
print(CM.hotness)
print(CM.hotness.long().sum())

print()
print(CM.train_seeds.numel())
print(CM.train_seeds)
print()

CM.presampling([25, 10])
print(CM.hotness)
print(CM.hotness.long().sum())
