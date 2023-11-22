import torch
from bifeat import CompressionManager
import dgl

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

CM.register(g, features, train_seeds)

import time

begin = time.time()
CM.presampling([25, 10])
torch.cuda.synchronize()
end = time.time()
print((end - begin) * 1000)

begin = time.time()
CM.presampling([25, 10])
torch.cuda.synchronize()
end = time.time()
print((end - begin) * 1000)

print(CM.hotness)

CM.graph_reorder()
