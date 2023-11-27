import torch
import time
from bifeat.cache import compute_feat_slope

torch.manual_seed(1)

length = 10_000_000
dim = 128
feat = torch.randn((length, dim)).float()
heat = torch.randint(0, 200, (length, ))

cache_rate = 0
step = 0.1
num_step = 10

num_iters = 30
fetch_size = 1_000_000

tic = time.time()
slope = compute_feat_slope(feat, heat, fetch_size, num_iters, step, num_step)
toc = time.time()
print(slope)
print(toc - tic)
