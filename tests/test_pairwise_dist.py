import torch
import F2CGTLib as capi


def raft_distance(X, Y, metric='cosine'):
    import cupy as cp
    from pylibraft.distance import pairwise_distance

    X = cp.array(X.cpu().numpy())
    Y = cp.array(Y.cpu().numpy())

    dis = pairwise_distance(X, Y, metric=metric)
    dis = cp.array(dis)
    return torch.tensor(dis.get()).cuda()


x = torch.randn((1000, 100)).cuda()
y = torch.randn((100, 100)).cuda()
dist = capi.pairwise_distance(x, y, "cosine")
print(dist)
print(dist.shape)

dist1 = raft_distance(x, y, "cosine")
print(dist1)

max_diff = (dist1 - dist).abs().max()
print(max_diff, dist1.abs().max(), dist1.abs().min())
