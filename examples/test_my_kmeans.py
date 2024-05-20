import torch
import os
from my_kmeans import KMeansTrainer, raft_distance, torch_distance

torch.manual_seed(42)

tol = 1e-5

if __name__ == "__main__":

    feature = torch.load(
        os.path.join("/home/gpzlx1/workspace/F2CGT_v2/datasets/products",
                     "features.pt"))

    index = torch.randperm(feature.shape[0])[:5_0000]
    X = feature[index, :2]

    n_cluster = 200
    km = KMeansTrainer(n_cluster, metric="cosine", tol=5e-2, mode='normal')
    km.fit(X.cuda())
    '''
    X = torch.load('X.pt')
    initial_state = torch.load('initial_state.pt')

    # print(X)
    # print(initial_state)

    dis1 = raft_distance(X, initial_state, "cosine")

    #print(dis1)
    print(torch.isnan(dis1).any())
    nan_index = torch.nonzero(torch.isnan(dis1))
    print(nan_index.shape)
    dis1_nan_data = dis1[nan_index[:, 0], nan_index[:, 1]]

    dis2 = torch_distance(X, initial_state, "cosine")
    #print(dis2)
    print(torch.isnan(dis2).any())
    dis2_nan_data = dis2[nan_index[:, 0], nan_index[:, 1]]

    print(dis2.max(), dis2.min())

    print((dis2_nan_data == 1).any())
    '''
