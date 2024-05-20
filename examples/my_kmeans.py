import numpy as np
import torch
import tqdm
import math
import F2CGTLib as capi


def generate_max_cosine_similarity_vectors(N, num_vectors):
    if N % 2 == 0:
        n = N // 2
        angles = np.random.rand(num_vectors, n) * 2 * np.pi
        x = np.cos(angles)
        y = np.sin(angles)
        vectors = np.concatenate((x, y), axis=1)
    else:
        n = N // 2
        radii = np.random.rand(num_vectors)
        angles = np.random.rand(num_vectors, n + 1) * 2 * np.pi
        x = np.cos(angles[:, :n])
        y = np.sin(angles[:, :n+1])
        vectors = radii[:, np.newaxis] * np.concatenate((x, y), axis=1)
    return vectors


def initialize(X, num_clusters, method='random'):
    if method == "random":
        nonzero_idxs = X.norm(dim=1, p=0).nonzero().squeeze()
        num_samples = len(nonzero_idxs)
        indices = np.random.choice(num_samples, num_clusters, replace=False)
        initial_state = X[nonzero_idxs[indices]].cuda()

    elif method == 'cosine':
        vectors = generate_max_cosine_similarity_vectors(
            X.shape[1], num_clusters)
        initial_state = torch.tensor(vectors)

    else:
        raise ValueError

    return initial_state.float().cuda()


def torch_pairwise_distance(data1, data2):
    A = data1.unsqueeze(dim=1)
    B = data2.unsqueeze(dim=0)
    dis = (A - B)**2.0
    dis = dis.sum(dim=-1).squeeze()
    return dis


def torch_pairwise_cosine(data1, data2):
    data1 = data1.unsqueeze(dim=1)
    data2 = data2.unsqueeze(dim=0)
    cosine = 1 - torch.nn.functional.cosine_similarity(data1, data2, dim=2)
    return cosine


def torch_distance(X, Y, metric='cosine'):
    if metric == 'cosine':
        return torch_pairwise_cosine(X, Y)
    elif metric == 'euclidean':
        return torch_pairwise_distance(X, Y)
    else:
        raise ValueError


def raft_distance(X, Y, metric='cosine'):
    X = X.float()
    Y = Y.float()
    return capi.pairwise_distance(X, Y, metric)


def get_centers(X, num_clusters, metric='cosine', tol=1e-4, mode='fast'):
    if mode == 'fast':
        pairwise_distance_function = raft_distance
    else:
        pairwise_distance_function = torch_distance

    X = X.float().cuda()

    # initialize
    if metric == 'cosine':
        initial_state = initialize(X, num_clusters, method='cosine')
    else:
        initial_state = initialize(X, num_clusters, method='random')

    while True:
        initial_state_pre = initial_state

        dis = pairwise_distance_function(X, initial_state, metric=metric)
        # print(torch.isnan(dis).any())
        nan_index = torch.nonzero(torch.isnan(dis))
        if nan_index.numel() > 0:
            dis[nan_index[:, 0], nan_index[:, 1]] = 1

        # update
        initial_state = compute_new_centers(X, dis, initial_state_pre,
                                            num_clusters)

        # compute shift
        center_shift = torch.sum(
            torch.sqrt(torch.sum((initial_state - initial_state_pre)**2,
                                 dim=1)))

        if center_shift**2 < tol:
            break

        if torch.isnan(center_shift**2):
            return None, None

        tol *= 1.2

    return initial_state


def kmeans_predict(X, cluster_centers, metric='cosine', mode='fast'):
    if mode == 'fast':
        pairwise_distance_function = raft_distance
    else:
        pairwise_distance_function = torch_distance

    X = X.float().cuda()
    dis = pairwise_distance_function(X, cluster_centers, metric=metric)
    labels = torch.argmin(dis, dim=1)
    return labels


def compute_new_centers(X, dis, initial_state_pre, num_clusters):
    choice_cluster = torch.argmin(dis, dim=1)
    indices, inverse, cnt = torch.unique(choice_cluster,
                                         return_inverse=True,
                                         return_counts=True)

    if cnt.numel() != num_clusters:
        new_cnt = torch.ones(num_clusters, device='cuda', dtype=torch.float32)
        new_cnt[indices] = cnt.float()
        cnt = new_cnt
    else:
        cnt = cnt.float()
    cnt = cnt.unsqueeze(dim=1)

    initial_state = initial_state_pre.clone()
    initial_state[indices, :] = 0.
    initial_state.index_add_(0, indices[inverse], X)
    initial_state = initial_state / cnt

    return initial_state


class KMeansTrainer:

    def __init__(self, num_clusters, metric='cosine', tol=1e-4, mode='fast'):
        self.num_clusters = num_clusters
        self.metric = metric
        self.mode = mode
        self.tol = tol

        if metric not in ['cosine', 'euclidean']:
            print("metric must be 'cosine' or 'euclidean'")
            raise ValueError

        if mode not in ['fast', 'normal']:
            print("mode must be 'fast' or 'normal'")
            raise ValueError

    def fit(self, data):
        new_centers = get_centers(data,
                                  num_clusters=self.num_clusters,
                                  metric=self.metric,
                                  tol=self.tol,
                                  mode=self.mode)
        self.centers = new_centers

    def get_centers(self):
        return self.centers

    def predict(self, data, normalize_weights=False):
        data = data.to(self.device)
        labels = kmeans_predict(data,
                                self.centers,
                                self.metric,
                                device=self.device,
                                mode=self.mode)
        return labels
