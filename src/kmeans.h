#pragma once

#ifdef RAFT_COMPILED
#define RAFT_EXPLICIT_INSTANTIATE_ONLY
#endif

#include <torch/extension.h>
#include <torch/script.h>
#include <raft/cluster/kmeans_types.hpp>
#include <raft/core/device_resources.hpp>
#include <string>

class KMeans {
 public:
  KMeans(int64_t cluster_num, int64_t feat_dim, std::string metric,
         int max_iter, double tol, int n_init, double oversampling_factor,
         int batch_samples, int batch_centroids, bool inertia_check);
  KMeans(torch::Tensor centers, std::string metric, int max_iter, double tol,
         int n_init, double oversampling_factor, int batch_samples,
         int batch_centroids, bool inertia_check);
  ~KMeans();
  void fit(torch::Tensor x);
  torch::Tensor predict(torch::Tensor x, bool normalize_weights);
  torch::Tensor get_centers();

 private:
  int64_t num_clusters;
  raft::cluster::kmeans::KMeansParams params;
  raft::device_resources handle;
  torch::Tensor cluster_centers;
};

torch::Tensor pairwise_distance(torch::Tensor x, torch::Tensor y,
                                std::string metric);