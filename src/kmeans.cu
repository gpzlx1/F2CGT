#include <raft/cluster/kmeans.cuh>
#include <raft/cluster/kmeans_types.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/mdarray.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft_runtime/distance/pairwise_distance.hpp>
// #include <raft/core/device_mdspan.hpp>

#include "common.h"
#include "kmeans.h"

KMeans::KMeans(int64_t cluster_num, int64_t feat_dim, std::string metric,
               int max_iter, double tol, int n_init, double oversampling_factor,
               int batch_samples, int batch_centroids, bool inertia_check) {
  if (metric == "cosine") {
    params.metric = raft::distance::DistanceType::CosineExpanded;
    params.init = raft::cluster::KMeansParams::Random;
  } else {
    params.metric = raft::distance::DistanceType::L2Unexpanded;
    params.init = raft::cluster::KMeansParams::KMeansPlusPlus;
  }
  params.n_clusters = cluster_num;
  params.max_iter = max_iter;
  params.tol = tol;
  params.n_init = n_init;
  params.oversampling_factor = oversampling_factor;
  params.batch_samples = batch_samples;
  params.batch_centroids = batch_centroids;
  params.inertia_check = inertia_check;
  cluster_centers = torch::zeros(
      {cluster_num, feat_dim},
      torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
  num_clusters = cluster_num;
}

KMeans::KMeans(torch::Tensor centers, std::string metric, int max_iter,
               double tol, int n_init, double oversampling_factor,
               int batch_samples, int batch_centroids, bool inertia_check) {
  CHECK_CUDA(centers);
  if (metric == "cosine") {
    params.metric = raft::distance::DistanceType::CosineExpanded;
  } else {
    params.metric = raft::distance::DistanceType::L2Unexpanded;
  }
  params.n_clusters = centers.size(0);
  params.max_iter = max_iter;
  params.tol = tol;
  params.n_init = n_init;
  params.oversampling_factor = oversampling_factor;
  params.batch_samples = batch_samples;
  params.batch_centroids = batch_centroids;
  params.inertia_check = inertia_check;
  params.init = raft::cluster::KMeansParams::Array;
  cluster_centers = centers;
  num_clusters = centers.size(0);
}

KMeans::~KMeans() {}

void KMeans::fit(torch::Tensor x) {
  CHECK_CUDA(x);
  auto x_view = raft::make_device_matrix_view<float, int64_t>(
      x.data_ptr<float>(), (int64_t)x.size(0), (int64_t)x.size(1));
  std::optional<raft::device_vector_view<float, int64_t>> sw = std::nullopt;
  auto centroids_view = raft::make_device_matrix_view<float, int64_t>(
      cluster_centers.data_ptr<float>(), (int64_t)params.n_clusters,
      (int64_t)x.size(1));
  float inertia = 0.0;
  int64_t n_iter = 0;
  auto inertia_view = raft::make_host_scalar_view<float>(&inertia);
  auto n_iter_view = raft::make_host_scalar_view<int64_t>(&n_iter);
  raft::cluster::kmeans::fit<float, int64_t>(
      handle, params, x_view, sw, centroids_view, inertia_view, n_iter_view);
}

torch::Tensor KMeans::predict(torch::Tensor x, bool normalize_weights) {
  CHECK_CUDA(x);
  torch::Tensor labels = torch::zeros(
      x.size(0),
      torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt64));
  auto x_view = raft::make_device_matrix_view<float, int64_t>(
      x.data_ptr<float>(), (int64_t)x.size(0), (int64_t)x.size(1));
  std::optional<raft::device_vector_view<float, int64_t>> sw = std::nullopt;
  auto centroids_view = raft::make_device_matrix_view<float, int64_t>(
      cluster_centers.data_ptr<float>(), (int64_t)params.n_clusters,
      (int64_t)x.size(1));
  auto labels_view = raft::make_device_vector_view<int64_t, int64_t>(
      labels.data_ptr<int64_t>(), (int64_t)x.size(0));
  float inertia = 0.0;
  auto inertia_view = raft::make_host_scalar_view<float>(&inertia);

  raft::cluster::kmeans_predict<float, int64_t>(
      handle, params, x_view, sw, centroids_view, labels_view,
      normalize_weights, inertia_view);

  return labels;
}

torch::Tensor KMeans::get_centers() { return cluster_centers; }

torch::Tensor pairwise_distance(torch::Tensor x, torch::Tensor y,
                                std::string metric) {
  torch::Tensor dist = torch::empty(
      {x.size(0), y.size(0)},
      torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
  raft::distance::DistanceType dist_type;
  if (metric == "cosine") {
    dist_type = raft::distance::DistanceType::CosineExpanded;
  } else {
    dist_type = raft::distance::DistanceType::L2Unexpanded;
  }
  raft::device_resources handle;
  raft::runtime::distance::pairwise_distance(
      handle, x.data_ptr<float>(), y.data_ptr<float>(), dist.data_ptr<float>(),
      x.size(0), y.size(0), x.size(1), dist_type, true, 2.0);
  return dist;
}