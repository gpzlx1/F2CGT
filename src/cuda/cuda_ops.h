#ifndef PG_PG_OPS_H_
#define PG_PG_OPS_H_

#include <torch/script.h>

namespace bifeat {
std::tuple<torch::Tensor, torch::Tensor> CreateHashMapTensorCUDA(
    torch::Tensor cache_nids);

torch::Tensor FeatureFetchDataWithCachingCUDA(
    torch::Tensor cpu_data, torch::Tensor gpu_data, torch::Tensor nid,
    torch::Tensor hashed_key_tensor, torch::Tensor hashed_value_tensor);

torch::Tensor FeatureFetchDataCUDA(torch::Tensor data, torch::Tensor nid);

void meanaggr(torch::Tensor &output, const torch::Tensor &input,
              const torch::Tensor &src, const torch::Tensor &dst, int64_t dim,
              int64_t node_num, int64_t edge_num);

torch::Tensor vq_decompress(torch::Tensor codebook_indices,
                            torch::Tensor compressed_features,
                            torch::Tensor codebooks, int64_t feat_dim);

torch::Tensor sq_decompress(torch::Tensor codebook_indices,
                            torch::Tensor compressed_features,
                            torch::Tensor codebooks, int64_t feat_dim);
};  // namespace bifeat

#endif