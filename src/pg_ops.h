#ifndef PG_PG_OPS_H_
#define PG_PG_OPS_H_

#include <torch/script.h>

namespace pg {
std::tuple<torch::Tensor, torch::Tensor> CreateHashMapTensorCUDA(
    torch::Tensor cache_nids);

torch::Tensor FeatureFetchDataWithCachingCUDA(
    torch::Tensor cpu_data, torch::Tensor gpu_data, torch::Tensor nid,
    torch::Tensor hashed_key_tensor, torch::Tensor hashed_value_tensor);

torch::Tensor FeatureFetchDataCUDA(torch::Tensor data, torch::Tensor nid);
};  // namespace pg

#endif