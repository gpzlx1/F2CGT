#ifndef PG_PG_OPS_H_
#define PG_PG_OPS_H_

#include <torch/script.h>

namespace pg {

torch::Tensor FeatureFetchDataWithCachingCUDA(torch::Tensor cpu_data,
                                              torch::Tensor gpu_data,
                                              torch::Tensor nid,
                                              int64_t cached_num);

torch::Tensor FeatureFetchDataCUDA(torch::Tensor data, torch::Tensor nid);

void meanaggr(torch::Tensor &output, const torch::Tensor &input,
              const torch::Tensor &src, const torch::Tensor &dst, int64_t dim,
              int64_t node_num, int64_t edge_num);
};  // namespace pg

#endif