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

std::tuple<torch::Tensor, torch::Tensor> RowWiseSamplingUniformCUDA(
    torch::Tensor seeds, torch::Tensor indptr, torch::Tensor indices,
    int64_t num_picks, bool replace);
std::tuple<torch::Tensor, torch::Tensor> RowWiseSamplingUniformWithCachingCUDA(
    torch::Tensor seeds, torch::Tensor gpu_indptr, torch::Tensor cpu_indptr,
    torch::Tensor gpu_indices, torch::Tensor cpu_indices,
    torch::Tensor orig_nids_hashed, torch::Tensor gpu_nids_hashed,
    int64_t num_picks, bool replace);
std::tuple<torch::Tensor, std::vector<torch::Tensor>> TensorRelabelCUDA(
    std::vector<torch::Tensor> mapping_tensors,
    std::vector<torch::Tensor> requiring_relabel_tensors);
torch::Tensor GetSubIndptr(torch::Tensor nids, torch::Tensor indptr);
torch::Tensor GetSubEdgeData(torch::Tensor nids, torch::Tensor indptr,
                             torch::Tensor sub_indptr, torch::Tensor edge_data);
int64_t CountCachedNidsNum(torch::Tensor input_nids,
                           torch::Tensor hashed_orig_nids,
                           torch::Tensor hashed_device_nids);

void meanaggr(torch::Tensor &output, const torch::Tensor &input,
              const torch::Tensor &src, const torch::Tensor &dst, int64_t dim,
              int64_t node_num, int64_t edge_num);
};  // namespace bifeat

#endif