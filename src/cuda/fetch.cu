#include "../common.h"
#include "cuda_ops.h"

#define BLOCK_SIZE 128

namespace bifeat {

template <typename IndexType, typename FloatType, int TILE_SIZE>
__global__ void _FeatureFetchDataWithCachingKernel(
    const int64_t num_nids, const int64_t data_dim, const int64_t cached_num,
    const IndexType *__restrict__ const in_nids,
    const FloatType *__restrict__ const cpu_data,
    const FloatType *__restrict__ const gpu_data,
    FloatType *__restrict__ const out_data) {
  assert(blockDim.x == BLOCK_SIZE);

  int64_t out_node = blockIdx.x * TILE_SIZE;
  const int64_t last_node =
      min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_nids);

  while (out_node < last_node) {
    int64_t nid = in_nids[out_node];
    bool cached = nid < cached_num;

    if (cached) {
      for (int idx = threadIdx.x; idx < data_dim; idx += BLOCK_SIZE) {
        out_data[out_node * data_dim + idx] = gpu_data[nid * data_dim + idx];
      }
    } else {
      for (int idx = threadIdx.x; idx < data_dim; idx += BLOCK_SIZE) {
        out_data[out_node * data_dim + idx] = cpu_data[nid * data_dim + idx];
      }
    }
    out_node += 1;
  }
}

torch::Tensor FeatureFetchDataWithCachingCUDA(torch::Tensor cpu_data,
                                              torch::Tensor gpu_data,
                                              torch::Tensor nid,
                                              int64_t cached_num) {
  CHECK_CUDA(gpu_data);
  CHECK_CUDA(nid);
  PG_ID_TYPE_SWITCH(nid.dtype(), IndexType, {
    PG_VALUE_TYPE_SWITCH(gpu_data.dtype(), FloatType, {
      int64_t num_items = nid.numel();
      int64_t dim = gpu_data.size(1);
      torch::Tensor data_buff =
          torch::empty({num_items, dim}, gpu_data.options());

      constexpr int TILE_SIZE = 128 / BLOCK_SIZE;
      const dim3 block(BLOCK_SIZE);
      const dim3 grid((num_items + TILE_SIZE - 1) / TILE_SIZE);
      _FeatureFetchDataWithCachingKernel<IndexType, FloatType, TILE_SIZE>
          <<<grid, block>>>(
              num_items, dim, cached_num, nid.data_ptr<IndexType>(),
              cpu_data.data_ptr<FloatType>(), gpu_data.data_ptr<FloatType>(),
              data_buff.data_ptr<FloatType>());

      return data_buff;
    });
  });

  return torch::Tensor();
}

template <typename IndexType, typename FloatType, int TILE_SIZE>
__global__ void _FeatureFetchDataKernel(
    const int64_t num_nids, const int64_t data_dim,
    const IndexType *__restrict__ const in_nids,
    const FloatType *__restrict__ const data,
    FloatType *__restrict__ const out_data) {
  assert(blockDim.x == BLOCK_SIZE);

  int64_t out_node = blockIdx.x * TILE_SIZE;
  const int64_t last_node =
      min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_nids);

  while (out_node < last_node) {
    for (int idx = threadIdx.x; idx < data_dim; idx += BLOCK_SIZE) {
      out_data[out_node * data_dim + idx] =
          data[in_nids[out_node] * data_dim + idx];
    }
    out_node += 1;
  }
}

torch::Tensor FeatureFetchDataCUDA(torch::Tensor data, torch::Tensor nid) {
  CHECK_CUDA(nid);
  PG_ID_TYPE_SWITCH(nid.dtype(), IndexType, {
    PG_VALUE_TYPE_SWITCH(data.dtype(), FloatType, {
      int num_items = nid.numel();
      int dim = data.size(1);
      torch::Tensor data_buff = torch::empty(
          {num_items, dim},
          torch::TensorOptions().device(torch::kCUDA).dtype(data.dtype()));
      constexpr int TILE_SIZE = 128 / BLOCK_SIZE;
      const dim3 block(BLOCK_SIZE);
      const dim3 grid((num_items + TILE_SIZE - 1) / TILE_SIZE);
      _FeatureFetchDataKernel<IndexType, FloatType, TILE_SIZE><<<grid, block>>>(
          num_items, dim, nid.data_ptr<IndexType>(), data.data_ptr<FloatType>(),
          data_buff.data_ptr<FloatType>());

      return data_buff;
    });
  });

  return torch::Tensor();
}

}  // namespace bifeat