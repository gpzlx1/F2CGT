#include "common.h"
#include "hashmap.h"
#include "pg_ops.h"

#define BLOCK_SIZE 128

namespace pg {

template <typename IdType, typename IndexType, typename FloatType,
          int TILE_SIZE>
__global__ void _FeatureFetchDataWithCachingKernel(
    const int64_t num_nids, const int64_t dir_size, const int64_t data_dim,
    const IndexType *__restrict__ const in_nids,
    const FloatType *__restrict__ const cpu_data,
    const FloatType *__restrict__ const gpu_data,
    IdType *__restrict__ const hashed_key,
    int32_t *__restrict__ const hashed_value,
    FloatType *__restrict__ const out_data) {
  assert(blockDim.x == BLOCK_SIZE);

  int64_t out_node = blockIdx.x * TILE_SIZE;
  const int64_t last_node =
      min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_nids);
  Hashmap<IdType> table(hashed_key, hashed_value, dir_size);

  while (out_node < last_node) {
    const int64_t pos = table.SearchForPos(in_nids[out_node]);

    if (pos != -1) {
      for (int idx = threadIdx.x; idx < data_dim; idx += BLOCK_SIZE) {
        out_data[out_node * data_dim + idx] =
            gpu_data[hashed_value[pos] * data_dim + idx];
      }
    } else {
      for (int idx = threadIdx.x; idx < data_dim; idx += BLOCK_SIZE) {
        out_data[out_node * data_dim + idx] =
            cpu_data[in_nids[out_node] * data_dim + idx];
      }
    }
    out_node += 1;
  }
}

torch::Tensor FeatureFetchDataWithCachingCUDA(torch::Tensor cpu_data,
                                              torch::Tensor gpu_data,
                                              torch::Tensor hashed_key_tensor,
                                              torch::Tensor hashed_value_tensor,
                                              torch::Tensor nid) {
  CHECK_CUDA(gpu_data);
  CHECK_CUDA(nid);
  CHECK_CUDA(hashed_key_tensor);
  CHECK_CUDA(hashed_value_tensor);
  PG_ID_TYPE_SWITCH(hashed_key_tensor.dtype(), IdType, {
    PG_ID_TYPE_SWITCH(nid.dtype(), IndexType, {
      PG_VALUE_TYPE_SWITCH(gpu_data.dtype(), FloatType, {
        int num_items = nid.numel();
        int dim = gpu_data.size(1);
        int dir_size = hashed_key_tensor.numel();
        torch::Tensor data_buff =
            torch::empty({num_items, dim}, gpu_data.options());

        constexpr int TILE_SIZE = 128 / BLOCK_SIZE;
        const dim3 block(BLOCK_SIZE);
        const dim3 grid((num_items + TILE_SIZE - 1) / TILE_SIZE);
        _FeatureFetchDataWithCachingKernel<IdType, IndexType, FloatType,
                                           TILE_SIZE><<<grid, block>>>(
            num_items, dir_size, dim, nid.data_ptr<IndexType>(),
            cpu_data.data_ptr<FloatType>(), gpu_data.data_ptr<FloatType>(),
            hashed_key_tensor.data_ptr<IdType>(),
            hashed_value_tensor.data_ptr<int32_t>(),
            data_buff.data_ptr<FloatType>());

        return data_buff;
      });
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

}  // namespace pg