#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <torch/script.h>

#include "../common.h"
#include "cub_function.h"
#include "cuda_ops.h"

#define BLOCK_SIZE 128

namespace bifeat {

template <typename IdType, typename ValueType, int TILE_SIZE>
__global__ void _GetSubEdgeDataKernel(
    const int64_t num_items, const IdType *__restrict__ const nids,
    const IdType *__restrict__ const indptr,
    const ValueType *__restrict__ const edge_data,
    const IdType *__restrict__ const sub_indptr,
    ValueType *__restrict__ const sub_edge_data) {
  assert(blockDim.x == BLOCK_SIZE);

  int64_t curr_item = blockIdx.x * TILE_SIZE;
  const int64_t last_item =
      min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_items);

  while (curr_item < last_item) {
    const int64_t nid = nids[curr_item];
    const int64_t in_start = indptr[nid];
    const int64_t degree = indptr[nid + 1] - in_start;
    const int64_t out_start = sub_indptr[curr_item];

    for (int idx = threadIdx.x; idx < degree; idx += BLOCK_SIZE) {
      sub_edge_data[out_start + idx] = edge_data[in_start + idx];
    }

    curr_item += 1;
  }
}

torch::Tensor GetSubIndptr(torch::Tensor nids, torch::Tensor indptr) {
  CHECK_CUDA(nids);
  PG_ID_TYPE_SWITCH(indptr.dtype(), IdType, {
    int64_t num_items = nids.numel();
    torch::Tensor sub_indptr = torch::empty(
        {
            num_items + 1,
        },
        torch::TensorOptions().dtype(indptr.dtype()).device(torch::kCUDA));
    using it = thrust::counting_iterator<IdType>;
    thrust::for_each(
        it(0), it(num_items),
        [in_nids = nids.data_ptr<IdType>(),
         in_indptr = indptr.data_ptr<IdType>(),
         out_indptr =
             sub_indptr.data_ptr<IdType>()] __device__(int64_t i) mutable {
          IdType nid = in_nids[i];
          IdType begin = in_indptr[nid];
          IdType end = in_indptr[nid + 1];
          out_indptr[i] = end - begin;
        });
    thrust::device_ptr<IdType> item_prefix(
        static_cast<IdType *>(sub_indptr.data_ptr<IdType>()));
    cub_exclusiveSum<IdType>(thrust::raw_pointer_cast(item_prefix),
                             num_items + 1);

    return sub_indptr;
  });
  return torch::Tensor();
}

torch::Tensor GetSubEdgeData(torch::Tensor nids, torch::Tensor indptr,
                             torch::Tensor sub_indptr,
                             torch::Tensor edge_data) {
  CHECK_CUDA(nids);
  PG_ID_TYPE_SWITCH(indptr.dtype(), IdType, {
    PG_VALUE_TYPE_SWITCH(edge_data.dtype(), ValueType, {
      int64_t num_items = nids.numel();
      thrust::device_ptr<IdType> item_prefix(
          static_cast<IdType *>(sub_indptr.data_ptr<IdType>()));
      torch::Tensor sub_edge_data = torch::empty(
          {
              item_prefix[num_items],
          },
          torch::TensorOptions().dtype(edge_data.dtype()).device(torch::kCUDA));
      constexpr int TILE_SIZE = 128 / BLOCK_SIZE;
      const dim3 block(BLOCK_SIZE);
      const dim3 grid((num_items + TILE_SIZE - 1) / TILE_SIZE);
      _GetSubEdgeDataKernel<IdType, ValueType, TILE_SIZE><<<grid, block>>>(
          num_items, nids.data_ptr<IdType>(), indptr.data_ptr<IdType>(),
          edge_data.data_ptr<ValueType>(), sub_indptr.data_ptr<IdType>(),
          sub_edge_data.data_ptr<ValueType>());

      return sub_edge_data;
    });
  });
  return torch::Tensor();
}

}  // namespace bifeat