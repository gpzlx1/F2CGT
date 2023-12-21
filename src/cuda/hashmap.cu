#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include "../common.h"
#include "cuda_ops.h"
#include "hashmap.h"

#define BLOCK_SIZE 128

namespace bifeat {

torch::Tensor SearchHashMapCUDA(torch::Tensor hash_key, torch::Tensor hash_val,
                                torch::Tensor input_key) {
  CHECK_CUDA(input_key);
  PG_ID_TYPE_SWITCH(input_key.dtype(), IdType, {
    int num_items = input_key.numel();
    int dir_size = hash_key.numel();
    torch::Tensor result = torch::full_like(input_key, -1, input_key.options());
    using it = thrust::counting_iterator<IdType>;
    thrust::for_each(
        it(0), it(num_items),
        [key = hash_key.data_ptr<IdType>(), val = hash_val.data_ptr<int32_t>(),
         in = input_key.data_ptr<IdType>(), out = result.data_ptr<IdType>(),
         size = dir_size] __device__(IdType i) mutable {
          Hashmap<IdType> table(key, val, size);
          const int32_t pos = table.SearchForPos(in[i]);
          if (pos != -1) {
            out[i] = IdType(val[pos]);
          } else {
            out[i] = -1;
          }
        });
    return result;
  });
  return torch::Tensor();
}

std::tuple<torch::Tensor, torch::Tensor> CreateHashMapTensorCUDA(
    torch::Tensor cache_nids) {
  CHECK_CUDA(cache_nids);
  PG_ID_TYPE_SWITCH(cache_nids.dtype(), IdType, {
    int num_items = cache_nids.numel();
    int dir_size = _UpPower(num_items) * 2;

    torch::Tensor key_buff_tensor =
        torch::full({dir_size}, -1, cache_nids.options());
    torch::Tensor value_buff_tensor = torch::full(
        {dir_size}, -1,
        torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32));

    // insert
    using it = thrust::counting_iterator<IdType>;
    thrust::for_each(it(0), it(num_items),
                     [in_key = cache_nids.data_ptr<IdType>(),
                      key_buff = key_buff_tensor.data_ptr<IdType>(),
                      value_buff = value_buff_tensor.data_ptr<int32_t>(),
                      size = dir_size] __device__(int i) mutable {
                       Hashmap<IdType> table(key_buff, value_buff, size);
                       table.Update(in_key[i], i);
                     });

    return std::make_tuple(key_buff_tensor, value_buff_tensor);
  });

  return std::make_tuple(torch::Tensor(), torch::Tensor());
}

template <typename IdType>
__global__ void _CountCachedNidsNumKernel(
    const int64_t num_items, const int64_t dir_size,
    const IdType *__restrict__ const input_nids,
    IdType *__restrict__ const hashed_orig_nids,
    int32_t *__restrict__ const hashed_device_nids,
    int64_t *__restrict__ const count) {
  int64_t curr_item = blockIdx.x * blockDim.x + threadIdx.x;
  if (curr_item < num_items) {
    Hashmap<IdType> table(hashed_orig_nids, hashed_device_nids, dir_size);
    const int64_t pos = table.SearchForPos(input_nids[curr_item]);
    if (pos != -1) {
      atomic::AtomicAdd(count, 1);
    }
  }
}

int64_t CountCachedNidsNum(torch::Tensor input_nids,
                           torch::Tensor hashed_orig_nids,
                           torch::Tensor hashed_device_nids) {
  PG_ID_TYPE_SWITCH(input_nids.dtype(), IdType, {
    int64_t num_items = input_nids.numel();
    int64_t dir_size = hashed_device_nids.numel();

    int64_t local_nids_num = 0;
    int64_t *local_count;
    CUDA_CALL(cudaMalloc((void **)&local_count, sizeof(int64_t)));
    CUDA_CALL(cudaMemcpy(local_count, &local_nids_num, sizeof(int64_t),
                         cudaMemcpyHostToDevice));

    const dim3 block(BLOCK_SIZE);
    const dim3 grid((num_items + BLOCK_SIZE - 1) / BLOCK_SIZE);
    _CountCachedNidsNumKernel<IdType>
        <<<grid, block>>>(num_items, dir_size, input_nids.data_ptr<IdType>(),
                          hashed_orig_nids.data_ptr<IdType>(),
                          hashed_device_nids.data_ptr<int32_t>(), local_count);

    CUDA_CALL(cudaMemcpy(&local_nids_num, local_count, sizeof(int64_t),
                         cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaFree(local_count));

    return local_nids_num;
  });
  return 0;
}

}  // namespace bifeat
