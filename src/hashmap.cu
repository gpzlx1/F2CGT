#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include "common.h"
#include "hashmap.h"
#include "pg_ops.h"

namespace pg {
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

}  // namespace pg
