#include <iostream>

#include <bcht.hpp>
#include "./hashmap_v2.h"

namespace bifeat {
namespace hashmap {

using pair_type = bght::pair<int32_t, int32_t>;

__global__ void InitPair(pair_type* pair, int32_t* key, int32_t* value,
                         int64_t capacity) {
  for (int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x;
       thread_idx < capacity; thread_idx += gridDim.x * blockDim.x) {
    pair[thread_idx].first = key[thread_idx];
    pair[thread_idx].second = value[thread_idx];
  }
}

BiFeatHashmaps::BiFeatHashmaps(int64_t hashmap_num,
                               std::vector<torch::Tensor> cache_nids) {
  memory_usage_ = 0;
  hashmap_num_ = hashmap_num;
  CHECK(cache_nids.size() == hashmap_num_);
  CHECK(hashmap_num_ <= 2);

  for (int i = 0; i < hashmap_num_; i++) {
    auto cache_nid = cache_nids[i];
    int64_t num_elem = cache_nid.numel();
    int64_t capacity = int64_t(num_elem * 1.5);

    // pre-allocate memory
    torch::Tensor key_tensor = cache_nid.to(torch::kInt32).to(torch::kCUDA);
    torch::Tensor value_tensor = torch::arange(
        num_elem,
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

    pair_type* pair_;
    cudaMalloc(&pair_, int64_t(int64_t(num_elem) * sizeof(pair_type)));
    int block_size = 1024;
    int grid_size = (num_elem + block_size - 1) / block_size;
    InitPair<<<grid_size, block_size>>>(pair_, key_tensor.data_ptr<int32_t>(),
                                        value_tensor.data_ptr<int32_t>(),
                                        num_elem);

    // create hashmap
    auto hashmap_ptr = new bght::bcht<int32_t, int32_t>(capacity, -1, -1);
    memory_usage_ = capacity * sizeof(pair_type);
    hashmap_ptr->insert(pair_, pair_ + num_elem);

    // free
    cudaFree(pair_);

    if (i == 0)
      hashmap1_ = hashmap_ptr;
    else
      hashmap2_ = hashmap_ptr;
  }
}

BiFeatHashmaps::~BiFeatHashmaps() {
  for (int i = 0; i < hashmap_num_; i++) {
    if (i == 0) {
      delete static_cast<bght::bcht<int32_t, int32_t>*>(hashmap1_);
    } else {
      delete static_cast<bght::bcht<int32_t, int32_t>*>(hashmap2_);
    }
  }
}

torch::Tensor BiFeatHashmaps::query(torch::Tensor keys,
                                    int64_t first_part_size) {
  CHECK(keys.device().is_cuda());

  int64_t num_elem = keys.numel();

  CHECK(first_part_size <= num_elem);

  if (hashmap_num_ == 1) {
    first_part_size = num_elem;
  }

  torch::Tensor query = keys.to(torch::kInt32);
  torch::Tensor result = torch::zeros_like(query);

  // for fisrt part
  if (first_part_size > 0) {
    bght::bcht<int32_t, int32_t>* hashmap_ptr =
        static_cast<bght::bcht<int32_t, int32_t>*>(hashmap1_);
    int32_t* result_ptr = result.data_ptr<int32_t>();
    int32_t* query_ptr = query.data_ptr<int32_t>();

    hashmap_ptr->find(query_ptr, query_ptr + first_part_size, result_ptr);
  }

  // for second part
  if (num_elem - first_part_size > 0) {
    bght::bcht<int32_t, int32_t>* hashmap_ptr =
        static_cast<bght::bcht<int32_t, int32_t>*>(hashmap2_);
    int32_t* result_ptr = result.data_ptr<int32_t>() + first_part_size;
    int32_t* query_ptr = query.data_ptr<int32_t>() + first_part_size;

    hashmap_ptr->find(query_ptr, query_ptr + num_elem - first_part_size,
                      result_ptr);
  }

  return result;
}

}  // namespace hashmap

}  // namespace bifeat