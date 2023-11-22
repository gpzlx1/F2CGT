#pragma once
#include <torch/script.h>
namespace cpu {

void omp_reorder_indices(torch::Tensor indptr, torch::Tensor indices,
                         torch::Tensor new_indptr, torch::Tensor new_indices,
                         torch::Tensor dst2src) {
  int64_t num_nodes = indptr.size(0) - 1;

#pragma omp parallel for
  for (int64_t i = 0; i < num_nodes; i++) {
    int64_t dst = i;
    int64_t src = dst2src.data_ptr<int64_t>()[dst];
    // new_indices[new_indptr[dst].item()] = indices[indptr[src].item()];

    int64_t dst_begin = new_indptr.data_ptr<int64_t>()[dst];
    int64_t src_begin = indptr.data_ptr<int64_t>()[src];
    int64_t length = indptr.data_ptr<int64_t>()[src + 1] - src_begin;

    memcpy(new_indices.data_ptr<int64_t>() + dst_begin,
           indices.data_ptr<int64_t>() + src_begin, length * sizeof(int64_t));
  }
}

}  // namespace cpu
