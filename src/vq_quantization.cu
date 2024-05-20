#include "common.h"
#include "cuda_operators.h"
#include "cuda_utils.cuh"

template <typename SrcType, typename DstType, typename IndexType,
          int WARP_SIZE = 32>
__global__ void vq_decompress_kernel(DstType* output, IndexType* input,
                                     SrcType* codebooks, int64_t num_items,
                                     int64_t feat_dim, int64_t num_codebooks,
                                     int length, int64_t column_slice) {
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id = thread_id / 32;
  int lane_id = thread_id % 32;
  int num_warps = gridDim.x * blockDim.x / 32;

  // num_codebooks, length, column_slice for codebooks

  for (int i = warp_id; i < num_items; i += num_warps) {
    DstType* output_ptr = output + i * feat_dim;
    IndexType* input_ptr = input + i * num_codebooks;

    for (int j = lane_id; j < feat_dim; j += WARP_SIZE) {
      int codebook_index = j / column_slice;
      IndexType index = input_ptr[codebook_index];
      int offset = j % column_slice;
      SrcType* codebook_ptr = codebooks +
                              codebook_index * length * column_slice +
                              index * column_slice;

      output_ptr[j] = codebook_ptr[offset];
    }
  }
}

torch::Tensor vq_decompress(torch::Tensor index_tensor, torch::Tensor codebooks,
                            int64_t feat_dim) {
  int64_t num_codebooks = codebooks.size(0);
  int64_t length = codebooks.size(1);
  int64_t column_slice = codebooks.size(2);
  int64_t num_items = index_tensor.size(0);

  int64_t block_size = 256;
  int64_t num_blocks = (num_items + block_size - 1) / block_size;

  torch::Tensor output = torch::zeros(
      {num_items, feat_dim},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

  PG_INT_TYPE_SWITCH(index_tensor.dtype(), IndexType, {
    PG_FLOAT_TYPE_SWITCH(codebooks.dtype(), SrcType, {
      vq_decompress_kernel<SrcType, float, IndexType>
          <<<num_blocks, block_size>>>(
              output.data_ptr<float>(), index_tensor.data_ptr<IndexType>(),
              codebooks.data_ptr<SrcType>(), num_items, feat_dim, num_codebooks,
              length, column_slice);
    });
  });

  return output;
}