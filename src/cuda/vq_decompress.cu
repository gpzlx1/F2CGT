#include <stdio.h>
#include <torch/script.h>
#include "cuda_ops.h"

#define WARP_SIZE 32

namespace bifeat {

template <typename DstType, typename SrcType>
__global__ void vq_compress_kernel(
    DstType* __restrict__ output, const SrcType* __restrict__ input,
    int64_t input_dim, DstType* __restrict__ codebooks,
    int64_t* __restrict__ codebook_indices, int64_t num_parts, int64_t length,
    int64_t width, int64_t feat_dim, int64_t num_items) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t warp_id = idx / WARP_SIZE;
  int64_t lane_id = idx % WARP_SIZE;
  int64_t num_warps = blockDim.x * gridDim.x / WARP_SIZE;

  for (int i = warp_id; i < num_items; i += num_warps) {
    DstType* output_ptr_1 = output + i * feat_dim;
    DstType* input_ptr_1 =
        codebooks + codebook_indices[i] * (num_parts * length * width);

    for (int j = 0; j < num_parts; j++) {
      DstType* output_ptr_2 = output_ptr_1 + j * width;
      DstType* input_ptr_2 =
          input_ptr_1 + j * (length * width) + input[i * input_dim + j] * width;

      for (int k = lane_id; k < width && k < feat_dim - j * width;
           k += WARP_SIZE) {
        *(output_ptr_2 + k) = *(input_ptr_2 + k);
      }
    }
  }
}

torch::Tensor vq_decompress(torch::Tensor codebook_indices,
                            torch::Tensor compressed_features,
                            torch::Tensor codebooks, int64_t feat_dim) {
  // codebooks: [num_codebooks, num_parts, length, width]
  int64_t num_parts = codebooks.size(1);
  int64_t length = codebooks.size(2);
  int64_t width = codebooks.size(3);

  int64_t num_items = compressed_features.size(0);
  int64_t input_dim = compressed_features.size(1);
  torch::Tensor output =
      torch::zeros({num_items, feat_dim},
                   torch::dtype(torch::kFloat32).device(torch::kCUDA));

  int64_t num_threads = 256;
  int64_t num_blocks = (num_items + num_threads - 1) / num_threads;

  vq_compress_kernel<float, uint8_t><<<num_blocks, num_threads>>>(
      output.data_ptr<float>(), compressed_features.data_ptr<uint8_t>(),
      input_dim, codebooks.data_ptr<float>(),
      codebook_indices.data_ptr<int64_t>(), num_parts, length, width, feat_dim,
      num_items);

  return output;
}
}  // namespace bifeat
