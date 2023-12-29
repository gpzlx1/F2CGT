#include <stdio.h>
#include <torch/script.h>
#include "../common.h"
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

  PG_INT_TYPE_SWITCH(compressed_features.dtype(), SrcType, {
    vq_compress_kernel<float, SrcType><<<num_blocks, num_threads>>>(
        output.data_ptr<float>(), compressed_features.data_ptr<SrcType>(),
        input_dim, codebooks.data_ptr<float>(),
        codebook_indices.data_ptr<int64_t>(), num_parts, length, width,
        feat_dim, num_items);
  });

  return output;
}

//////////////////////////////////////////////////////////////////

template <typename DstType, typename SrcType, typename IndexType>
__global__ void vq_compress_kernel_v2(
    DstType* __restrict__ output, const SrcType* __restrict__ input,
    int64_t input_dim, DstType* __restrict__ codebooks,
    IndexType* __restrict__ local_index, int64_t num_parts, int64_t length,
    int64_t width, int64_t feat_dim, int64_t num_items, int64_t chunk_size) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t warp_id = idx / WARP_SIZE;
  int64_t lane_id = idx % WARP_SIZE;
  int64_t num_warps = blockDim.x * gridDim.x / WARP_SIZE;

  for (int i = warp_id; i < num_items; i += num_warps) {
    IndexType codebook_index = local_index[i] / chunk_size;
    DstType* output_ptr_1 = output + i * feat_dim;
    DstType* input_ptr_1 =
        codebooks + codebook_index * (num_parts * length * width);

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

void vq_decompress_v2(torch::Tensor index, int64_t chunk_size,
                      torch::Tensor compressed_features,
                      torch::Tensor codebooks, torch::Tensor output,
                      int64_t output_offset) {
  // codebooks: [num_codebooks, num_parts, length, width]
  int64_t num_parts = codebooks.size(1);
  int64_t length = codebooks.size(2);
  int64_t width = codebooks.size(3);

  int64_t num_items = compressed_features.size(0);
  int64_t input_dim = compressed_features.size(1);
  int64_t feat_dim = output.size(1);

  CHECK(output.dtype() == torch::kFloat32);

  int64_t num_threads = 256;
  int64_t num_blocks = (num_items + num_threads - 1) / num_threads;

  PG_INT_TYPE_SWITCH(compressed_features.dtype(), SrcType, {
    PG_INT_TYPE_SWITCH(index.dtype(), IndexType, {
      vq_compress_kernel_v2<float, SrcType, IndexType>
          <<<num_blocks, num_threads>>>(
              output.data_ptr<float>() + output_offset * feat_dim,
              compressed_features.data_ptr<SrcType>(), input_dim,
              codebooks.data_ptr<float>(), index.data_ptr<IndexType>(),
              num_parts, length, width, feat_dim, num_items, chunk_size);
    });
  });
}

}  // namespace bifeat
