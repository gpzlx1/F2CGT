#include <stdio.h>
#include <torch/script.h>
#include "../common.h"
#include "cuda_ops.h"

#define WARP_SIZE 32

namespace bifeat {

template <typename DstType, typename SrcType>
__global__ void sq_decompress_kernel(
    DstType* __restrict__ output, SrcType* __restrict__ input,
    const int64_t input_dim, float* __restrict__ codebooks,
    const int64_t codebook_dim, const int64_t* codebook_indices,
    const int64_t feat_dim, const int64_t num_items) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t warp_id = idx / WARP_SIZE;
  int64_t lane_id = idx % WARP_SIZE;
  int64_t num_warps = blockDim.x * gridDim.x / WARP_SIZE;

  for (int i = warp_id; i < num_items; i += num_warps) {
    DstType* output_ptr = output + i * feat_dim;
    SrcType* input_ptr = input + i * input_dim;

    float* codebook_ptr = codebooks + codebook_indices[i] * codebook_dim;
    float emin = codebook_ptr[0];
    float emax = codebook_ptr[1];
    float mean = codebook_ptr[2];
    float drange = codebook_ptr[3];
    float target_bits = codebook_ptr[4];

    if (target_bits == 8) {
      for (int j = lane_id; j < feat_dim; j += WARP_SIZE) {
        DstType exp = *(input_ptr + j);
        exp = exp + 0.5;
        DstType sign = exp > 0 ? 1 : -1;
        exp = fabs(exp) * ((emax - emin) / drange) + emin;
        exp = exp2f(exp) * sign;
        *(output_ptr + j) = exp;
      }

    } else if (target_bits == 4) {
      int8_t mask = 0xf;  // 00001111
      for (int j = lane_id; j < feat_dim; j += WARP_SIZE) {
        int target = 4 * (1 - j / input_dim);
        DstType exp = DstType((*(input_ptr + j % input_dim) >> target) & mask);
        exp = exp - drange + 0.5;
        DstType sign = exp > 0 ? 1 : -1;
        exp = fabs(exp) * ((emax - emin) / drange) + emin;
        exp = exp2f(exp) * sign;
        *(output_ptr + j) = exp;
      }

    } else if (target_bits == 2) {
      int8_t mask = 0x3;  // 00000011
      for (int j = lane_id; j < feat_dim; j += WARP_SIZE) {
        int target = 2 * (3 - j / input_dim);
        DstType exp = DstType((*(input_ptr + j % input_dim) >> target) & mask);
        exp = exp - drange + 0.5;
        DstType sign = exp > 0 ? 1 : -1;
        exp = fabs(exp) * ((emax - emin) / drange) + emin;
        exp = exp2f(exp) * sign;
        *(output_ptr + j) = exp;
      }

    } else if (target_bits == 1) {
      int8_t mask = 0x1;  // 00000001
      for (int j = lane_id; j < feat_dim; j += WARP_SIZE) {
        int target = 1 * (7 - j / input_dim);
        SrcType exp = (*(input_ptr + j % input_dim) >> target) & mask;
        *(output_ptr + j) = (DstType(exp) - 0.5) * (2 * mean);
      }
    }
  }
}

torch::Tensor sq_decompress(torch::Tensor codebook_indices,
                            torch::Tensor compressed_features,
                            torch::Tensor codebooks, int64_t feat_dim) {
  // codebook: [emin, emax, mean, drange, target_bits]
  int64_t num_items = compressed_features.size(0);
  int64_t input_dim = compressed_features.size(1);
  int64_t codebook_dim = codebooks.size(1);

  torch::Tensor output =
      torch::zeros({num_items, feat_dim},
                   torch::dtype(torch::kFloat32).device(torch::kCUDA));

  int64_t num_threads = 256;
  int64_t num_blocks = (num_threads + num_items - 1) / num_threads;

  PG_INT_TYPE_SWITCH(compressed_features.dtype(), SrcType, {
    sq_decompress_kernel<float, SrcType><<<num_blocks, num_threads>>>(
        output.data_ptr<float>(), compressed_features.data_ptr<SrcType>(),
        input_dim, codebooks.data_ptr<float>(), codebook_dim,
        codebook_indices.data_ptr<int64_t>(), feat_dim, num_items);
  });

  return output;
}

////////////////////////////////////////////////////////////////////////////////
template <typename DstType, typename SrcType, typename IndexType>
__global__ void sq_decompress_kernel_v2(
    DstType* __restrict__ output, SrcType* __restrict__ input,
    const int64_t input_dim, float* __restrict__ codebooks,
    const int64_t codebook_dim, const IndexType* local_index,
    const int64_t feat_dim, const int64_t num_items, const int64_t chunk_size) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t warp_id = idx / WARP_SIZE;
  int64_t lane_id = idx % WARP_SIZE;
  int64_t num_warps = blockDim.x * gridDim.x / WARP_SIZE;

  for (int i = warp_id; i < num_items; i += num_warps) {
    DstType* output_ptr = output + i * feat_dim;
    SrcType* input_ptr = input + i * input_dim;
    IndexType codebook_indices = local_index[i] / chunk_size;

    float* codebook_ptr = codebooks + codebook_indices * codebook_dim;
    float emin = codebook_ptr[0];
    float emax = codebook_ptr[1];
    float mean = codebook_ptr[2];
    float drange = codebook_ptr[3];
    float target_bits = codebook_ptr[4];

    if (target_bits == 8) {
      for (int j = lane_id; j < feat_dim; j += WARP_SIZE) {
        DstType exp = *(input_ptr + j);
        exp = exp + 0.5;
        DstType sign = exp > 0 ? 1 : -1;
        exp = fabs(exp) * ((emax - emin) / drange) + emin;
        exp = exp2f(exp) * sign;
        *(output_ptr + j) = exp;
      }

    } else if (target_bits == 4) {
      int8_t mask = 0xf;  // 00001111
      for (int j = lane_id; j < feat_dim; j += WARP_SIZE) {
        int target = 4 * (1 - j / input_dim);
        DstType exp = DstType((*(input_ptr + j % input_dim) >> target) & mask);
        exp = exp - drange + 0.5;
        DstType sign = exp > 0 ? 1 : -1;
        exp = fabs(exp) * ((emax - emin) / drange) + emin;
        exp = exp2f(exp) * sign;
        *(output_ptr + j) = exp;
      }

    } else if (target_bits == 2) {
      int8_t mask = 0x3;  // 00000011
      for (int j = lane_id; j < feat_dim; j += WARP_SIZE) {
        int target = 2 * (3 - j / input_dim);
        DstType exp = DstType((*(input_ptr + j % input_dim) >> target) & mask);
        exp = exp - drange + 0.5;
        DstType sign = exp > 0 ? 1 : -1;
        exp = fabs(exp) * ((emax - emin) / drange) + emin;
        exp = exp2f(exp) * sign;
        *(output_ptr + j) = exp;
      }

    } else if (target_bits == 1) {
      int8_t mask = 0x1;  // 00000001
      for (int j = lane_id; j < feat_dim; j += WARP_SIZE) {
        int target = 1 * (7 - j / input_dim);
        SrcType exp = (*(input_ptr + j % input_dim) >> target) & mask;
        *(output_ptr + j) = (DstType(exp) - 0.5) * (2 * mean);
      }
    }
  }
}

void sq_decompress_v2(torch::Tensor index, int64_t chunk_size,
                      torch::Tensor compressed_features,
                      torch::Tensor codebooks, torch::Tensor output,
                      int64_t output_offset) {
  // codebook: [emin, emax, mean, drange, target_bits]
  int64_t num_items = compressed_features.size(0);
  int64_t input_dim = compressed_features.size(1);
  int64_t codebook_dim = codebooks.size(1);

  CHECK(output.dtype() == torch::kFloat32);

  int64_t feat_dim = output.size(1);

  int64_t num_threads = 256;
  int64_t num_blocks = (num_threads + num_items - 1) / num_threads;

  PG_INT_TYPE_SWITCH(compressed_features.dtype(), SrcType, {
    PG_INT_TYPE_SWITCH(index.dtype(), IndexType, {
      sq_decompress_kernel_v2<float, SrcType, IndexType>
          <<<num_blocks, num_threads>>>(
              output.data_ptr<float>() + output_offset * feat_dim,
              compressed_features.data_ptr<SrcType>(), input_dim,
              codebooks.data_ptr<float>(), codebook_dim,
              index.data_ptr<IndexType>(), feat_dim, num_items, chunk_size);
    });
  });
}

}  // namespace bifeat
