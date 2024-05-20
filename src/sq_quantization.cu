#include "common.h"
#include "cuda_operators.h"
#include "cuda_utils.cuh"

template <typename DataType, int TARGET_BITS, int WARP_SIZE = 32,
          int MAX_CODEBOOK_NUM = 8>
__global__ void sq_compress_kernel(DataType* src, DataType* dst,
                                   float* codebooks, int64_t num_items,
                                   int64_t feat_dim, int64_t column_slice,
                                   int64_t num_codebooks) {
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id = thread_id / WARP_SIZE;
  int lane_id = thread_id % WARP_SIZE;
  int num_warps = gridDim.x * blockDim.x / WARP_SIZE;

  constexpr float drange = 1 << (TARGET_BITS - 1);
  constexpr int codebook_dim = 5;

  __shared__ float codebooks_cache[MAX_CODEBOOK_NUM * codebook_dim];

  // if (TARGET_BITS != 1) {
  //   for (int i = thread_id; i < num_codebooks * codebook_dim; i +=
  //   blockDim.x) {
  //     codebooks_cache[i] = codebooks[i];
  //   }
  // }

  for (int i = warp_id; i < num_items; i += num_warps) {
    int index = i * feat_dim;
    DataType* input_ptr = index + src;
    DataType* output_ptr = index + dst;

    for (int j = lane_id; j < feat_dim; j += WARP_SIZE) {
      if (TARGET_BITS == 1) {
        output_ptr[j] = input_ptr[j] <= 0 ? 0 : 1;
      } else {
        // fmin, fmax, emin, emax, mean in codebook
        int codebook_index = j / column_slice;
        float fmin = codebooks[codebook_index * codebook_dim + 0];
        float fmax = codebooks[codebook_index * codebook_dim + 1];
        float emin = codebooks[codebook_index * codebook_dim + 2];
        float emax = codebooks[codebook_index * codebook_dim + 3];

        // if (thread_id == 0) {
        //   printf("%f, %f, %f, %f\n", fmin, fmax, emin, emax);
        // }

        // begin sq compress
        DataType value = input_ptr[j];
        bool is_neg = value <= 0;
        value = abs(value);
        value = min(max(value, fmin), fmax);
        value = log2f(value);
        value = (value - emin) / (emax - emin) * drange;
        value = floorf(value);
        value = is_neg ? -1 - value : value;
        if (TARGET_BITS < 8) value += drange;

        output_ptr[j] = value;
      }
    }
  }
}

torch::Tensor sq_compress(torch::Tensor src, torch::Tensor codebooks,
                          int64_t target_bits, int64_t column_slice) {
  torch::Tensor output = torch::empty_like(src);
  int64_t feat_dim = src.size(1);
  int64_t numel = src.size(0);
  int64_t num_codebooks = codebooks.size(0);

  int64_t block_size = 256;
  int64_t num_blocks = (numel + block_size - 1) / block_size;

  PG_TARGET_BITS_SWITCH(target_bits, TARGET_BITS, {
    sq_compress_kernel<float, TARGET_BITS><<<num_blocks, block_size>>>(
        src.data_ptr<float>(), output.data_ptr<float>(),
        codebooks.data_ptr<float>(), numel, feat_dim, column_slice,
        num_codebooks);
  });

  return output;
}

template <typename CompressType, typename DecompressType, int TARGET_BITS,
          int WARP_SIZE = 32, int MAX_CODEBOOK_NUM = 8>
__global__ void sq_decompress_kernel(CompressType* src, DecompressType* dst,
                                     float* codebooks, int64_t num_items,
                                     int64_t src_dim, int64_t dst_dim,
                                     int64_t column_slice,
                                     int64_t num_codebooks) {
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id = thread_id / WARP_SIZE;
  int lane_id = thread_id % WARP_SIZE;
  int num_warps = gridDim.x * blockDim.x / WARP_SIZE;

  constexpr float drange = 1 << (TARGET_BITS - 1);
  constexpr int codebook_dim = 5;
  constexpr int pack_size = 8 / TARGET_BITS;

  uint8_t mask;
  if (TARGET_BITS == 1) {
    mask = 0x01;
  } else if (TARGET_BITS == 2) {
    mask = 0x03;
  } else if (TARGET_BITS == 4) {
    mask = 0x0f;
  }

  //__shared__ float codebooks_cache[MAX_CODEBOOK_NUM * codebook_dim];
  //
  // if (TARGET_BITS != 1) {
  //  for (int i = thread_id; i < num_codebooks * codebook_dim; i += blockDim.x)
  //  {
  //    codebooks_cache[i] = codebooks[i];
  //  }
  //}

  for (int i = warp_id; i < num_items; i += num_warps) {
    CompressType* src_ptr = src + i * src_dim;
    DecompressType* dst_ptr = dst + i * dst_dim;

    for (int j = lane_id; j < dst_dim; j += WARP_SIZE) {
      DecompressType value = 0;
      if (TARGET_BITS >= 8) {
        value = (DecompressType)src_ptr[j];

      } else {
        int bit_offset = j % pack_size;
        int src_index = j / pack_size;
        value = (DecompressType)unpackbits_func(
            reinterpret_cast<uint8_t*>(src_ptr) + src_index, bit_offset,
            TARGET_BITS, mask);
      }

      // fmin, fmax, emin, emax, mean in codebook
      int codebook_index = j / column_slice;
      if (TARGET_BITS == 1) {
        float mean = codebooks[codebook_index * codebook_dim + 4];
        dst_ptr[j] = (value - 0.5) * (2 * mean);

      } else {
        float emin = codebooks[codebook_index * codebook_dim + 2];
        float emax = codebooks[codebook_index * codebook_dim + 3];

        if (TARGET_BITS < 8) value -= drange;

        value = value + 0.5;
        int sign = value >= 0 ? 1 : -1;
        value = fabs(value) * ((emax - emin) / drange) + emin;
        value = exp2f(value) * sign;
        dst_ptr[j] = value;
      }
    }
  }
}

torch::Tensor sq_decompress(torch::Tensor compress_data,
                            torch::Tensor codebooks, int64_t target_bits,
                            int64_t column_slice, int64_t feat_dim) {
  int64_t numel = compress_data.size(0);
  int64_t src_dim = compress_data.size(1);
  int64_t num_codebooks = codebooks.size(0);

  torch::Tensor output = torch::zeros(
      {numel, feat_dim},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

  int64_t block_size = 256;
  int64_t num_blocks = (numel + block_size - 1) / block_size;
  PG_INT_TYPE_SWITCH(compress_data.dtype(), CompressType, {
    PG_TARGET_BITS_SWITCH(target_bits, TARGET_BITS, {
      sq_decompress_kernel<CompressType, float, TARGET_BITS>
          <<<num_blocks, block_size>>>(
              compress_data.data_ptr<CompressType>(), output.data_ptr<float>(),
              codebooks.data_ptr<float>(), numel, src_dim, feat_dim,
              column_slice, num_codebooks);
    });
  });

  return output;
}
