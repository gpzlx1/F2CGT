#include <curand_kernel.h>
#include "common.h"
#include "cuda_operators.h"
#include "cuda_utils.cuh"

/**
 * @brief CUDA kernel of g-SpMM on CSC format.
 * @note it uses node parallel strategy, different threadblocks (on y-axis)
 *       is responsible for the computation on different destination nodes.
 *       Threadblocks on the x-axis are responsible for the computation on
 *       different positions in feature dimension.
 */
template <typename Idx, typename DType>
__global__ void SpMMCsrKernel(const DType* __restrict__ ufeat,
                              DType* __restrict__ out,
                              const Idx* __restrict__ indptr,
                              const Idx* __restrict__ indices, int64_t num_cols,
                              int64_t feat_dim) {
  // SPMM with CSC.
  int ty = blockIdx.x * blockDim.y + threadIdx.y;
  const Idx stride_y = blockDim.y * gridDim.x;
  const int stride_x = blockDim.x * gridDim.y;

  while (ty < num_cols) {
    int tx = blockIdx.y * blockDim.x + threadIdx.x;
    Idx begin = indptr[ty];
    Idx end = indptr[ty + 1];
    float deg = end - begin == 0 ? 1 : end - begin;

    while (tx < feat_dim) {
      DType local_accum = 0;
      for (Idx i = begin; i < end; ++i) {
        Idx index = indices[i] * feat_dim + tx;
        local_accum += ufeat[index];
      }
      local_accum = local_accum / deg;
      int out_pos = ty * feat_dim + tx;
      out[out_pos] += local_accum;
      tx += stride_x;
    }
    ty += stride_y;
  }
}

torch::Tensor SpMMCsr(torch::Tensor ufeat, torch::Tensor indptr,
                      torch::Tensor indices) {
  int num_cols = indptr.numel() - 1;
  int out_len = ufeat.size(1);

  const int ntx = (out_len + 31) / 32 * 32;
  const int nty = 1024 / ntx;
  const int nbx = (num_cols + nty - 1) / nty;
  const dim3 nblks(nbx);
  const dim3 nthrs(ntx, nty);

  torch::Tensor out;

  PG_SPMM_FLOAT_TYPE_SWITCH(ufeat.dtype(), DType, {
    PG_SPMM_INT_TYPE_SWITCH(indptr.dtype(), Idx, {
      out = torch::zeros(
          {num_cols, out_len},
          torch::TensorOptions().dtype(ufeat.dtype()).device(torch::kCUDA));
      SpMMCsrKernel<Idx, DType><<<nblks, nthrs>>>(
          ufeat.data_ptr<DType>(), out.data_ptr<DType>(),
          indptr.data_ptr<Idx>(), indices.data_ptr<Idx>(), num_cols, out_len);
    });
  });

  return out;
}

template <typename Idx, typename DType, typename CompressDType>
__global__ void VqDecompressSpMMCsrFusionKernel(
    CompressDType* __restrict__ compress_feat, DType* __restrict__ codebooks,
    DType* __restrict__ out, const Idx* __restrict__ indptr,
    const Idx* __restrict__ indices, int64_t num_cols, int64_t feat_dim,
    int64_t num_codebooks, int64_t length, int64_t column_slice) {
  // SPMM with CSC.
  int ty = blockIdx.x * blockDim.y + threadIdx.y;
  const Idx stride_y = blockDim.y * gridDim.x;
  const int stride_x = blockDim.x * gridDim.y;

  while (ty < num_cols) {
    int tx = blockIdx.y * blockDim.x + threadIdx.x;
    Idx begin = indptr[ty];
    Idx end = indptr[ty + 1];
    float deg = end - begin == 0 ? 1 : end - begin;

    while (tx < feat_dim) {
      int codebook_index = tx / column_slice;
      int offset = tx % column_slice;
      DType local_accum = 0;
      for (Idx i = begin; i < end; ++i) {
        local_accum += vq_decompress_func(compress_feat, codebooks,
                                          num_codebooks, length, column_slice,
                                          codebook_index, indices[i], offset);
      }
      local_accum = local_accum / deg;
      int out_pos = ty * feat_dim + tx;
      out[out_pos] += local_accum;
      tx += stride_x;
    }
    ty += stride_y;
  }
}

torch::Tensor VqDecompressSpMMFusionCsr(torch::Tensor compress_data,
                                        torch::Tensor codebooks,
                                        torch::Tensor indptr,
                                        torch::Tensor indices,
                                        int64_t feat_dim) {
  int64_t num_cols = indptr.numel() - 1;
  int64_t num_codebooks = codebooks.size(0);
  int64_t length = codebooks.size(1);
  int64_t column_slice = codebooks.size(2);
  // int64_t feat_dim = output.size(1);

  const int ntx = (feat_dim + 31) / 32 * 32;
  const int nty = 1024 / ntx;
  const int nbx = (num_cols + nty - 1) / nty;
  const dim3 nblks(nbx);
  const dim3 nthrs(ntx, nty);

  torch::Tensor out = torch::zeros(
      {num_cols, feat_dim},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

  PG_SPMM_FLOAT_TYPE_SWITCH(codebooks.dtype(), DType, {
    PG_SPMM_INT_TYPE_SWITCH(indptr.dtype(), Idx, {
      PG_INT_TYPE_SWITCH(compress_data.dtype(), CompressDType, {
        VqDecompressSpMMCsrFusionKernel<Idx, DType, CompressDType>
            <<<nblks, nthrs>>>(compress_data.data_ptr<CompressDType>(),
                               codebooks.data_ptr<DType>(),
                               out.data_ptr<DType>(), indptr.data_ptr<Idx>(),
                               indices.data_ptr<Idx>(), num_cols, feat_dim,
                               num_codebooks, length, column_slice);
      });
    });
  });
  return out;
}

template <typename Idx, typename DType, typename CompressDType, int TARGET_BITS,
          int WARP_SIZE = 16>
__global__ void SqDecompressSpMMCsrFusionKernel2(
    CompressDType* __restrict__ compress_feat, DType* __restrict__ codebooks,
    float* __restrict__ out, const Idx* __restrict__ indptr,
    const Idx* __restrict__ indices, int64_t num_cols, int64_t src_dim,
    int64_t dst_dim, int64_t column_slice) {
  // one dim kenrel
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = gridDim.x * blockDim.x;
  int warp_id = thread_id / WARP_SIZE;
  int lane_id = thread_id % WARP_SIZE;

  int num_warps = num_threads / WARP_SIZE;

  constexpr float drange = 1 << (TARGET_BITS - 1);
  constexpr int codebook_dim = 5;
  constexpr int pack_size = 8 / TARGET_BITS > 1 ? 8 / TARGET_BITS : 1;

  extern __shared__ float tmp_result[];

  uint8_t mask;
  if (TARGET_BITS == 1) {
    mask = 0x01;
  } else if (TARGET_BITS == 2) {
    mask = 0x03;
  } else if (TARGET_BITS == 4) {
    mask = 0x0f;
  }

  for (int i = warp_id; i < num_cols; i += num_warps) {
    float* output_ptr = out + i * dst_dim;
    Idx begin = indptr[i];
    Idx end = indptr[i + 1];
    float deg = end - begin == 0 ? 1 : end - begin;

    for (int j = lane_id; j < src_dim; j += WARP_SIZE) {
      int codebook_index = j * pack_size / column_slice;
      float emin, emax, mean;

      // fmin, fmax, emin, emax, mean in codebook
      if (TARGET_BITS == 1) {
        mean = codebooks[codebook_index * codebook_dim + 4];
      } else {
        emin = codebooks[codebook_index * codebook_dim + 2];
        emax = codebooks[codebook_index * codebook_dim + 3];
      }

      // reset tmp_result
      for (int k = 0; k < pack_size; ++k) {
        tmp_result[threadIdx.x * pack_size + k] = 0.0f;
      }

      for (int k = begin; k < end; ++k) {
        CompressDType exp = compress_feat[indices[k] * src_dim + j];

        // decompress
        if (TARGET_BITS >= 8) {
          float value = (float)exp;
          value = value + 0.5;
          int sign = value >= 0 ? 1 : -1;
          value = fabs(value) * ((emax - emin) / drange) + emin;
          value = exp2f(value) * sign;
          tmp_result[threadIdx.x * pack_size] += value;

        } else {
          for (int offset = 0; offset < pack_size; ++offset) {
            float value = (float)(exp & mask);
            exp = exp >> TARGET_BITS;

            if (TARGET_BITS == 1) {
              value -= 0.5;
              value *= 2;
              value *= mean;
            } else {
              value -= drange;
              value = value + 0.5;
              int sign = value >= 0 ? 1 : -1;
              value = fabs(value) * ((emax - emin) / drange) + emin;
              value = exp2f(value) * sign;
            }

            tmp_result[threadIdx.x * pack_size + offset] += value;
          }
        }
      }

      // write back
      for (int k = 0; k < pack_size; ++k) {
        int index = j * pack_size + k;
        if (index < dst_dim)
          output_ptr[index] = tmp_result[threadIdx.x * pack_size + k] / deg;
      }
    }
  }
}

torch::Tensor SqDecompressSpMMFusionCsr(
    torch::Tensor compress_data, torch::Tensor codebooks, torch::Tensor indptr,
    torch::Tensor indices, int64_t target_bits, int64_t column_slice,
    int64_t feat_dim) {
  int64_t num_cols = indptr.numel() - 1;
  int64_t src_dim = compress_data.size(1);
  torch::Tensor out = torch::zeros(
      {num_cols, feat_dim},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

  PG_SPMM_FLOAT_TYPE_SWITCH(codebooks.dtype(), DType, {
    PG_SPMM_INT_TYPE_SWITCH(indptr.dtype(), Idx, {
      PG_INT_TYPE_SWITCH(compress_data.dtype(), CompressDType, {
        PG_TARGET_BITS_SWITCH(target_bits, TARGET_BITS, {
          constexpr int num_pack = 8 / TARGET_BITS > 1 ? 8 / TARGET_BITS : 1;
          constexpr int shared_memory_size = sizeof(float) * 256 * num_pack;
          constexpr int WARP_SIZE = 128 / num_pack;
          const int nbxs = (num_cols + WARP_SIZE - 1) / WARP_SIZE;
          const int nthrs = 256;
          SqDecompressSpMMCsrFusionKernel2<Idx, DType, CompressDType,
                                           TARGET_BITS, WARP_SIZE>
              <<<nbxs, nthrs, shared_memory_size>>>(
                  compress_data.data_ptr<CompressDType>(),
                  codebooks.data_ptr<DType>(), out.data_ptr<float>(),
                  indptr.data_ptr<Idx>(), indices.data_ptr<Idx>(), num_cols,
                  src_dim, feat_dim, column_slice);
        });
      });
    });
  });

  return out;
}