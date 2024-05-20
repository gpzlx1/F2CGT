#pragma once
#include "common.h"
#include "cuda_operators.h"

__device__ __forceinline__ uint8_t unpackbits_func(uint8_t* src, int bit_offset,
                                                   int target_bits,
                                                   uint8_t mask) {
  return ((*src) >> (target_bits * bit_offset)) & mask;
}

template <typename CompressDType, typename DType, int TARGET_BITS,
          typename IndexType>
__device__ __forceinline__ DType
sq_decompress_func(CompressDType* __restrict__ compress_feat,
                   const DType* __restrict__ codebooks, int64_t src_dim,
                   float drange, int codebook_dim, int pack_size, uint8_t mask,
                   int codebook_index, IndexType index, int feat_offset) {
  CompressDType* src_ptr = compress_feat + index * src_dim;
  DType value = 0;
  if (TARGET_BITS >= 8) {
    value = (DType)src_ptr[feat_offset];
  } else {
    int bit_offset = feat_offset % pack_size;
    int src_index = feat_offset / pack_size;
    value =
        (DType)unpackbits_func(reinterpret_cast<uint8_t*>(src_ptr) + src_index,
                               bit_offset, TARGET_BITS, mask);
  }

  // fmin, fmax, emin, emax, mean in codebook
  if (TARGET_BITS == 1) {
    float mean = codebooks[codebook_index * codebook_dim + 4];
    value = (value - 0.5) * (2 * mean);
  } else {
    float emin = codebooks[codebook_index * codebook_dim + 2];
    float emax = codebooks[codebook_index * codebook_dim + 3];

    if (TARGET_BITS < 8) value -= drange;

    value = value + 0.5;
    int sign = value >= 0 ? 1 : -1;
    value = fabs(value) * ((emax - emin) / drange) + emin;
    value = exp2f(value) * sign;
  }
  return value;
}

template <typename CompressDType, typename DType, typename IndexType>
__device__ __forceinline__ DType vq_decompress_func(
    CompressDType* __restrict__ compress_feat, DType* __restrict__ codebooks,
    int64_t num_codebooks, int64_t length, int64_t column_slice,
    int codebook_index, IndexType index, int offset) {
  CompressDType* input_ptr = compress_feat + index * num_codebooks;
  CompressDType _index = input_ptr[codebook_index];
  DType* codebook_ptr = codebooks + codebook_index * length * column_slice +
                        _index * column_slice;
  return codebook_ptr[offset];
}