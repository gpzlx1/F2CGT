
#include "common.h"
#include "cuda_operators.h"
#include "cuda_utils.cuh"

template <int target_bits>
__global__ void packbits_kernel(uint8_t* unpack, uint8_t* pack,
                                int64_t num_items, int64_t unpack_dim,
                                int64_t pack_dim) {
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = gridDim.x * blockDim.x;

  int numel = 8 / target_bits;

  uint8_t mask;
  if (target_bits == 1) {
    mask = 0x01;
  } else if (target_bits == 2) {
    mask = 0x03;
  } else if (target_bits == 4) {
    mask = 0x0f;
  }

  for (int dst_index = thread_id; dst_index < num_items;
       dst_index += num_threads) {
    uint8_t dst_value = 0;
    int row_id = dst_index / pack_dim;
    int row_offset = dst_index % pack_dim;
    int src_begin = row_id * unpack_dim + row_offset * numel;
    int src_end = src_begin + numel < (row_id + 1) * unpack_dim
                      ? src_begin + numel
                      : (row_id + 1) * unpack_dim;

    for (int i = 0; i < src_end - src_begin; i += 1) {
      int src_index = src_begin + i;
      uint8_t value = unpack[src_index];
      // small end
      dst_value |= (value & mask) << (target_bits * i);
    }
    pack[dst_index] = dst_value;
  }
}

torch::Tensor packbits(torch::Tensor unpack, int64_t target_bits) {
  int64_t unpack_dim = unpack.size(1);
  int64_t pack_ratio = 8 / target_bits;
  int64_t pack_dim = (unpack_dim + pack_ratio - 1) / pack_ratio;

  torch::Tensor pack = torch::empty(
      {unpack.size(0), pack_dim},
      torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));

  int64_t num_items = pack.numel();
  int64_t block_size = 1024;
  int64_t num_blocks = (num_items + block_size - 1) / block_size;

  PG_TARGET_BITS_SWITCH(target_bits, TARGET_BITS, {
    packbits_kernel<TARGET_BITS><<<num_blocks, block_size>>>(
        unpack.data_ptr<uint8_t>(), pack.data_ptr<uint8_t>(), num_items,
        unpack_dim, pack_dim);
  });

  return pack;
}

template <int target_bits>
__global__ void unpackbits_kernel(uint8_t* pack, uint8_t* unpack,
                                  int64_t num_items, int64_t unpack_dim,
                                  int64_t pack_dim) {
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = gridDim.x * blockDim.x;

  uint8_t mask;
  if (target_bits == 1) {
    mask = 0x01;
  } else if (target_bits == 2) {
    mask = 0x03;
  } else if (target_bits == 4) {
    mask = 0x0f;
  }

  int numel = 8 / target_bits;

  for (int i = thread_id; i < num_items; i += num_threads) {
    int dst_index = i;
    int row_id = dst_index / unpack_dim;
    int offset = dst_index % unpack_dim;
    int src_index = row_id * pack_dim + offset / numel;
    int bit_offset = offset % numel;
    // unpack[dst_index] = (pack[src_index] >> (target_bits * bit_offset)) &
    // mask;
    unpack[dst_index] =
        unpackbits_func(pack + src_index, bit_offset, target_bits, mask);
  }
}

torch::Tensor unpackbits(torch::Tensor pack, int64_t target_bits,
                         int64_t unpack_dim) {
  torch::Tensor unpack = torch::empty(
      {pack.size(0), unpack_dim},
      torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));

  int64_t pack_dim = pack.size(1);
  int64_t num_items = unpack.numel();
  int64_t block_size = 1024;
  int64_t num_blocks = (num_items + block_size - 1) / block_size;

  PG_TARGET_BITS_SWITCH(target_bits, TARGET_BITS, {
    unpackbits_kernel<TARGET_BITS><<<num_blocks, block_size>>>(
        pack.data_ptr<uint8_t>(), unpack.data_ptr<uint8_t>(), num_items,
        unpack_dim, pack_dim);
  });
  return unpack;
}