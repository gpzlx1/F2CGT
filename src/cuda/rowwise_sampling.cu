#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <torch/script.h>

#include "../common.h"
#include "atomic.h"
#include "cub_function.h"
#include "cuda_ops.h"
#include "hashmap.h"

#define BLOCK_SIZE 128
namespace bifeat {

template <typename IdType>
inline torch::Tensor _GetSubIndptr(torch::Tensor seeds, torch::Tensor indptr,
                                   int64_t num_pick, bool replace) {
  int64_t num_items = seeds.numel();
  torch::Tensor sub_indptr = torch::empty(
      (num_items + 1),
      torch::TensorOptions().dtype(indptr.dtype()).device(torch::kCUDA));
  thrust::device_ptr<IdType> item_prefix(
      static_cast<IdType *>(sub_indptr.data_ptr<IdType>()));

  using it = thrust::counting_iterator<IdType>;
  thrust::for_each(
      it(0), it(num_items),
      [in = seeds.data_ptr<IdType>(), in_indptr = indptr.data_ptr<IdType>(),
       out = thrust::raw_pointer_cast(item_prefix), replace,
       num_pick] __device__(int i) mutable {
        IdType row = in[i];
        IdType begin = in_indptr[row];
        IdType end = in_indptr[row + 1];
        if (replace) {
          out[i] = (end - begin) == 0 ? 0 : num_pick;
        } else {
          out[i] = MIN(end - begin, num_pick);
        }
      });

  cub_exclusiveSum<IdType>(thrust::raw_pointer_cast(item_prefix),
                           num_items + 1);
  return sub_indptr;
}

template <typename IdType>
inline torch::Tensor _GetSubIndptrWithCaching(torch::Tensor seeds,
                                              torch::Tensor cpu_indptr,
                                              torch::Tensor gpu_indptr,
                                              torch::Tensor orig_nids_hashed,
                                              torch::Tensor gpu_nids_hashed,
                                              int64_t num_pick, bool replace) {
  int64_t num_items = seeds.numel();
  torch::Tensor sub_indptr =
      torch::empty((num_items + 1), gpu_indptr.options());
  thrust::device_ptr<IdType> item_prefix(
      static_cast<IdType *>(sub_indptr.data_ptr<IdType>()));
  int64_t dir_size = orig_nids_hashed.numel();

  using it = thrust::counting_iterator<IdType>;
  thrust::for_each(it(0), it(num_items),
                   [in_hash_size = dir_size, in = seeds.data_ptr<IdType>(),
                    in_cpu_indptr = cpu_indptr.data_ptr<IdType>(),
                    in_gpu_indptr = gpu_indptr.data_ptr<IdType>(),
                    in_hash_key = orig_nids_hashed.data_ptr<IdType>(),
                    in_hash_value = gpu_nids_hashed.data_ptr<int32_t>(),
                    out = thrust::raw_pointer_cast(item_prefix), replace,
                    num_pick] __device__(int i) mutable {
                     Hashmap<IdType> table(in_hash_key, in_hash_value,
                                           in_hash_size);
                     const int64_t pos = table.SearchForPos(in[i]);
                     IdType begin = 0;
                     IdType end = 0;
                     if (pos != -1) {
                       begin = in_gpu_indptr[in_hash_value[pos]];
                       end = in_gpu_indptr[in_hash_value[pos] + 1];
                     } else {
                       begin = in_cpu_indptr[in[i]];
                       end = in_cpu_indptr[in[i] + 1];
                     }
                     if (replace) {
                       out[i] = (end - begin) == 0 ? 0 : num_pick;
                     } else {
                       out[i] = MIN(end - begin, num_pick);
                     }
                   });

  cub_exclusiveSum<IdType>(thrust::raw_pointer_cast(item_prefix),
                           num_items + 1);
  return sub_indptr;
}

template <typename IdType, int TILE_SIZE>
__global__ void _CSRRowWiseSampleUniformKernel(
    const uint64_t rand_seed, const int64_t num_picks, const int64_t num_rows,
    const IdType *__restrict__ const in_rows,
    const IdType *__restrict__ const in_ptr,
    const IdType *__restrict__ const in_index,
    const IdType *__restrict__ const out_ptr,
    IdType *__restrict__ const out_rows, IdType *__restrict__ const out_cols) {
  // we assign one warp per row
  assert(blockDim.x == BLOCK_SIZE);

  int64_t out_row = blockIdx.x * TILE_SIZE;
  const int64_t last_row =
      min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_rows);

  curandStatePhilox4_32_10_t rng;
  curand_init(rand_seed * gridDim.x + blockIdx.x, threadIdx.x, 0, &rng);

  while (out_row < last_row) {
    const int64_t row = in_rows[out_row];
    const int64_t in_row_start = in_ptr[row];
    const int64_t deg = in_ptr[row + 1] - in_row_start;
    const int64_t out_row_start = out_ptr[out_row];

    if (deg <= num_picks) {
      // just copy row when there is not enough nodes to sample.
      for (int idx = threadIdx.x; idx < deg; idx += BLOCK_SIZE) {
        const IdType in_idx = in_row_start + idx;
        out_rows[out_row_start + idx] = row;
        out_cols[out_row_start + idx] = in_index[in_idx];
      }
    } else {
      // generate permutation list via reservoir algorithm
      for (int idx = threadIdx.x; idx < num_picks; idx += BLOCK_SIZE) {
        out_cols[out_row_start + idx] = idx;
      }
      __syncthreads();

      for (int idx = num_picks + threadIdx.x; idx < deg; idx += BLOCK_SIZE) {
        const int num = curand(&rng) % (idx + 1);
        if (num < num_picks) {
          // use max so as to achieve the replacement order the serial
          // algorithm would have
          atomic::AtomicMax(out_cols + out_row_start + num, IdType(idx));
        }
      }
      __syncthreads();

      // copy permutation over
      for (int idx = threadIdx.x; idx < num_picks; idx += BLOCK_SIZE) {
        const IdType perm_idx = out_cols[out_row_start + idx] + in_row_start;
        out_rows[out_row_start + idx] = row;
        out_cols[out_row_start + idx] = in_index[perm_idx];
      }
    }
    out_row += 1;
  }
}

template <typename IdType, int TILE_SIZE>
__global__ void _CSRRowWiseSampleUniformReplaceKernel(
    const uint64_t rand_seed, const int64_t num_picks, const int64_t num_rows,
    const IdType *__restrict__ const in_rows,
    const IdType *__restrict__ const in_ptr,
    const IdType *__restrict__ const in_index,
    const IdType *__restrict__ const out_ptr,
    IdType *__restrict__ const out_rows, IdType *__restrict__ const out_cols) {
  // we assign one warp per row
  assert(blockDim.x == BLOCK_SIZE);

  int64_t out_row = blockIdx.x * TILE_SIZE;
  const int64_t last_row =
      min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_rows);

  curandStatePhilox4_32_10_t rng;
  curand_init(rand_seed * gridDim.x + blockIdx.x, threadIdx.x, 0, &rng);

  while (out_row < last_row) {
    const int64_t row = in_rows[out_row];
    const int64_t in_row_start = in_ptr[row];
    const int64_t out_row_start = out_ptr[out_row];
    const int64_t deg = in_ptr[row + 1] - in_row_start;

    if (deg > 0) {
      // each thread then blindly copies in rows only if deg > 0.
      for (int idx = threadIdx.x; idx < num_picks; idx += BLOCK_SIZE) {
        const int64_t edge = curand(&rng) % deg;
        const int64_t out_idx = out_row_start + idx;
        out_rows[out_idx] = row;
        out_cols[out_idx] = in_index[in_row_start + edge];
      }
    }
    out_row += 1;
  }
}

template <typename IdType, int TILE_SIZE>
__global__ void _CSRRowWiseSampleUniformWithCachingKernel(
    const uint64_t rand_seed, const int64_t num_picks, const int64_t num_rows,
    const int64_t hash_dir_size, const IdType *__restrict__ const in_rows,
    const IdType *__restrict__ const in_gpu_ptr,
    const IdType *__restrict__ const in_cpu_ptr,
    const IdType *__restrict__ const in_gpu_index,
    const IdType *__restrict__ const in_cpu_index,
    const IdType *__restrict__ const out_ptr,
    IdType *__restrict__ const orig_nids_hashed,
    int32_t *__restrict__ const gpu_nids_hashed,
    IdType *__restrict__ const out_rows, IdType *__restrict__ const out_cols) {
  // we assign one warp per row
  assert(blockDim.x == BLOCK_SIZE);

  int64_t out_row = blockIdx.x * TILE_SIZE;
  const int64_t last_row =
      min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_rows);

  curandStatePhilox4_32_10_t rng;
  curand_init(rand_seed * gridDim.x + blockIdx.x, threadIdx.x, 0, &rng);

  Hashmap<IdType> table(orig_nids_hashed, gpu_nids_hashed, hash_dir_size);

  while (out_row < last_row) {
    const int64_t row = in_rows[out_row];
    const int64_t pos = table.SearchForPos(row);
    int64_t in_row_start = 0;
    int64_t deg = 0;
    if (pos != -1) {
      in_row_start = in_gpu_ptr[gpu_nids_hashed[pos]];
      deg = in_gpu_ptr[gpu_nids_hashed[pos] + 1] - in_row_start;
    } else {
      in_row_start = in_cpu_ptr[row];
      deg = in_cpu_ptr[row + 1] - in_row_start;
    }
    const int64_t out_row_start = out_ptr[out_row];

    if (deg <= num_picks) {
      // just copy row when there is not enough nodes to sample.
      for (int idx = threadIdx.x; idx < deg; idx += BLOCK_SIZE) {
        const IdType in_idx = in_row_start + idx;
        out_rows[out_row_start + idx] = row;
        if (pos != -1) {
          out_cols[out_row_start + idx] = in_gpu_index[in_idx];
        } else {
          out_cols[out_row_start + idx] = in_cpu_index[in_idx];
        }
      }
    } else {
      // generate permutation list via reservoir algorithm
      for (int idx = threadIdx.x; idx < num_picks; idx += BLOCK_SIZE) {
        out_cols[out_row_start + idx] = idx;
      }
      __syncthreads();

      for (int idx = num_picks + threadIdx.x; idx < deg; idx += BLOCK_SIZE) {
        const int num = curand(&rng) % (idx + 1);
        if (num < num_picks) {
          // use max so as to achieve the replacement order the serial
          // algorithm would have
          atomic::AtomicMax(out_cols + out_row_start + num, IdType(idx));
        }
      }
      __syncthreads();

      // copy permutation over
      for (int idx = threadIdx.x; idx < num_picks; idx += BLOCK_SIZE) {
        const IdType perm_idx = out_cols[out_row_start + idx] + in_row_start;
        out_rows[out_row_start + idx] = row;
        if (pos != -1) {
          out_cols[out_row_start + idx] = in_gpu_index[perm_idx];
        } else {
          out_cols[out_row_start + idx] = in_cpu_index[perm_idx];
        }
      }
    }
    out_row += 1;
  }
}

template <typename IdType, int TILE_SIZE>
__global__ void _CSRRowWiseSampleUniformReplaceWithCachingKernel(
    const uint64_t rand_seed, const int64_t num_picks, const int64_t num_rows,
    const int64_t hash_dir_size, const IdType *__restrict__ const in_rows,
    const IdType *__restrict__ const in_gpu_ptr,
    const IdType *__restrict__ const in_cpu_ptr,
    const IdType *__restrict__ const in_gpu_index,
    const IdType *__restrict__ const in_cpu_index,
    const IdType *__restrict__ const out_ptr,
    IdType *__restrict__ const orig_nids_hashed,
    int32_t *__restrict__ const gpu_nids_hashed,
    IdType *__restrict__ const out_rows, IdType *__restrict__ const out_cols) {
  // we assign one warp per row
  assert(blockDim.x == BLOCK_SIZE);

  int64_t out_row = blockIdx.x * TILE_SIZE;
  const int64_t last_row =
      min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_rows);

  curandStatePhilox4_32_10_t rng;
  curand_init(rand_seed * gridDim.x + blockIdx.x, threadIdx.x, 0, &rng);

  Hashmap<IdType> table(orig_nids_hashed, gpu_nids_hashed, hash_dir_size);

  while (out_row < last_row) {
    const int64_t row = in_rows[out_row];
    const int64_t pos = table.SearchForPos(row);
    int64_t in_row_start = 0;
    int64_t deg = 0;
    if (pos != -1) {
      in_row_start = in_gpu_ptr[gpu_nids_hashed[pos]];
      deg = in_gpu_ptr[gpu_nids_hashed[pos] + 1] - in_row_start;
    } else {
      in_row_start = in_cpu_ptr[row];
      deg = in_cpu_ptr[row + 1] - in_row_start;
    }
    const int64_t out_row_start = out_ptr[out_row];

    if (deg > 0) {
      // each thread then blindly copies in rows only if deg > 0.
      for (int idx = threadIdx.x; idx < num_picks; idx += BLOCK_SIZE) {
        const int64_t edge = curand(&rng) % deg;
        const int64_t out_idx = out_row_start + idx;
        out_rows[out_idx] = row;
        if (pos != -1) {
          out_cols[out_idx] = in_gpu_index[in_row_start + edge];
        } else {
          out_cols[out_idx] = in_cpu_index[in_row_start + edge];
        }
      }
    }
    out_row += 1;
  }
}

std::tuple<torch::Tensor, torch::Tensor> RowWiseSamplingUniformCUDA(
    torch::Tensor seeds, torch::Tensor indptr, torch::Tensor indices,
    int64_t num_picks, bool replace) {
  CHECK_CUDA(seeds);
  PG_ID_TYPE_SWITCH(indptr.dtype(), IdType, {
    int num_rows = seeds.numel();
    torch::Tensor sub_indptr =
        _GetSubIndptr<IdType>(seeds, indptr, num_picks, replace);
    thrust::device_ptr<IdType> item_prefix(
        static_cast<IdType *>(sub_indptr.data_ptr<IdType>()));
    int nnz = item_prefix[num_rows];

    torch::Tensor coo_row = torch::empty(
        nnz, torch::TensorOptions().dtype(seeds.dtype()).device(torch::kCUDA));
    torch::Tensor coo_col = torch::empty(
        nnz,
        torch::TensorOptions().dtype(indices.dtype()).device(torch::kCUDA));

    const uint64_t random_seed = 7777;

    constexpr int TILE_SIZE = 128 / BLOCK_SIZE;
    if (replace) {
      const dim3 block(BLOCK_SIZE);
      const dim3 grid((num_rows + TILE_SIZE - 1) / TILE_SIZE);
      _CSRRowWiseSampleUniformReplaceKernel<IdType, TILE_SIZE><<<grid, block>>>(
          random_seed, num_picks, num_rows, seeds.data_ptr<IdType>(),
          indptr.data_ptr<IdType>(), indices.data_ptr<IdType>(),
          sub_indptr.data_ptr<IdType>(), coo_row.data_ptr<IdType>(),
          coo_col.data_ptr<IdType>());
    } else {
      const dim3 block(BLOCK_SIZE);
      const dim3 grid((num_rows + TILE_SIZE - 1) / TILE_SIZE);
      _CSRRowWiseSampleUniformKernel<IdType, TILE_SIZE><<<grid, block>>>(
          random_seed, num_picks, num_rows, seeds.data_ptr<IdType>(),
          indptr.data_ptr<IdType>(), indices.data_ptr<IdType>(),
          sub_indptr.data_ptr<IdType>(), coo_row.data_ptr<IdType>(),
          coo_col.data_ptr<IdType>());
    }

    return std::make_tuple(coo_row, coo_col);
  });
  return std::make_tuple(torch::Tensor(), torch::Tensor());
}

std::tuple<torch::Tensor, torch::Tensor> RowWiseSamplingUniformWithCachingCUDA(
    torch::Tensor seeds, torch::Tensor gpu_indptr, torch::Tensor cpu_indptr,
    torch::Tensor gpu_indices, torch::Tensor cpu_indices,
    torch::Tensor orig_nids_hashed, torch::Tensor gpu_nids_hashed,
    int64_t num_picks, bool replace) {
  CHECK_CUDA(seeds);
  CHECK_CUDA(gpu_indptr);
  CHECK_CUDA(gpu_indices);
  CHECK_CUDA(orig_nids_hashed);
  CHECK_CUDA(gpu_nids_hashed);
  PG_ID_TYPE_SWITCH(gpu_indptr.dtype(), IdType, {
    int num_rows = seeds.numel();
    torch::Tensor sub_indptr = _GetSubIndptrWithCaching<IdType>(
        seeds, cpu_indptr, gpu_indptr, orig_nids_hashed, gpu_nids_hashed,
        num_picks, replace);
    thrust::device_ptr<IdType> item_prefix(
        static_cast<IdType *>(sub_indptr.data_ptr<IdType>()));
    int nnz = item_prefix[num_rows];

    torch::Tensor coo_row = torch::empty(
        nnz, torch::TensorOptions().dtype(seeds.dtype()).device(torch::kCUDA));
    torch::Tensor coo_col = torch::empty(nnz, gpu_indices.options());

    const uint64_t random_seed = 7777;
    constexpr int TILE_SIZE = 128 / BLOCK_SIZE;
    int dir_size = orig_nids_hashed.numel();
    if (replace) {
      const dim3 block(BLOCK_SIZE);
      const dim3 grid((num_rows + TILE_SIZE - 1) / TILE_SIZE);
      _CSRRowWiseSampleUniformReplaceWithCachingKernel<IdType, TILE_SIZE>
          <<<grid, block>>>(
              random_seed, num_picks, num_rows, dir_size,
              seeds.data_ptr<IdType>(), gpu_indptr.data_ptr<IdType>(),
              cpu_indptr.data_ptr<IdType>(), gpu_indices.data_ptr<IdType>(),
              cpu_indices.data_ptr<IdType>(), sub_indptr.data_ptr<IdType>(),
              orig_nids_hashed.data_ptr<IdType>(),
              gpu_nids_hashed.data_ptr<int32_t>(), coo_row.data_ptr<IdType>(),
              coo_col.data_ptr<IdType>());
    } else {
      const dim3 block(BLOCK_SIZE);
      const dim3 grid((num_rows + TILE_SIZE - 1) / TILE_SIZE);
      _CSRRowWiseSampleUniformWithCachingKernel<IdType, TILE_SIZE>
          <<<grid, block>>>(
              random_seed, num_picks, num_rows, dir_size,
              seeds.data_ptr<IdType>(), gpu_indptr.data_ptr<IdType>(),
              cpu_indptr.data_ptr<IdType>(), gpu_indices.data_ptr<IdType>(),
              cpu_indices.data_ptr<IdType>(), sub_indptr.data_ptr<IdType>(),
              orig_nids_hashed.data_ptr<IdType>(),
              gpu_nids_hashed.data_ptr<int32_t>(), coo_row.data_ptr<IdType>(),
              coo_col.data_ptr<IdType>());
    }

    return std::make_tuple(coo_row, coo_col);
  });
  return std::make_tuple(torch::Tensor(), torch::Tensor());
}

}  // namespace bifeat