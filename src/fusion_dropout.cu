#include <curand_kernel.h>
#include "common.h"
#include "cuda_operators.h"
#include "cuda_utils.cuh"

__global__ void dropout_inplace_kernel(float* output, float p, bool is_training,
                                       int64_t numel, int64_t rand_seed) {
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int total_threads = gridDim.x * blockDim.x;

  if (is_training) {
    curandState rng;
    curand_init(rand_seed + threadIdx.x, thread_id, 0, &rng);

    for (int i = thread_id; i < numel; i += total_threads) {
      output[i] = curand_uniform(&rng) < p ? 0.0f : output[i];
    }
  }
}

torch::Tensor dropout_inplace(torch::Tensor output, float p, bool is_training) {
  int64_t numel = output.numel();
  int block = 256;
  int grid = (numel + block - 1) / block;
  int64_t random_seed = (int64_t)time(0);
  dropout_inplace_kernel<<<grid, block>>>(output.data_ptr<float>(), p,
                                          is_training, numel, random_seed);
  return output;
}