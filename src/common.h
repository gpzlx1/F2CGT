#ifndef PG_COMMON_H_
#define PG_COMMON_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <torch/script.h>

#define CHECK_CPU(x) \
  TORCH_CHECK(!x.device().is_cuda(), #x " must be a CPU tensor")
#define CHECK_CUDA(x) \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define AlignUp(X, ALIGN_SIZE) (((X) + (ALIGN_SIZE)-1) / (ALIGN_SIZE))
#define MIN(x, y) ((x < y) ? x : y)

#define CUDA_CALL(call)                                                  \
  {                                                                      \
    cudaError_t cudaStatus = call;                                       \
    if (cudaSuccess != cudaStatus) {                                     \
      fprintf(stderr,                                                    \
              "%s:%d ERROR: CUDA RT call \"%s\" failed "                 \
              "with "                                                    \
              "%s (%d).\n",                                              \
              __FILE__, __LINE__, #call, cudaGetErrorString(cudaStatus), \
              cudaStatus);                                               \
      exit(cudaStatus);                                                  \
    }                                                                    \
  }

#define PG_ID_TYPE_SWITCH(val, IdType, ...)         \
  do {                                               \
    if ((val) == torch::kInt32) {                    \
      typedef int32_t IdType;                        \
      { __VA_ARGS__ }                                \
    } else if ((val) == torch::kInt64) {             \
      typedef int64_t IdType;                        \
      { __VA_ARGS__ }                                \
    } else {                                         \
      LOG(FATAL) << "ID can only be int32 or int64"; \
    }                                                \
  } while (0);

#define PG_VALUE_TYPE_SWITCH(val, VType, ...)                     \
  do {                                                             \
    if ((val) == torch::kInt32) {                                  \
      typedef int32_t VType;                                       \
      { __VA_ARGS__ }                                              \
    } else if ((val) == torch::kInt64) {                           \
      typedef int64_t VType;                                       \
      { __VA_ARGS__ }                                              \
    } else if ((val) == torch::kFloat32) {                         \
      typedef float VType;                                         \
      { __VA_ARGS__ }                                              \
    } else {                                                       \
      LOG(FATAL) << "Value can only be int32 or int64 or float32"; \
    }                                                              \
  } while (0);
#endif