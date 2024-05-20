#pragma once

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

#define PG_SPMM_FLOAT_TYPE_SWITCH(val, IdType, ...)               \
  do {                                                            \
    if ((val) == torch::kFloat32) {                               \
      typedef float IdType;                                       \
      { __VA_ARGS__ }                                             \
    } else if ((val) == torch::kFloat16) {                        \
      typedef at::Half IdType;                                    \
      { __VA_ARGS__ }                                             \
    } else if ((val) == torch::kFloat64) {                        \
      typedef double IdType;                                      \
      { __VA_ARGS__ }                                             \
    } else {                                                      \
      LOG(FATAL) << "ID can only be float64, float32 or float16"; \
    }                                                             \
  } while (0);

#define PG_SPMM_INT_TYPE_SWITCH(val, IntType, ...)                        \
  do {                                                                    \
    if ((val) == torch::kInt16) {                                         \
      typedef int16_t IntType;                                            \
      { __VA_ARGS__ }                                                     \
    } else if ((val) == torch::kUInt8) {                                  \
      typedef uint8_t IntType;                                            \
      { __VA_ARGS__ }                                                     \
    } else if ((val) == torch::kInt8) {                                   \
      typedef int8_t IntType;                                             \
      { __VA_ARGS__ }                                                     \
    } else if ((val) == torch::kInt32) {                                  \
      typedef int32_t IntType;                                            \
      { __VA_ARGS__ }                                                     \
    } else if ((val) == torch::kInt64) {                                  \
      typedef int64_t IntType;                                            \
      { __VA_ARGS__ }                                                     \
    } else {                                                              \
      LOG(FATAL) << "Int can only be int64, int32, int16, int8 or uint8"; \
    }                                                                     \
  } while (0);

#define PG_FLOAT_TYPE_SWITCH(val, IdType, ...)       \
  do {                                               \
    if ((val) == torch::kFloat32) {                  \
      typedef float IdType;                          \
      { __VA_ARGS__ }                                \
    } else if ((val) == torch::kFloat16) {           \
      typedef at::Half IdType;                       \
      { __VA_ARGS__ }                                \
    } else {                                         \
      LOG(FATAL) << "ID can only be int32 or int64"; \
    }                                                \
  } while (0);

#define PG_INT_TYPE_SWITCH(val, IntType, ...)                 \
  do {                                                        \
    if ((val) == torch::kInt16) {                             \
      typedef int16_t IntType;                                \
      { __VA_ARGS__ }                                         \
    } else if ((val) == torch::kUInt8) {                      \
      typedef uint8_t IntType;                                \
      { __VA_ARGS__ }                                         \
    } else if ((val) == torch::kInt8) {                       \
      typedef int8_t IntType;                                 \
      { __VA_ARGS__ }                                         \
    } else {                                                  \
      LOG(FATAL) << "Int can only be int16 or int8 or uint8"; \
    }                                                         \
  } while (0);

#define PG_TARGET_BITS_SWITCH(val, TARGET_BITS, ...)        \
  do {                                                      \
    if ((val) == 1) {                                       \
      const int TARGET_BITS = 1;                            \
      { __VA_ARGS__ }                                       \
    } else if ((val) == 2) {                                \
      const int TARGET_BITS = 2;                            \
      { __VA_ARGS__ }                                       \
    } else if ((val) == 4) {                                \
      const int TARGET_BITS = 4;                            \
      { __VA_ARGS__ }                                       \
    } else if ((val) == 8) {                                \
      const int TARGET_BITS = 8;                            \
      { __VA_ARGS__ }                                       \
    } else if ((val) == 16) {                               \
      const int TARGET_BITS = 16;                           \
      { __VA_ARGS__ }                                       \
    } else {                                                \
      LOG(FATAL) << "TARGET_BITS can only be [1,2,4,8,16]"; \
    }                                                       \
  } while (0);

#define CHECK_CPU(x) \
  TORCH_CHECK(!x.device().is_cuda(), #x " must be a CPU tensor")

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
