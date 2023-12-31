#ifndef PG_PIN_MEMORY_H_
#define PG_PIN_MEMORY_H_
#include "common.h"

namespace bifeat {
void TensorPinMemory(torch::Tensor data) {
  void *mem_ptr = const_cast<void *>(data.storage().data());
  CUDA_CALL(cudaHostRegister(mem_ptr, data.numel() * data.element_size(),
                             cudaHostRegisterDefault));
};

void TensorUnpinMemory(torch::Tensor data) {
  void *mem_ptr = const_cast<void *>(data.storage().data());
  CUDA_CALL(cudaHostUnregister(mem_ptr));
};
}  // namespace bifeat

#endif