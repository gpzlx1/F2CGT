#include <torch/script.h>
#include "cuda/cuda_ops.h"
#include "packbits.h"
#include "pin_memory.h"

using namespace pg;

TORCH_LIBRARY(pg_ops, m) {
  m.def("_CAPI_pin_tensor", &TensorPinMemory)
      .def("_CAPI_unpin_tensor", &TensorUnpinMemory)
      .def("_CAPI_fetch_feature_data", &FeatureFetchDataCUDA)
      .def("_CAPI_fetch_feature_data_with_caching",
           &FeatureFetchDataWithCachingCUDA)
      .def("_CAPI_meanaggr", &meanaggr)
      .def("_CAPI_packbits", &packbits);
  //  .def("_CAPI_unpackbits", &unpackbits);
}