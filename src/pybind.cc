#include <torch/script.h>
#include "pg_ops.h"
#include "pin_memory.h"

using namespace pg;

TORCH_LIBRARY(pg_ops, m) {
  m.def("_CAPI_create_hashmap", &CreateHashMapTensorCUDA)
      .def("_CAPI_pin_tensor", &TensorPinMemory)
      .def("_CAPI_unpin_tensor", &TensorUnpinMemory)
      .def("_CAPI_fetch_feature_data", &FeatureFetchDataCUDA)
      .def("_CAPI_fetch_feature_data_with_caching",
           &FeatureFetchDataWithCachingCUDA);
}