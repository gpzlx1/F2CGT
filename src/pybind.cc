#include <pybind11/pybind11.h>
#include <torch/script.h>
#include "cpu/omp.h"
#include "cuda/cuda_ops.h"
#include "pin_memory.h"
#include "shm/shared_memory.h"

namespace py = pybind11;
using namespace bifeat;

PYBIND11_MODULE(BiFeatLib, m) {
  m.def("create_shared_mem", &shm::create_shared_mem, py::arg("name"),
        py::arg("size"), py::arg("pin_memory") = true)
      .def("open_shared_mem", &shm::open_shared_mem, py::arg("name"),
           py::arg("size"), py::arg("pin_memory") = true)
      .def("release_shared_mem", &shm::release_shared_mem, py::arg("name"),
           py::arg("size"), py::arg("ptr"), py::arg("fd"),
           py::arg("pin_memory") = true)
      .def("open_shared_tensor", &shm::open_shared_tensor, py::arg("ptr"),
           py::arg("dtype"), py::arg("shape"))
      .def("omp_reorder_indices", &cpu::omp_reorder_indices, py::arg("indptr"),
           py::arg("indices"), py::arg("new_indptr"), py::arg("new_indices"),
           py::arg("dst2src"));

  m.def("_CAPI_create_hashmap", &CreateHashMapTensorCUDA, py::arg("cache_nids"))
      .def("_CAPI_pin_tensor", &TensorPinMemory, py::arg("data"))
      .def("_CAPI_unpin_tensor", &TensorUnpinMemory, py::arg("data"))
      .def("_CAPI_fetch_feature_data", &FeatureFetchDataCUDA, py::arg("data"),
           py::arg("nid"))
      .def("_CAPI_fetch_feature_data_with_caching",
           &FeatureFetchDataWithCachingCUDA, py::arg("cpu_data"),
           py::arg("gpu_data"), py::arg("hashed_key_tensor"),
           py::arg("hashed_value_tensor"), py::arg("nid"));

  m.def("_CAPI_vq_decompress", &vq_decompress, py::arg("codebook_indices"),
        py::arg("compressed_features"), py::arg("codebooks"),
        py::arg("feat_dim"))
      .def("_CAPI_sq_decompress", &sq_decompress, py::arg("codebook_indices"),
           py::arg("compressed_features"), py::arg("codebooks"),
           py::arg("feat_dim"));
  // .def("_CAPI_meanaggr", & meanaggr);
}