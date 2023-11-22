#include <pybind11/pybind11.h>
#include <torch/script.h>
#include "cuda/cuda_ops.h"
// #include "packbits.h"
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
           py::arg("dtype"), py::arg("shape"));

  m.def("_CAPI_pin_tensor", &TensorPinMemory, py::arg("data"))
      .def("_CAPI_unpin_tensor", &TensorUnpinMemory, py::arg("data"))
      .def("_CAPI_fetch_feature_data", &FeatureFetchDataCUDA, py::arg("data"),
           py::arg("nid"))
      .def("_CAPI_fetch_feature_data_with_caching",
           &FeatureFetchDataWithCachingCUDA, py::arg("cpu_data"),
           py::arg("gpu_data"), py::arg("nid"), py::arg("cached_num"));
  // .def("_CAPI_meanaggr", & meanaggr);
}