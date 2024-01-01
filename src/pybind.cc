#include <pybind11/pybind11.h>
#include <torch/script.h>
#include "cpu/omp.h"
#include "cuda/cuda_ops.h"
#include "hashmap/hashmap_v2.h"
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
      .def("_CAPI_search_hashmap", &SearchHashMapCUDA, py::arg("hash_key"),
           py::arg("hash_val"), py::arg("input_key"))
      .def("_CAPI_pin_tensor", &TensorPinMemory, py::arg("data"))
      .def("_CAPI_unpin_tensor", &TensorUnpinMemory, py::arg("data"))
      .def("_CAPI_fetch_feature_data", &FeatureFetchDataCUDA, py::arg("data"),
           py::arg("nid"))
      .def("_CAPI_fetch_feature_data_with_caching",
           &FeatureFetchDataWithCachingCUDA, py::arg("cpu_data"),
           py::arg("gpu_data"), py::arg("hashed_key_tensor"),
           py::arg("hashed_value_tensor"), py::arg("nid"))
      .def("_CAPI_fetch_feature_data_with_caching_v2",
           &FeatureFetchDataWithCachingCUDA_V2, py::arg("cpu_data"),
           py::arg("gpu_data"), py::arg("nid"), py::arg("local_nid"))
      .def("_CAPI_cuda_index_fetch", &CUDAIndexFetch, py::arg("src"),
           py::arg("src_index"), py::arg("dst"), py::arg("dst_index"))
      .def("_CAPI_cuda_sample_neighbors", &RowWiseSamplingUniformCUDA,
           py::arg("seeds"), py::arg("indptr"), py::arg("indices"),
           py::arg("num_picks"), py::arg("replace"))
      .def("_CAPI_cuda_sample_neighbors_with_caching",
           &RowWiseSamplingUniformWithCachingCUDA, py::arg("seeds"),
           py::arg("gpu_indptr"), py::arg("cpu_indptr"), py::arg("gpu_indices"),
           py::arg("cpu_indices"), py::arg("cache_index"), py::arg("num_picks"),
           py::arg("replace"))
      .def("_CAPI_cuda_tensor_relabel", &TensorRelabelCUDA,
           py::arg("mapping_tensors"), py::arg("requiring_relabel_tensors"))
      .def("_CAPI_get_sub_indptr", &GetSubIndptr, py::arg("nids"),
           py::arg("indptr"))
      .def("_CAPI_get_sub_edge_data", &GetSubEdgeData, py::arg("nids"),
           py::arg("indptr"), py::arg("sub_indptr"), py::arg("edge_data"))
      .def("_CAPI_count_cached_nids", &CountCachedNidsNum,
           py::arg("input_nids"), py::arg("hashed_orig_nids"),
           py::arg("hashed_device_nids"));

  m.def("_CAPI_vq_decompress", &vq_decompress, py::arg("codebook_indices"),
        py::arg("compressed_features"), py::arg("codebooks"),
        py::arg("feat_dim"))
      .def("_CAPI_vq_decompress_v2", &vq_decompress_v2, py::arg("index"),
           py::arg("chunk_size"), py::arg("compressed_features"),
           py::arg("codebooks"), py::arg("output"), py::arg("output_offset"))
      .def("_CAPI_sq_decompress", &sq_decompress, py::arg("codebook_indices"),
           py::arg("compressed_features"), py::arg("codebooks"),
           py::arg("feat_dim"))
      .def("_CAPI_sq_decompress_v2", &sq_decompress_v2, py::arg("index"),
           py::arg("chunk_size"), py::arg("compressed_features"),
           py::arg("codebooks"), py::arg("output"), py::arg("output_offset"));
  // .def("_CAPI_meanaggr", & meanaggr);

  py::class_<hashmap::BiFeatHashmaps>(m, "BiFeatHashmaps")
      .def(py::init<int64_t, std::vector<torch::Tensor>>())
      .def("query", &hashmap::BiFeatHashmaps::query)
      .def("get_memory_usage", &hashmap::BiFeatHashmaps::get_memory_usage);
}