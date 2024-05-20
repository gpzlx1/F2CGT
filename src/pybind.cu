#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "cuda_operators.h"
#include "kmeans.h"

PYBIND11_MODULE(F2CGTLib, m) {
  m.def("packbits", &packbits, py::arg("unpack"), py::arg("target_bits"))
      .def("unpackbits", &unpackbits, py::arg("pack"), py::arg("target_bits"),
           py::arg("unpack_dim"))
      .def("sq_compress", &sq_compress, py::arg("src"), py::arg("codebooks"),
           py::arg("target_bits"), py::arg("column_slice"))
      .def("sq_decompress", &sq_decompress, py::arg("compress_data"),
           py::arg("codebooks"), py::arg("target_bits"),
           py::arg("column_slice"), py::arg("feat_dim"))
      .def("vq_decompress", &vq_decompress, py::arg("index_tensor"),
           py::arg("codebooks"), py::arg("feat_dim"))
      .def("pairwise_distance", &pairwise_distance, py::arg("x"), py::arg("y"),
           py::arg("metric"))
      .def("spmm_csr", &SpMMCsr, py::arg("feat"), py::arg("indptr"),
           py::arg("indices"))
      .def("vq_decompress_spmm_csr", &VqDecompressSpMMFusionCsr,
           py::arg("compress_feat"), py::arg("codebooks"), py::arg("indptr"),
           py::arg("indices"), py::arg("feat_dim"))
      .def("sq_decompress_spmm_csr", &SqDecompressSpMMFusionCsr,
           py::arg("compress_feat"), py::arg("codebooks"), py::arg("indptr"),
           py::arg("indices"), py::arg("target_bit"), py::arg("column_slice"),
           py::arg("feat_dim"));
  py::class_<KMeans>(m, "KMeans")
      .def(py::init<int64_t, int64_t, std::string, int, double, int, double,
                    int, int, bool>(),
           py::arg("n_cluster"), py::arg("feat_dim"), py::arg("metric"),
           py::arg("max_iter") = 300, py::arg("tol") = 1e-4,
           py::arg("n_init") = 1, py::arg("oversampling_factor") = 2.0,
           py::arg("batch_samples") = 1 << 15, py::arg("batch_centroids") = 0,
           py::arg("inertia_check") = false)
      .def(py::init<torch::Tensor, std::string, int, double, int, double, int,
                    int, bool>(),
           py::arg("centers"), py::arg("metric"), py::arg("max_iter") = 300,
           py::arg("tol") = 1e-4, py::arg("n_init") = 1,
           py::arg("oversampling_factor") = 2.0,
           py::arg("batch_samples") = 1 << 15, py::arg("batch_centroids") = 0,
           py::arg("inertia_check") = false)
      .def("fit", &KMeans::fit, py::arg("data"))
      .def("predict", &KMeans::predict, py::arg("data"),
           py::arg("normalize_weights") = true)
      .def("get_centers", &KMeans::get_centers);
}