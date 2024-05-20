#pragma once
#include <torch/extension.h>
#include <torch/script.h>

torch::Tensor packbits(torch::Tensor src, int64_t target_bits);

torch::Tensor unpackbits(torch::Tensor pack, int64_t target_bits,
                         int64_t unpack_dim);

torch::Tensor sq_compress(torch::Tensor src, torch::Tensor codebooks,
                          int64_t target_bits, int64_t column_slice);

torch::Tensor sq_decompress(torch::Tensor compress_data, torch::Tensor codebooks,
                   int64_t target_bits, int64_t column_slice,
                   int64_t feat_dim);

torch::Tensor vq_decompress(torch::Tensor index_tensor, torch::Tensor codebooks,
                   int64_t feat_dim);

torch::Tensor SpMMCsr(torch::Tensor ufeat, torch::Tensor indptr,
                      torch::Tensor indices);

torch::Tensor VqDecompressSpMMFusionCsr(torch::Tensor compress_data,
                                        torch::Tensor codebooks,
                                        torch::Tensor indptr,
                                        torch::Tensor indices,
                                        int64_t feat_dim);

torch::Tensor SqDecompressSpMMFusionCsr(torch::Tensor compress_data,
                                        torch::Tensor codebooks,
                                        torch::Tensor indptr,
                                        torch::Tensor indices,
                                        int64_t target_bits,
                                        int64_t column_slice, int64_t feat_dim);
