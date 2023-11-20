#ifndef PACKBITS_H_

#include <torch/script.h>

namespace pg {


// packbits
// note: "Only int8_t, int64_t and bool are supported as an integral argument type"
// #include <iostream>
torch::Tensor packbits(torch::Tensor tensor, int64_t dim, int64_t mask) {
    uint8_t _mask = mask & (0b11111111);
    auto shape = tensor.sizes().vec();
    // nbits_element from the dtype of tensor
    int64_t nbits_element = 8;
    auto dtype = tensor.dtype();
    if (dtype == torch::kInt8 || dtype == torch::kUInt8) {
        nbits_element = 8;
    } else if (dtype == torch::kInt16) {
        nbits_element = 16;
    } else if (dtype == torch::kInt32) {
        nbits_element = 32;
    } else if (dtype == torch::kInt64) {
        nbits_element = 64;
    } else {
        throw std::runtime_error("Unsupported dtype");
    }

    // std::cerr << __LINE__ << std::endl;

    int64_t nbits = 1;
    if (_mask == 0b00000001) {
        nbits = 1;
    } else if (_mask == 0b00000011) {
        nbits = 2;
    } else if (_mask == 0b00001111) {
        nbits = 4;
    } else if (_mask == 0b11111111) {
        nbits = 8;
    } else {
        throw std::runtime_error("Unsupported mask");
    }

    // packed_size: how many elements are packed into one element
    int64_t packed_size = nbits_element / nbits;
    size_t orig_shape = shape[dim];
    shape[dim] = int64_t((shape[dim] - 1) / packed_size) + 1;

    // create a new tensor
    torch::Tensor out = torch::zeros(shape, tensor.options());

    TORCH_CHECK(tensor.is_contiguous(), "tensor must be contiguous");
    size_t stride = tensor.strides()[dim];    // should be the same for out
    size_t full_len = tensor.numel();

    // I might also need to deal with data types other than uint8
    // in that case, just apply a stride to omit MSB
    auto baseptr = tensor.data_ptr<uint8_t>();
    auto dst_baseptr = out.data_ptr<uint8_t>();

    size_t block_stride = stride * orig_shape;
    size_t dst_block_stride = stride * shape[dim];
    for (size_t outer_block = 0;
         outer_block < full_len;
         outer_block += block_stride,
           baseptr += block_stride,
           dst_baseptr += dst_block_stride) {

        // the divided dimension: (0, orig_shape) -> (0, shape[dim])
        size_t orig_idx = 0;
        size_t dst_idx = 0;

        // enumerate the bit offset = (e * nbits)
        // constant number of loops
        for (size_t e = 0; e < packed_size; e++) {
            size_t lsh_offset = (packed_size - e - 1) * nbits;

            dst_idx = 0;
            size_t width = (orig_shape - e - 1) / packed_size + 1;
            size_t orig_end = orig_idx + width;
            if (orig_end > orig_shape)
                orig_end = orig_shape;

            while (orig_idx < orig_end) {
                auto orig_ptr = baseptr + orig_idx * stride;
                auto orig_end = orig_ptr + stride;
                auto dst_ptr = dst_baseptr + dst_idx * stride;

                for (; orig_ptr != orig_end;
                       orig_ptr += 1,
                       dst_ptr += 1) {
                    // OPTIMIZE: I can use SIMD here.
                    uint8_t val = *orig_ptr;
                    *dst_ptr |= uint8_t(val << lsh_offset);
                }

                ++orig_idx;
                ++dst_idx;
            }
        }
    }

    return out;
};

};

#define PACKBITS_H_
#endif