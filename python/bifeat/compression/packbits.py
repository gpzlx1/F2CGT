import math
import torch
import time

mask2nbits = {
    0b00000001: 1,
    0b00000011: 2,
    0b00001111: 4,
    0b11111111: 8,
}


def tensor_dim_slice(tensor, dim, s):
    return tensor[(slice(None), ) * (dim if dim >= 0 else dim + tensor.dim()) +
                  (s, )]


def packshape(shape, dim, mask, dtype):
    nbits_element = torch.iinfo(dtype).bits
    nbits = mask2nbits[mask]
    packed_size = nbits_element // nbits
    shape = list(shape)
    shape[dim] = math.ceil(shape[dim] / packed_size)
    return shape, packed_size, nbits


def packbits(tensor, dim=-1, mask=0b00000001, dtype=torch.uint8):
    shape, packed_size, nbits = packshape(tensor.shape,
                                          dim=dim,
                                          mask=mask,
                                          dtype=dtype)
    out = torch.zeros(shape, device=tensor.device, dtype=dtype)
    idx = 0
    for e in range(packed_size):
        width = (tensor.shape[dim] - e - 1) // packed_size + 1
        sliced_input = tensor_dim_slice(tensor, dim,
                                        slice(idx, idx + width, 1))
        idx += width
        compress = (sliced_input << (nbits * (packed_size - e - 1)))
        sliced_output = out.narrow(dim, 0, sliced_input.shape[dim])
        sliced_output |= compress
    return out


def unpackbits(tensor, shape, dim=-1, mask=0b00000001):
    _, packed_size, nbits = packshape(
        shape,
        dim=dim,
        mask=mask,
        dtype=tensor.dtype,
    )

    ts = []
    for e in range(packed_size):
        ts.append(
            ((tensor >>
              (nbits *
               (packed_size - e - 1))).bitwise_and_((1 << nbits) - 1)).narrow(
                   dim, 0, (shape[dim] - e - 1) // packed_size + 1))

    return torch.cat(ts, -1)
