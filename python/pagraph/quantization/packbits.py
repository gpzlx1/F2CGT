import math
import torch
import time


def tensor_dim_slice(tensor, dim, s):
    return tensor[(slice(None), ) * (dim if dim >= 0 else dim + tensor.dim()) +
                  (s, )]


def packshape(shape, dim, mask, dtype):
    nbits_element = torch.iinfo(dtype).bits
    nbits = 1 if mask == 0b00000001 else 2 if mask == 0b00000011 else 4 if mask == 0b00001111 else 8 if mask == 0b11111111 else None
    assert nbits is not None and nbits <= nbits_element and nbits_element % nbits == 0
    packed_size = nbits_element // nbits
    shape = list(shape)
    shape[dim] = int(math.ceil(shape[dim] / packed_size))
    return shape, packed_size, nbits


def packbits(tensor, dim=-1, mask=0b00000001, out=None, dtype=torch.uint8):
    shape, packed_size, nbits = packshape(tensor.shape,
                                          dim=dim,
                                          mask=mask,
                                          dtype=dtype)
    out = out.zero_() if out is not None else torch.zeros(
        shape, device=tensor.device, dtype=dtype)
    assert tuple(out.shape) == tuple(shape)
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


def unpackbits(tensor,
               shape,
               dim=-1,
               mask=0b00000001,
               out=None,
               dtype=torch.uint8):

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
