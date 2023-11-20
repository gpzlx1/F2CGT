import torch
from pagraph import packbits, unpackbits

if __name__ == '__main__':
    shape = (10, 20)
    K = 1
    for nbits in [1]:
        mask = (1 << nbits) - 1
        for dtype in [torch.uint8]:
            for k in range(K):
                x = torch.randint(0, 1 << nbits, shape, dtype=dtype)
                print("compressing")

                y = packbits(x, mask=mask)
                print("done.", y.size())
                print("decompressing")

                z = unpackbits(y, mask=mask, dtype=x.dtype, shape=x.shape)
                print("done.", z.size())

                # print(t1-t0, t2-t1)
                # assert torch.allclose(x, z)
