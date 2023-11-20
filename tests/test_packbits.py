import torch
import time
from pagraph import packbits, unpackbits, mypackbits

if __name__ == '__main__':
    # shape = (10, 20)
    shape = (500, 200000)
    K = 1
    for nbits in [1]:
        mask = (1 << nbits) - 1
        for dtype in [torch.uint8]:
            for k in range(K):
                x = torch.randint(0, 1 << nbits, shape, dtype=dtype)
                x = torch.concat([x] * 10, dim=0)
                print("compressing")

                t0 = time.time()
                y1 = packbits(x, mask=mask)
                t1 = time.time()

                y2 = mypackbits(x, mask=mask)
                t2 = time.time()

                y = y2
                print("done.", y.size())
                print("decompressing")

                z = unpackbits(y, mask=mask, dtype=x.dtype, shape=x.shape)
                # print(z)
                diff = (x - z).abs().sum().item()
                print("difference =", diff)
                if diff != 0:
                    print("error: mismatch")
                    exit(1)
                print("done.", z.size())

                print(t1-t0, t2-t1)
                # assert torch.allclose(x, z)
