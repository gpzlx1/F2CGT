import torch
import time
from bifeat import packbits, unpackbits

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
                y = packbits(x, mask=mask)
                t1 = time.time()

                print("done.", y.size())
                print("decompressing")

                z = unpackbits(y, mask=mask, shape=x.shape)

                print(x.shape, x)
                print(y.shape, y)
                print(z.shape, z)

                diff = (x - z).abs().sum().item()
                print("difference =", diff)
                if diff != 0:
                    print("error: mismatch")
                    exit(1)
                print("done.", z.size())
