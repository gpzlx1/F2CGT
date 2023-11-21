import torch
from pagraph import sq_compress, sq_decompress
import time

DEVICE = 'cpu'
SIZE = 500_0000
FEAT_DIM = 128
TARGET_BITS = 8

features = torch.randn(SIZE, FEAT_DIM).float()

begin = time.time()
compressed_features, codebook = sq_compress(features, TARGET_BITS, DEVICE)
torch.cuda.synchronize()
end = time.time()
print("Time: {}".format((end - begin) * 1000))
print(compressed_features.shape, codebook.shape)
print(compressed_features.dtype, codebook.dtype)
print(compressed_features.device, codebook.device)

print()
begin = time.time()
decompress_features = sq_decompress(compressed_features, FEAT_DIM, codebook)
torch.cuda.synchronize()
end = time.time()
print("Time: {}".format((end - begin) * 1000))
print(decompress_features.shape)
print(decompress_features.dtype)
print(decompress_features.device)

# print(features[0])
# print(decompress_features[0])
print()

for i in range(5):
    begin = time.time()
    compressed_features, codebook = sq_compress(features, TARGET_BITS, 'cuda')
    torch.cuda.synchronize()
    end = time.time()
    print("Time: {}".format((end - begin) * 1000))

print()

compressed_features = compressed_features.to(DEVICE)
codebook = codebook.to(DEVICE)
torch.cuda.synchronize()
for i in range(5):
    print()
    begin = time.time()
    decompress_features = sq_decompress(compressed_features, FEAT_DIM,
                                        codebook)
    torch.cuda.synchronize()
    end = time.time()
    print(decompress_features.shape, decompress_features.device,
          decompress_features.dtype)
    print("Time: {}".format((end - begin) * 1000))
