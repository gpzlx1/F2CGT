import torch
from pagraph import vq_compress, vq_decompress
import time

torch.manual_seed(42)

DEVICE = 'cuda'
SIZE = 500_0000
FEAT_DIM = 128
WIDTH = 32
LENGTH = 256

features = torch.randn(SIZE, FEAT_DIM).float()

begin = time.time()
compressed_features, codebook = vq_compress(features, WIDTH, LENGTH, DEVICE)
torch.cuda.synchronize()
end = time.time()
print("Time: {}".format((end - begin) * 1000))
print(compressed_features.shape, codebook.shape)
print(compressed_features.dtype, codebook.dtype)
print(compressed_features.device, codebook.device)

print()
begin = time.time()
decompress_features = vq_decompress(compressed_features, FEAT_DIM, codebook)
torch.cuda.synchronize()
end = time.time()
print("Time: {}".format((end - begin) * 1000))
print(decompress_features.shape)
print(decompress_features.dtype)
print(decompress_features.device)

print(features[0])
print(decompress_features[0])
print(
    torch.nn.functional.cosine_similarity(features[10:],
                                          decompress_features[10:],
                                          dim=1))

print()

for i in range(2):
    begin = time.time()
    compressed_features, codebook = vq_compress(features, WIDTH, LENGTH,
                                                DEVICE)
    torch.cuda.synchronize()
    end = time.time()
    print("Time: {}".format((end - begin) * 1000))

print()

compressed_features = compressed_features.to(DEVICE)
codebook = codebook.to(DEVICE)
torch.cuda.synchronize()
for i in range(2):
    print()
    begin = time.time()
    decompress_features = vq_decompress(compressed_features, FEAT_DIM,
                                        codebook)
    torch.cuda.synchronize()
    end = time.time()
    print("Time: {}".format((end - begin) * 1000))
