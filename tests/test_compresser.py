import torch as th
from pagraph import Compresser

if __name__ == "__main__":
    compresser = Compresser("sq")
    features = th.randn(5000, 16)
    compressed_features = compresser.compress(features)
    # decompressed_features = compresser.decompress(compressed_features)
    # print(features, decompressed_features)
    # print(features.shape, compressed_features.shape,
    #      decompressed_features.shape)
    # print(features.abs().mean(),
    #      (decompressed_features - features).abs().mean())
