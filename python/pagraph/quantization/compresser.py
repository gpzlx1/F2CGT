import torch as th
import math
import numpy as np
import os
import tqdm

from .kmeans import kmeans, get_centers, kmeans_predict
from .packbits import packbits, unpackbits


class Compresser(object):

    def __init__(self,
                 mode="sq,sp",
                 length="1,16",
                 width="1,256",
                 device="cpu"):
        self.mode = mode.split(",")
        self.length = []
        length = length.split(",")
        self.width = [int(t) for t in width.split(",")]
        self.fn = []
        self.compress_level = len(self.mode)
        self.compression_ratio = [1 for i in range(self.compress_level)]

        for i in range(len(self.mode)):
            if self.mode[i] == "sq" or self.mode[i] == "sp":
                self.length.append(int(length[i]))
            else:
                self.length.append([int(l) for l in length[i].split("-")])
            self.fn.append(self.mode[i] + "_" + length[i] + "_" +
                           str(self.width[i]) + ".pkl")

        print(self.mode, self.width, self.length)

        # self.batch_size = batch_size
        self.device = device
        self.feat_dim = -1
        self.quantized = [True for i in range(self.compress_level)]
        self.codebooks = []

    def compress(self,
                 features,
                 dn=None,
                 batch_size=-1,
                 ignore_cache=False,
                 root="/data/quantized/"):
        self.batch_size = batch_size
        self.cache_path = []
        compressed = []
        self.feat_dim = features.shape[1]
        for i in range(self.compress_level):
            print("Compressing tier", i + 1, self.mode[i], self.width[i],
                  self.length[i])
            if self.mode[i] == "sq":
                self.compression_ratio[i] = 32 / self.length[i]
            elif self.mode[i] == "sp":
                self.compression_ratio[i] = 4 * self.feat_dim / math.ceil(
                    self.feat_dim / 256) / self.length[i]
            elif self.mode[i] == "vq":
                self.compression_ratio[i] = self.width[i] * 32 / math.log2(
                    self.length[i][0])

            if dn:
                self.cache_path.append(
                    os.path.join(root, dn + "_" + self.fn[i]))
            if dn and not ignore_cache and os.path.exists(self.cache_path[i]):
                (_compressed, codebook) = th.load(self.cache_path[i])
                self.codebooks.append(codebook)
                self.quantized[i] = True
                compressed.append(_compressed)
                del _compressed
                print("  Loaded", self.cache_path[i])
            else:

                if self.mode[i] == "sq":
                    compressed.append(self.sq_compresser(features, i))

                elif self.mode[i] == "vq":
                    if self.batch_size == -1:
                        compressed.append(self.vq_compresser(features, i))
                    else:
                        compressed.append(self.vq_compresser_batch(
                            features, i))
                elif self.mode[i] == "sp":
                    compressed.append(self.sp_compresser(features, i))
                else:
                    raise ValueError("mode must be sq or vq or sp")
                if self.quantized[i] and dn and not ignore_cache:
                    th.save((compressed[i], self.codebooks[i]),
                            self.cache_path[i])
                    print("  Cached", self.cache_path[i])
        print("compression ratio per tier:", self.compression_ratio)
        del features
        return compressed

    def vq_compresser(self, features, tier):

        self.quantized[tier] = True
        vqlevel = len(self.length[tier])
        print(vqlevel, "vq levels")
        length = self.length[tier]
        width = self.width[tier]
        # if not th.is_tensor(features):
        #     features = th.tensor(features, dtype=th.float32)
        print("in total ", math.ceil(features.shape[1] / width), " parts")
        codebooks = th.zeros((math.ceil(features.shape[1] / width), vqlevel,
                              max(length), width))
        if max(length) <= 256:
            dtype = th.uint8
        elif max(length) <= 32767:
            dtype = th.int16
        else:
            dtype = th.int32
        cluster_ids = th.empty(
            (features.shape[0], math.ceil(features.shape[1] / width), vqlevel),
            dtype=dtype)

        perm = th.randperm(features.shape[0])

        decay = 0.95
        for i in range(math.ceil(features.shape[1] / width)):
            print("quantizing part ", i)
            X = features[perm[:50000], i * width:i * width + width]
            if not th.is_tensor(X):
                X = th.tensor(X, dtype=th.float32)
            # for level, ll in enumerate(length):
            level = 0
            ll = length[0]
            print(level, X.norm(dim=1, p=2).mean().div(np.sqrt(X.shape[1])))
            method = "cosine"
            out_num = ll
            dist = X.norm(dim=1, p=2)
            rim = th.quantile(dist[:50000], 0.8 / ll)

            cluster_ids_x = th.empty(X.shape[0], dtype=th.int32)

            out = th.ge(dist, rim)

            cluster_centers_o = get_centers(X=X[out],
                                            num_clusters=out_num,
                                            distance=method,
                                            tol=5e-2 * out_num,
                                            device=self.device)

            if level > 0:
                cluster_centers_o *= decay
            codebooks[i, level, :, :features.shape[1] -
                      i * width] = cluster_centers_o
        del X
        for j in tqdm.trange(math.ceil(features.shape[0] / 65536),
                             mininterval=1):
            start = j * 65536
            end = (j + 1) * 65536

            features_ = th.tensor(features[start:end, :], dtype=th.float32)
            for i in range(math.ceil(features.shape[1] / width)):
                method = "cosine"
                X = features_[:, i * width:i * width + width]

                cluster_ids_x = kmeans_predict(
                    X,
                    codebooks[i, 0, :, :features.shape[1] - i * width],
                    method,
                    device=self.device)

                cluster_ids[start:end, i, 0] = cluster_ids_x
        del features_
        del features
        self.codebooks.append(codebooks)
        return cluster_ids

    def sp_compresser(self, features, tier):

        self.quantized[tier] = True
        length = self.length[tier]
        width = self.width[tier]

        compressed = th.empty(
            (features.shape[0], math.ceil(self.feat_dim / 256) * length),
            dtype=th.uint8)

        dn = length - length // 2
        codebooks = th.quantile(
            th.abs(th.tensor(features[:10000], dtype=th.float32)),
            (th.arange(128, 128 - dn, -1) - 0.5) / 128, 1).mean(1)
        # print(codebooks)
        for part in range(math.ceil(self.feat_dim / 256)):
            for batch in range(math.ceil(features.shape[0] / 65536)):
                h = batch * 65536
                t = min(h + 65536, features.shape[0])
                st = part * 256
                X = features[h:t, st:st + 256]
                if not th.is_tensor(X):
                    X = th.tensor(X, dtype=th.float32)
                tn = length // 2
                dn = length - length // 2
                # if st+256>self.feat_dim:
                #     tn = int(tn*(self.feat_dim-st)/256)
                #     dn = int(dn*(self.feat_dim-st)/256)
                tk = th.topk(X, tn, dim=1)
                dk = th.topk(X, dn, dim=1, largest=False)
                compressed[h:t, part * length:part * length + tn] = tk.indices
                compressed[h:t, part * length + length // 2:part * length +
                           length // 2 + dn] = dk.indices
        self.codebooks.append(codebooks)

        print(compressed.shape)
        return compressed

    def sq_compresser(self, features, tier):
        if not th.is_tensor(features):
            features = th.tensor(features, dtype=th.float16)
        # scalar quantization
        self.feat_dim = features.shape[1]
        # print(features[0:10])
        length = self.length[tier]
        width = self.width[tier]
        if not th.is_tensor(features):
            features = th.tensor(features, dtype=th.float16)
        if length == 32 or (length == 16 and features.dtype == th.float16):
            self.quantized[tier] = False
            return features
        else:
            self.quantized[tier] = True

        emin = 0
        emax = 0
        drange = 2**(length - 1)

        if length <= 8:
            dtype = th.int8
        elif length <= 16:
            dtype = th.int16
        else:
            dtype = th.int32

        if length < 8:
            tfeat_dim = int(math.ceil(self.feat_dim / 8 * length))
        else:
            tfeat_dim = self.feat_dim

        t_features = th.empty((features.shape[0], tfeat_dim), dtype=dtype)
        epsilon = 1e-5
        print("start compressing, precision=", length)
        perm = th.randperm(features.shape[0])
        sample = features[perm[:100000]]
        fmin = max(np.percentile(np.abs(sample), 0.5), epsilon)
        fmax = max(np.percentile(np.abs(sample), 99.5), 2 * epsilon)
        print(fmin, fmax)

        fmin = th.tensor(fmin)
        fmax = th.tensor(fmax)
        quantize_batch_size = 1000000
        for start in tqdm.trange(0, features.shape[0], quantize_batch_size):
            end = min(features.shape[0], start + quantize_batch_size)
            features_ = features[start:end].to(th.float32)

            sign = th.sign(features_)
            if drange == 1:
                features_ = th.where(sign <= 0, 0, 1)
            else:
                features_ = th.abs(features_)
                features_ = th.clip(features_, fmin, fmax)
                exp = th.log2(features_)
                emin = th.log2(fmin)
                emax = th.log2(fmax).add(epsilon)

                exp = th.floor((exp - emin) / (emax - emin) * drange)
                if length < 8:
                    features_ = th.where(sign <= 0, drange - 1 - exp,
                                         exp + drange)
                else:
                    features_ = th.where(sign <= 0, -1 - exp, exp)

            if length < 8:
                t_features[start:end] = packbits(features_.to(th.uint8),
                                                 mask=(1 << length) - 1)
            elif length == 8:
                t_features[start:end] = th.tensor(features_).to(th.int8)
            elif length <= 16:
                t_features[start:end] = th.tensor(features_).to(th.int16)
            else:
                t_features[start:end] = th.tensor(features_).to(th.int32)
            del features_

        mean = features[:10000].float().abs().mean()
        # mean = features[:10000].float().norm(1).div(features[:10000].shape[0]*features.shape[1])
        if mean < 0.1:
            mean += 0.1
        print(emin, emax, mean)
        info = th.zeros(4)
        info[0] = emin
        info[1] = emax
        info[2] = mean
        info[3] = drange
        self.codebooks.append(info)
        del features
        return t_features

    def load(self, compressed, indices, indptr, device):
        if device is None:
            device = self.device
        else:
            self.device = device
        for i in range(len(self.codebooks)):
            self.codebooks[i] = self.codebooks[i].to(device)
        result = th.zeros((len(indices), self.feat_dim),
                          dtype=th.float32,
                          device=device)
        indptr[self.compress_level] = len(indices)
        cr = 0
        for tier in range(self.compress_level):
            st, en = indptr[tier], indptr[tier + 1]
            feats = compressed[tier][indices[st:en]].to(device)
            if self.quantized[tier]:
                if self.mode[tier] == "vq":
                    self.vq_decompresser(feats, tier, result[st:en])
                elif self.mode[tier] == "sq":
                    self.sq_decompresser(feats, tier, result[st:en])
                elif self.mode[tier] == "sp":
                    self.sp_decompresser(feats, tier, result[st:en])
                else:
                    raise ValueError("mode should be vq or sq")
            else:
                result[st:en] = feats
            # print(tier, result[st:st+10])
            cr += (en - st) / self.compression_ratio[tier]
        cr = len(indices) / cr
        return result, cr

    def load_with_cache(self, compressed, indices, indptr, device, cacher):
        if device is None:
            device = self.device
        else:
            self.device = device
        for i in range(len(self.codebooks)):
            self.codebooks[i] = self.codebooks[i].to(device)
        result = th.zeros((len(indices), self.feat_dim),
                          dtype=th.float32,
                          device=device)
        indptr[self.compress_level] = len(indices)
        cr = 0
        for tier in range(self.compress_level - 1):
            st, en = indptr[tier], indptr[tier + 1]
            feats = compressed[tier][indices[st:en]].to(device)
            if self.quantized[tier]:
                if self.mode[tier] == "vq":
                    self.vq_decompresser(feats, tier, result[st:en])
                elif self.mode[tier] == "sq":
                    self.sq_decompresser(feats, tier, result[st:en])
                elif self.mode[tier] == "sp":
                    self.sp_decompresser(feats, tier, result[st:en])
                else:
                    raise ValueError("mode should be vq or sq")
            else:
                result[st:en] = feats
            # print(tier, result[st:st+10])
            cr += (en - st) / self.compression_ratio[tier]
        tier = self.compress_level - 1
        st, en = indptr[tier], indptr[tier + 1]
        # print(indices[st:en].shape)
        feats = cacher.fetch_data(indices[st:en]).to(device)
        if self.quantized[tier]:
            if self.mode[tier] == "vq":
                self.vq_decompresser(feats, tier, result[st:en])
            elif self.mode[tier] == "sq":
                self.sq_decompresser(feats, tier, result[st:en])
            elif self.mode[tier] == "sp":
                self.sp_decompresser(feats, tier, result[st:en])
            else:
                raise ValueError("mode should be vq or sq")
        else:
            result[st:en] = feats
        # print(tier, result[st:st+10])
        cr += (en - st) / self.compression_ratio[tier]
        cr = len(indices) / cr
        return result, cr

    # def decompress(self, compressed_features, seed_num, device=None, ratio=1.0):
    #     if device is None:
    #         device = self.device
    #     else:
    #         self.device = device
    #     for tier in range(self.compress_level):
    #         if self.quantized:
    #             if self.mode == "vq":
    #                 return self.vq_decompresser(compressed_features, seed_num, ratio)
    #             elif self.mode == "sq":
    #                 return self.sq_decompresser(compressed_features, seed_num, ratio)
    #             elif self.mode == "sp":
    #                 return self.sp_decompresser(compressed_features, seed_num, ratio)
    #             else:
    #                 raise ValueError("mode should be vq or sq")
    #         else:
    #             return compressed_features.to(th.float32).to(device)

    def vq_decompresser(self, compressed_features, tier, result):
        compressed_features = compressed_features.to(th.int64)
        codebooks = self.codebooks[tier]
        num_parts = codebooks.shape[0]
        width = self.width[tier]
        vqlevel = len(self.length[tier])

        for i in range(num_parts - 1):
            h = i * width
            t = (i + 1) * width
            for j in range(vqlevel):
                result[:, h:t] += th.index_select(
                    codebooks[i, j], 0, compressed_features[:, i, j].flatten())
        for j in range(vqlevel):
            result[:, (num_parts - 1) * width:] += th.index_select(
                codebooks[num_parts - 1,
                          j, :, :self.feat_dim - (num_parts - 1) * width], 0,
                compressed_features[:, num_parts - 1, j].flatten())

    def sp_decompresser(self, compressed_features, tier, result):
        codebooks = self.codebooks[tier]
        device = self.device
        compressed_features = compressed_features.to(th.int64)
        length = self.length[tier]

        for part in range(math.ceil(self.feat_dim / 256)):
            st = part * 256
            tn = length // 2
            dn = length - tn

            tidx = compressed_features[:, part*length: part*length+tn] \
                 + th.arange(compressed_features.size(0), device=device).view(-1, 1) * self.feat_dim + st
            result.view(-1)[tidx] = codebooks[:tn]
            didx = compressed_features[:, part*length+length//2: part*length+length//2+dn] \
                 + th.arange(compressed_features.size(0), device=device).view(-1, 1) * self.feat_dim + st
            result.view(-1)[didx] = -codebooks

    def sq_decompresser(self, compressed_features, tier, result):
        device = self.device
        # print(compressed_features.shape)

        info = self.codebooks[tier]
        emin = info[0]
        emax = info[1]
        mean = info[2]
        drange = info[3]
        length = self.length[tier]

        exp = compressed_features
        if length < 8:
            exp = unpackbits(exp,
                             mask=2 * drange - 1,
                             shape=[exp.shape[0], self.feat_dim],
                             dtype=th.uint8)
        if length > 1:
            if length < 8:
                exp = exp.to(th.float32) - drange
            exp = exp + 0.5

            sign = th.sign(exp)
            result[:] = exp.abs_().mul_(
                (emax - emin) / drange).add_(emin).exp2_().mul_(sign)
        else:
            result[:] = (exp.to(th.float32).sub_(0.5)).mul_(2 * mean)
