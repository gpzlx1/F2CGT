import BiFeatLib as capi
import torch


class Decompresser(object):

    def __init__(self, feat_dim, codebooks, methods, part_size_list):
        self.codebooks = codebooks
        self.feat_dim = feat_dim
        self.methods = methods
        self.chunk_num = codebooks[0].shape[0]
        self.part_size_list = torch.tensor(part_size_list).long()
        self.chunk_size = (self.part_size_list + self.chunk_num -
                           1) // self.chunk_num

    def decompress(self, compressed_feat, local_part_indices, partition_id):
        """
        decompress features of a single compression partition
        """
        assert compressed_feat.shape[0] == local_part_indices.shape[0]

        local_codebook_indices = local_part_indices // self.chunk_size[
            partition_id]
        if self.methods[partition_id] == 'sq':
            return capi._CAPI_sq_decompress(
                local_codebook_indices, compressed_feat.cuda(),
                self.codebooks[partition_id].cuda(), self.feat_dim)
        elif self.methods[partition_id] == 'vq':
            return capi._CAPI_vq_decompress(
                local_codebook_indices, compressed_feat.cuda(),
                self.codebooks[partition_id].cuda(), self.feat_dim)
        else:
            raise ValueError
