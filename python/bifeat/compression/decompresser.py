import BiFeatLib as capi
import torch


class Decompresser(object):

    def __init__(self, feat_dim, codebooks, methods, part_size_list):
        self.codebooks = codebooks
        self.codebooks = [i.cuda() for i in self.codebooks]

        self.feat_dim = feat_dim
        self.methods = methods
        # bugs for chunknum???
        chunk_num = [i.shape[0] for i in codebooks]
        self.chunk_num = torch.tensor(chunk_num)
        self.part_size_list = torch.tensor(part_size_list).long()
        self.chunk_size = (self.part_size_list + self.chunk_num -
                           1) // self.chunk_num
        self.chunk_size = self.chunk_size.tolist()

    def decompress(self, compressed_feat, local_part_indices, partition_id):
        """
        decompress features of a single compression partition
        """
        assert compressed_feat.shape[0] == local_part_indices.shape[0]

        local_codebook_indices = local_part_indices // self.chunk_size[
            partition_id]
        if self.methods[partition_id] == 'sq':
            return capi._CAPI_sq_decompress(local_codebook_indices,
                                            compressed_feat,
                                            self.codebooks[partition_id],
                                            self.feat_dim)
        elif self.methods[partition_id] == 'vq':
            return capi._CAPI_vq_decompress(local_codebook_indices,
                                            compressed_feat,
                                            self.codebooks[partition_id],
                                            self.feat_dim)
        else:
            raise ValueError

    def decompress_v2(self, compressed_feat, local_part_indices, partition_id,
                      output_tensor, output_offset):
        if self.methods[partition_id] == 'sq':
            capi._CAPI_sq_decompress_v2(local_part_indices,
                                        self.chunk_size[partition_id],
                                        compressed_feat,
                                        self.codebooks[partition_id],
                                        output_tensor, output_offset)
        elif self.methods[partition_id] == 'vq':
            capi._CAPI_vq_decompress_v2(local_part_indices,
                                        self.chunk_size[partition_id],
                                        compressed_feat,
                                        self.codebooks[partition_id],
                                        output_tensor, output_offset)
        else:
            raise ValueError
