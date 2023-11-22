import torch
import time


def graph_reorder(coo_row, coo_col, reorder_map):
    reroder_row = reorder_map[coo_row]
    reorder_col = reorder_col[coo_col]
    sort_idx = torch.argsort(reroder_row)
    return reroder_row[sort_idx], reorder_col[sort_idx]


def dtype_sizeof(input):
    if isinstance(input, str):
        if input == "int32":
            return 4
        elif input == "int64":
            return 8
        elif input == "float32":
            return 4
        elif input == "float64":
            return 8
        elif input == "bool":
            return 1
    else:
        if input == torch.int32:
            return 4
        elif input == torch.int64:
            return 8
        elif input == torch.float32:
            return 4
        elif input == torch.float64:
            return 8
        elif input == torch.bool:
            return 1


class FeatureCache:

    def __init__(self, feature, idx_dtype=torch.int64):
        self._feature = feature
        self._cached_feature = None
        self._hashmap = None
        self._full_cached = False
        self._no_cached = False
        self._num_nodes = feature.shape[0]
        self._idx_dtype = idx_dtype
        self._hit_num = 0
        self._lookup_num = 0

    def create_cache(self, alloca_mem_size, hotness):

        tic = time.time()

        mem_used = 0

        info = "========================================\n"
        info += "Rank {} builds feature cache in GPU\n".format(
            torch.cuda.current_device())
        info += "GPU alloca_mem_size {:.3f} GB\n".format(alloca_mem_size /
                                                         1024 / 1024 / 1024)

        idx_dtype_size = dtype_sizeof(self._idx_dtype)
        item_size = self._feature.element_size() * self._feature.shape[1]
        cached_num = min(alloca_mem_size // item_size, self._num_nodes)
        if cached_num < self._num_nodes:
            item_size_with_hash = item_size + idx_dtype_size * 8
            cached_num = alloca_mem_size // item_size_with_hash

        info += "Total node num {}\n".format(self._num_nodes)
        info += "GPU cached node num {}\n".format(cached_num)

        if cached_num <= 0:
            self._no_cached = True
            torch.ops.pg_ops._CAPI_pin_tensor(self._feature)
            mem_used = 0
        elif cached_num == self._num_nodes:
            self._full_cached = True
            self._cached_feature = self._feature.cuda()
            mem_used = self._feature.element_size() * self._feature.numel()
        else:
            cached_idx = torch.argsort(hotness, descending=True)[:cached_num]
            self._hashmap = torch.ops.pg_ops._CAPI_create_hashmap(
                cached_idx.cuda())
            self._cached_feature = self._feature[cached_idx].cuda()
            torch.ops.pg_ops._CAPI_pin_tensor(self._feature)
            mem_used = cached_num * item_size_with_hash

        toc = time.time()

        self._hit_num = 0
        self._lookup_num = 0
        info += "GPU cache size {:.3f} GB\n".format(mem_used / 1024 / 1024 /
                                                    1024)
        info += "Build cache time {:.3f} ms\n".format((toc - tic) * 1000)
        info += "========================================"
        print(info)

        return mem_used

    def __getitem__(self, index):
        '''
        index is a gpu tensor
        and return gpu tensors (step, mem, power)
        '''
        if self._full_cached:
            return self._cached_feature[index]
        elif self._no_cached:
            return torch.ops.pg_ops._CAPI_fetch_feature_data(
                self._feature, index)
        else:
            return torch.ops.pg_ops._CAPI_fetch_feature_data_with_caching(
                self._feature,
                self._cached_feature,
                self._hashmap[0],
                self._hashmap[1],
                index,
            )

    def __del__(self):
        if not self._full_cached:
            torch.ops.pg_ops._CAPI_unpin_tensor(self._feature)
