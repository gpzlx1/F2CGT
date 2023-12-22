import torch
import BiFeatLib
from collections import namedtuple
import os
import torch.distributed as dist

SHM_ALLOC_AMPL_TIMES: float = 1.01
TensorMeta = namedtuple(
    "TensorMeta", ["shm_name", "fd", "ptr", "shm_size", "dtype", "shape"])


def str2dtype(input: str):
    if input == "int32":
        return torch.int32
    elif input == "int64":
        return torch.int64
    elif input == "float32":
        return torch.float32
    elif input == "float64":
        return torch.float64
    elif input == "bool":
        return torch.bool
    elif input == "uint8":
        return torch.uint8
    elif input == "int8":
        return torch.int8
    elif input == "int16":
        return torch.int16
    elif input == "float16":
        return torch.float16


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
        elif input == "int16" or input == "float16":
            return 2
        elif input == "bool" or input == "uint8" or input == "int8":
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
        elif input == torch.int16 or input == torch.float16:
            return 2
        elif input == torch.bool or input == torch.uint8 or input == torch.int8:
            return 1


def create_shm_mem(shm_name, malloc_size, pin_memory=True):
    ptr, fd = BiFeatLib.create_shared_mem(shm_name, malloc_size, pin_memory)
    return ptr, fd


def release_shm_mem(shm_name, shm_size, ptr, fd, pin_memory=True):
    BiFeatLib.release_shared_mem(shm_name, shm_size, ptr, fd, pin_memory)


def open_shm_mem(shm_name, shm_size, pin_memory=True):
    ptr, fd = BiFeatLib.open_shared_mem(shm_name, shm_size, pin_memory)
    return ptr, fd


def open_shared_tensor(ptr, dtype, shape):
    tensor = BiFeatLib.open_shared_tensor(ptr, dtype, shape)
    return tensor


class ShmManager(object):

    def __init__(
        self,
        rank,
        num_gpus,
        dataset_path,
        dataset_name,
        pin_memory=False,
    ):
        self._shm_tensor_meta = {}
        self._create_tensor_count = 0
        self._pin_memory = pin_memory
        self._rank = rank
        self._num_gpus = num_gpus
        self._is_chief = rank % num_gpus == 0
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name

        cur_local_group, local_groups = dist.new_subgroups(num_gpus)
        self._local_group = cur_local_group

    def _create_shm_name(self, tensor_name):
        shm_name = tensor_name + str(self._create_tensor_count)
        self._create_tensor_count += 1
        return shm_name

    def _create_shm_tensor(self, tensor_name, dtype, shape):
        if isinstance(dtype, str):
            dtype = str2dtype(dtype)
        numel = 1
        for dim_size in shape:
            numel *= dim_size
        if numel <= 0:
            return None

        need_size = numel * dtype_sizeof(dtype)

        if tensor_name not in self._shm_tensor_meta:
            malloc_size = int(need_size * SHM_ALLOC_AMPL_TIMES)
            # alloc shm
            shm_name = self._create_shm_name(tensor_name)
            ptr, fd = create_shm_mem(shm_name, malloc_size, self._pin_memory)
            # insert meta
            self._shm_tensor_meta[tensor_name] = TensorMeta(
                shm_name, fd, ptr, malloc_size, dtype, shape)
            return open_shared_tensor(ptr, dtype, shape)

        elif need_size <= self._shm_tensor_meta[tensor_name].shm_size:
            # update meta
            old_meta = self._shm_tensor_meta[tensor_name]
            self._shm_tensor_meta[tensor_name] = TensorMeta(
                old_meta.shm_name, old_meta.fd, old_meta.ptr,
                old_meta.shm_size, dtype, shape)
            return open_shared_tensor(self._shm_tensor_meta[tensor_name].ptr,
                                      dtype, shape)

        else:
            # realloc shm
            old_meta = self._shm_tensor_meta[tensor_name]
            release_shm_mem(old_meta.shm_name, old_meta.shm_size, old_meta.ptr,
                            old_meta.fd, self._pin_memory)
            malloc_size = int(need_size * SHM_ALLOC_AMPL_TIMES)
            shm_name = self._create_shm_name(tensor_name)
            ptr, fd = create_shm_mem(shm_name, malloc_size, self._pin_memory)
            # update meta
            self._shm_tensor_meta[tensor_name] = TensorMeta(
                shm_name, fd, ptr, malloc_size, dtype, shape)
            return open_shared_tensor(ptr, dtype, shape)

    def _open_shm_tensor(self, tensor_name, shm_name, shm_size, dtype, shape):
        if tensor_name in self._shm_tensor_meta:
            old_meta = self._shm_tensor_meta[tensor_name]
            if shm_name != old_meta.shm_name:
                # release old shm and open the new one
                release_shm_mem(old_meta.shm_name, old_meta.shm_size,
                                old_meta.ptr, old_meta.fd, self._pin_memory)
                ptr, fd = open_shm_mem(shm_name, shm_size, self._pin_memory)

                # update meta
                self._shm_tensor_meta[tensor_name] = TensorMeta(
                    shm_name, fd, ptr, shm_size, dtype, shape)
                return open_shared_tensor(ptr, dtype, shape)

            else:
                # just re-open tensor, update meta
                self._shm_tensor_meta[tensor_name] = TensorMeta(
                    shm_name, old_meta.fd, old_meta.ptr, shm_size, dtype,
                    shape)
                return open_shared_tensor(old_meta.ptr, dtype, shape)
        else:
            # open shm
            ptr, fd = open_shm_mem(shm_name, shm_size, self._pin_memory)

            # insert meta
            self._shm_tensor_meta[tensor_name] = TensorMeta(
                shm_name, fd, ptr, shm_size, dtype, shape)
            return open_shared_tensor(ptr, dtype, shape)

    def _release_shm_tensor(self, tensor_name: str):
        meta = self._shm_tensor_meta.pop(tensor_name)
        release_shm_mem(meta.shm_name, meta.shm_size, meta.ptr, meta.fd,
                        self._pin_memory)

    def __del__(self):
        # release all shared memory tensors
        # stop all connections
        for tensor_name in self._shm_tensor_meta:
            meta = self._shm_tensor_meta[tensor_name]
            release_shm_mem(meta.shm_name, meta.shm_size, meta.ptr, meta.fd,
                            self._pin_memory)

    def create_shm_tensor(self, tensor_name, dtype, shape):
        ret = None
        if self._is_chief:
            ret = self._create_shm_tensor(tensor_name, dtype, shape)
            meta = self._shm_tensor_meta[tensor_name]
        else:
            meta = None

        objects = [meta]
        dist.broadcast_object_list(objects)
        meta = objects[0]

        if not self._is_chief:
            ret = self._open_shm_tensor(tensor_name, meta.shm_name,
                                        meta.shm_size, meta.dtype, meta.shape)

        return ret

    def release_shm_tensor(self, tensor_name):
        self._release_shm_tensor(tensor_name)

    def load_dataset(self, with_feature=True, with_valid=True, with_test=True):
        print("load {}...".format(self.dataset_name))

        meta_data = torch.load(os.path.join(self.dataset_path, "metadata.pt"))
        assert meta_data["dataset"] == self.dataset_name
        self.graph_meta_data = meta_data

        if self._is_chief:
            labels = torch.load(os.path.join(self.dataset_path, "labels.pt"))
            indptr = torch.load(os.path.join(self.dataset_path, "indptr.pt"))
            indices = torch.load(os.path.join(self.dataset_path, "indices.pt"))
            train_idx = torch.load(
                os.path.join(self.dataset_path, "train_idx.pt"))

            shm_labels = self.create_shm_tensor(
                self.dataset_name + "_shm_labels", labels.dtype, labels.shape)
            shm_indptr = self.create_shm_tensor(
                self.dataset_name + "_shm_indptr", indptr.dtype, indptr.shape)
            shm_indices = self.create_shm_tensor(
                self.dataset_name + "_shm_indices", indices.dtype,
                indices.shape)
            shm_train_idx = self.create_shm_tensor(
                self.dataset_name + "_shm_train_idx", train_idx.dtype,
                train_idx.shape)

            shm_labels.copy_(labels)
            shm_indptr.copy_(indptr)
            shm_indices.copy_(indices)
            shm_train_idx.copy_(train_idx)

            if with_valid:
                valid_idx = torch.load(
                    os.path.join(self.dataset_path, "valid_idx.pt"))
                shm_valid_idx = self.create_shm_tensor(
                    self.dataset_name + "_shm_valid_idx", valid_idx.dtype,
                    valid_idx.shape)
                shm_valid_idx.copy_(valid_idx)

            if with_test:
                test_idx = torch.load(
                    os.path.join(self.dataset_path, "test_idx.pt"))
                shm_test_idx = self.create_shm_tensor(
                    self.dataset_name + "_shm_test_idx", test_idx.dtype,
                    test_idx.shape)
                shm_test_idx.copy_(test_idx)

            if with_feature:
                features = torch.load(
                    os.path.join(self.dataset_path, "features.pt"))
                shm_features = self.create_shm_tensor(
                    self.dataset_name + "_shm_features", features.dtype,
                    features.shape)
                shm_features.copy_(features)

        else:
            shm_labels = self.create_shm_tensor(
                self.dataset_name + "_shm_labels", None, None)
            shm_indptr = self.create_shm_tensor(
                self.dataset_name + "_shm_indptr", None, None)
            shm_indices = self.create_shm_tensor(
                self.dataset_name + "_shm_indices", None, None)
            shm_train_idx = self.create_shm_tensor(
                self.dataset_name + "_shm_train_idx", None, None)
            if with_valid:
                shm_valid_idx = self.create_shm_tensor(
                    self.dataset_name + "_shm_valid_idx", None, None)
            if with_test:
                shm_test_idx = self.create_shm_tensor(
                    self.dataset_name + "_shm_test_idx", None, None)
            if with_feature:
                shm_features = self.create_shm_tensor(
                    self.dataset_name + "_shm_features", None, None)

        graph_tensors = {
            "labels": shm_labels,
            "indptr": shm_indptr,
            "indices": shm_indices,
            "train_idx": shm_train_idx,
        }
        if with_valid:
            graph_tensors["valid_idx"] = shm_valid_idx
        if with_test:
            graph_tensors["test_idx"] = shm_test_idx
        if with_feature:
            graph_tensors["features"] = shm_features

        dist.barrier(self._local_group)
        print("finish loading {}...".format(self.dataset_name))

        return graph_tensors, meta_data
