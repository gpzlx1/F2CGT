import torch
import BiFeatLib
from collections import namedtuple

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


class EmbCacheBase(object):

    def __init__(self, pin_memory=False):
        self._shm_tensor_meta = {}
        self._create_tensor_count = 0
        self._pin_memory = pin_memory

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
