#pragma once

#include <fcntl.h>
#include <string.h>
#include <sys/mman.h>
#include <torch/python.h>
#include <torch/script.h>
#include <unistd.h>

namespace shm {

std::tuple<int64_t, int64_t> create_shared_mem(std::string name, int64_t size,
                                               bool pin_memory = true) {
  int flag = O_RDWR | O_CREAT;
  int fd = shm_open(name.c_str(), flag, S_IRUSR | S_IWUSR);
  CHECK(fd != -1) << "fail to open " << name << ": " << strerror(errno);
  // Shared memory cannot be deleted if the process exits abnormally in Linux.
  int res = ftruncate(fd, (size_t)size);
  CHECK(res != -1) << "Failed to truncate the file. " << strerror(errno);
  void *ptr =
      mmap(NULL, (size_t)size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  CHECK(ptr != MAP_FAILED)
      << "Failed to map shared memory. mmap failed with error "
      << strerror(errno);
  if (pin_memory) {
    CUDA_CALL(cudaHostRegister(ptr, (size_t)size, cudaHostRegisterDefault));
  }
  return std::make_tuple((int64_t)ptr, (int64_t)fd);
}

std::tuple<int64_t, int64_t> open_shared_mem(std::string name, int64_t size,
                                             bool pin_memory = true) {
  int flag = O_RDWR;
  int fd = shm_open(name.c_str(), flag, S_IRUSR | S_IWUSR);
  CHECK(fd != -1) << "fail to open " << name << ": " << strerror(errno);
  void *ptr =
      mmap(NULL, (size_t)size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  CHECK(ptr != MAP_FAILED)
      << "Failed to map shared memory. mmap failed with error "
      << strerror(errno);
  if (pin_memory) {
    CUDA_CALL(cudaHostRegister(ptr, (size_t)size, cudaHostRegisterDefault));
  }
  return std::make_tuple((int64_t)ptr, (int64_t)fd);
}

void release_shared_mem(std::string name, int64_t size, int64_t ptr, int64_t fd,
                        bool pin_memory = true) {
  if (pin_memory) {
    CUDA_CALL(cudaHostUnregister((void *)ptr));
  }
  CHECK(munmap((void *)ptr, (size_t)size) != -1) << strerror(errno);
  close((int)fd);
  shm_unlink(name.c_str());
}

torch::Tensor open_shared_tensor(int64_t ptr, pybind11::object dtype,
                                 std::vector<int64_t> shape) {
  torch::ScalarType type = torch::python::detail::py_object_to_dtype(dtype);
  return torch::from_blob((void *)ptr, shape,
                          torch::TensorOptions().dtype(type));
}

class SharedMemory {
  int _fd = 0;
  void *_ptr = nullptr;
  size_t _size = 0;
  bool _pin_memory = true;
  std::string _name;

 public:
  std::string GetName_() const { return _name; }

  explicit SharedMemory(const std::string &name, bool pin_memory = true) {
    this->_name = name;
    this->_fd = -1;
    this->_ptr = nullptr;
    this->_size = 0;
    this->_pin_memory = pin_memory;
  }

  ~SharedMemory() {
    release_shared_mem(this->_name, this->_size, (int64_t)this->_ptr,
                       (int64_t)this->_fd, this->_pin_memory);
  }

  void *CreateNew(size_t size) {
    int64_t ptr, fd;
    std::tie(ptr, fd) =
        create_shared_mem(_name.c_str(), size, this->_pin_memory);
    this->_ptr = (void *)ptr;
    this->_fd = (int)fd;
    this->_size = size;
    return this->_ptr;
  }

  void *Open(size_t size) {
    int64_t ptr, fd;
    std::tie(ptr, fd) = open_shared_mem(_name.c_str(), size, this->_pin_memory);
    this->_ptr = (void *)ptr;
    this->_fd = (int)fd;
    this->_size = size;
    return this->_ptr;
  }

  static bool Exist(const std::string &name) {
    int fd = shm_open(name.c_str(), O_RDONLY, S_IRUSR | S_IWUSR);
    if (fd >= 0) {
      close(fd);
      return true;
    } else {
      return false;
    }
  }
};

}  // namespace shm