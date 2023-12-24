#ifndef PG_HASHMAP_H_
#define PG_HASHMAP_H_

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <torch/script.h>
#include <cuco/dynamic_map.cuh>
#include "../common.h"
#include "atomic.h"

#define INIT_CAPACITY 10

namespace bifeat {

template <typename KeyType, typename ValueType>
class CacheHashMap {
  using PairType = cuco::pair<KeyType, ValueType>;

 private:
  cuco::dynamic_map<KeyType, ValueType> map{static_cast<size_t>(INIT_CAPACITY),
                                            cuco::empty_key<KeyType>{-1},
                                            cuco::empty_value<ValueType>{-1}};

 public:
  CacheHashMap() {}
  void insert(torch::Tensor key_nids) {
    ValueType num_keys = key_nids.numel();
    thrust::device_vector<PairType> pairs(num_keys);
    using it = thrust::counting_iterator<ValueType>;
    thrust::for_each(it(0), it(num_keys),
                     [in = key_nids.data_ptr<KeyType>(),
                      out = pairs.begin()] __device__(ValueType i) mutable {
                       out[i] = PairType({in[i], i});
                     });
    map.insert(pairs.begin(), pairs.end());
  }
  torch::Tensor find(torch::Tensor key_nids) {
    torch::Tensor result =
        torch::full_like(key_nids, -1,
                         torch::TensorOptions()
                             .dtype(torch::CppTypeToScalarType<ValueType>())
                             .device(torch::kCUDA));
    ValueType num_keys = key_nids.numel();
    thrust::device_ptr<KeyType> keys(
        static_cast<KeyType *>(key_nids.data_ptr<KeyType>()));
    thrust::device_ptr<ValueType> values(
        static_cast<ValueType *>(result.data_ptr<ValueType>()));
    map.find(keys, keys + num_keys, values);
    return result;
  }
};

template <typename IdType>
struct Hashmap {
  __device__ inline Hashmap(IdType *__restrict__ Kptr,
                            int32_t *__restrict__ Vptr, size_t numel)
      : kptr(Kptr), vptr(Vptr), capacity(numel){};

  __device__ inline void Update(IdType key, int32_t value) {
    uint32_t delta = 1;
    uint32_t pos = hash(key);
    IdType prev = atomic::AtomicCAS(&kptr[pos], kEmptyKey, key);

    while (prev != key and prev != kEmptyKey) {
      pos = hash(pos + delta);
      delta += 1;
      prev = atomic::AtomicCAS(&kptr[pos], kEmptyKey, key);
    }

    vptr[pos] = value;
  }

  __device__ inline int32_t SearchForPos(IdType key) {
    uint32_t delta = 1;
    uint32_t pos = hash(key);

    while (true) {
      if (kptr[pos] == key) {
        return pos;
      }
      if (kptr[pos] == kEmptyKey) {
        return -1;
      }
      pos = hash(pos + delta);
      delta += 1;
    }
  }

  __device__ inline uint32_t Hash32Shift(uint32_t key) {
    key = ~key + (key << 15);  // key = (key << 15) - key - 1;
    key = key ^ (key >> 12);
    key = key + (key << 2);
    key = key ^ (key >> 4);
    key = key * 2057;  // key = (key + (key << 3)) + (key << 11);
    key = key ^ (key >> 16);
    return key;
  }

  __device__ inline uint64_t Hash64Shift(uint64_t key) {
    key = (~key) + (key << 21);  // key = (key << 21) - key - 1;
    key = key ^ (key >> 24);
    key = (key + (key << 3)) + (key << 8);  // key * 265
    key = key ^ (key >> 14);
    key = (key + (key << 2)) + (key << 4);  // key * 21
    key = key ^ (key >> 28);
    key = key + (key << 31);
    return key;
  }

  __device__ inline uint32_t hash(int32_t key) {
    return Hash32Shift(key) & (capacity - 1);
  }

  __device__ inline uint32_t hash(uint32_t key) {
    return Hash32Shift(key) & (capacity - 1);
  }

  __device__ inline uint32_t hash(int64_t key) {
    return static_cast<uint32_t>(Hash64Shift(key)) & (capacity - 1);
  }

  __device__ inline uint32_t hash(uint64_t key) {
    return static_cast<uint32_t>(Hash64Shift(key)) & (capacity - 1);
  }

  IdType kEmptyKey{-1};
  IdType *kptr;
  int32_t *vptr;
  uint32_t capacity{0};
};

inline int _UpPower(int key) {
  int ret = 1 << static_cast<uint32_t>(std::log2(key) + 1);
  return ret;
}

}  // namespace bifeat

#endif