#ifndef BIFEAT_HASHMAP_V2_H
#define BIFEAT_HASHMAP_V2_H

#include <pybind11/pybind11.h>
#include <torch/script.h>
// #include <bcht.hpp>

namespace bifeat {
namespace hashmap {

// key int64_t, value int32_t
class BiFeatHashmaps {
 public:
  BiFeatHashmaps(int64_t hashmap_num, std::vector<torch::Tensor> cache_nids);
  ~BiFeatHashmaps();
  torch::Tensor query(torch::Tensor keys, int64_t first_part_size);

  // private:
  // bght::bcht<int64_t, int32_t> hashmap1_;
  // bght::bcht<int64_t, int32_t> hashmap2_;

 private:
  int64_t memory_usage_;
  int64_t hashmap_num_;
  void* hashmap1_;
  void* hashmap2_;
};

}  // namespace hashmap

}  // namespace bifeat

#endif  // BIFEAT_HASHMAP_V2_H
