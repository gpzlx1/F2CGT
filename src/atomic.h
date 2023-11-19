#ifndef PG_ATOMIC_H_
#define PG_ATOMIC_H_

#include "common.h"

namespace pg {
namespace atomic {

inline __device__ int64_t AtomicCAS(int64_t *const address,
                                    const int64_t compare, const int64_t val) {
  // match the type of "::atomicCAS", so ignore lint warning
  using Type = unsigned long long int;  // NOLINT

  static_assert(sizeof(Type) == sizeof(*address), "Type width must match");

  return atomicCAS(reinterpret_cast<Type *>(address),
                   static_cast<Type>(compare), static_cast<Type>(val));
}

inline __device__ uint64_t AtomicCAS(uint64_t *const address,
                                     const uint64_t compare,
                                     const uint64_t val) {
  // match the type of "::atomicCAS", so ignore lint warning
  using Type = unsigned long long int;  // NOLINT

  static_assert(sizeof(Type) == sizeof(*address), "Type width must match");

  return atomicCAS(reinterpret_cast<Type *>(address),
                   static_cast<Type>(compare), static_cast<Type>(val));
}

inline __device__ int32_t AtomicCAS(int32_t *const address,
                                    const int32_t compare, const int32_t val) {
  // match the type of "::atomicCAS", so ignore lint warning
  using Type = int;  // NOLINT

  static_assert(sizeof(Type) == sizeof(*address), "Type width must match");

  return atomicCAS(reinterpret_cast<Type *>(address),
                   static_cast<Type>(compare), static_cast<Type>(val));
}

inline __device__ uint32_t AtomicCAS(uint32_t *const address,
                                     const uint32_t compare,
                                     const uint32_t val) {
  // match the type of "::atomicCAS", so ignore lint warning
  using Type = int;  // NOLINT

  static_assert(sizeof(Type) == sizeof(*address), "Type width must match");

  return atomicCAS(reinterpret_cast<Type *>(address),
                   static_cast<Type>(compare), static_cast<Type>(val));
}
}  // namespace atomic
}  // namespace pg

#endif