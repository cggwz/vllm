#pragma once

// Simple device-friendly reduction functors to avoid relying on CUB-provided
// operator types (e.g. cub::Sum/cub::Max), which may not exist in newer CCCL/CUB
// releases bundled with recent CUDA toolkits.

namespace vllm {

struct ReduceSum {
  template <typename T>
  __host__ __device__ constexpr T operator()(const T& a, const T& b) const {
    return a + b;
  }
};

struct ReduceMax {
  template <typename T>
  __host__ __device__ constexpr T operator()(const T& a, const T& b) const {
    return a > b ? a : b;
  }
};

}  // namespace vllm

