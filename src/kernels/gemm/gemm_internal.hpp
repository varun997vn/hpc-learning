#pragma once
#include "engine/tensor.hpp"

#include <stdexcept>

// Private header — not part of the public API.
// Shared helpers used by all gemm_fp32_* translation units.

namespace ie {
namespace kernels {
namespace detail {

// Validates A[M,K] * B[K,N] -> C[M,N] shapes.
// Throws std::invalid_argument with a descriptive message on mismatch.
inline void check_gemm_shapes(const char *func, const Tensor &A,
                              const Tensor &B, const Tensor &C) {
  if (A.shape().rank != 2 || B.shape().rank != 2 || C.shape().rank != 2)
    throw std::invalid_argument(std::string(func) +
                                ": all tensors must be rank-2");
  const int64_t K = A.shape()[1];
  const int64_t Kb = B.shape()[0];
  const int64_t M = A.shape()[0];
  const int64_t N = B.shape()[1];
  const int64_t Mc = C.shape()[0];
  const int64_t Nc = C.shape()[1];
  if (K != Kb)
    throw std::invalid_argument(std::string(func) + ": A columns != B rows");
  if (M != Mc || N != Nc)
    throw std::invalid_argument(std::string(func) +
                                ": C shape does not match M x N");
}

// Validates INT8 A[M,K] * B[K,N] → FP32 C[M,N] shapes and dtypes.
// Throws std::invalid_argument on violation.
inline void check_int8_gemm_shapes(const char *func, const Tensor &A,
                                   const Tensor &B, const Tensor &C) {
  if (A.shape().rank != 2 || B.shape().rank != 2 || C.shape().rank != 2)
    throw std::invalid_argument(std::string(func) +
                                ": all tensors must be rank-2");
  if (A.dtype() != DType::INT8 || B.dtype() != DType::INT8)
    throw std::invalid_argument(std::string(func) +
                                ": A and B must be DType::INT8");
  if (C.dtype() != DType::FP32)
    throw std::invalid_argument(std::string(func) + ": C must be DType::FP32");
  if (A.shape()[1] != B.shape()[0])
    throw std::invalid_argument(
        std::string(func) + ": inner dimension mismatch (A.cols != B.rows)");
  if (C.shape()[0] != A.shape()[0] || C.shape()[1] != B.shape()[1])
    throw std::invalid_argument(std::string(func) + ": C shape must be [M, N]");
}

} // namespace detail
} // namespace kernels
} // namespace ie
