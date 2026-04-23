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
inline void check_gemm_shapes(const char* func, const Tensor& A, const Tensor& B, const Tensor& C) {
    if (A.shape().rank != 2 || B.shape().rank != 2 || C.shape().rank != 2)
        throw std::invalid_argument(std::string(func) + ": all tensors must be rank-2");
    const int64_t K = A.shape()[1];
    const int64_t Kb = B.shape()[0];
    const int64_t M = A.shape()[0];
    const int64_t N = B.shape()[1];
    const int64_t Mc = C.shape()[0];
    const int64_t Nc = C.shape()[1];
    if (K != Kb)
        throw std::invalid_argument(std::string(func) + ": A columns != B rows");
    if (M != Mc || N != Nc)
        throw std::invalid_argument(std::string(func) + ": C shape does not match M x N");
}

} // namespace detail
} // namespace kernels
} // namespace ie
