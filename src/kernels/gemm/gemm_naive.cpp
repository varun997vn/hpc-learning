#include "engine/kernels/gemm.hpp"

#include <stdexcept>

namespace ie {
namespace kernels {

namespace {
// Validates that A[M,K] * B[K,N] -> C[M,N] shapes are consistent.
// Throws std::invalid_argument on mismatch.
void check_gemm_shapes(const Tensor& A, const Tensor& B, const Tensor& C) {
    if (A.shape().rank != 2 || B.shape().rank != 2 || C.shape().rank != 2)
        throw std::invalid_argument("gemm: all tensors must be rank-2");
    const int64_t M = A.shape()[0];
    const int64_t K = A.shape()[1];
    const int64_t Kb = B.shape()[0];
    const int64_t N = B.shape()[1];
    const int64_t Mc = C.shape()[0];
    const int64_t Nc = C.shape()[1];
    if (K != Kb)
        throw std::invalid_argument("gemm: A columns != B rows");
    if (M != Mc || N != Nc)
        throw std::invalid_argument("gemm: C shape does not match M x N");
}
} // namespace

void gemm_fp32_naive(const Tensor& A, const Tensor& B, Tensor& C, float alpha, float beta) {
    check_gemm_shapes(A, B, C);

    const int64_t M = A.shape()[0];
    const int64_t K = A.shape()[1];
    const int64_t N = B.shape()[1];

    const float* a = A.data<float>();
    const float* b = B.data<float>();
    float* c = C.data<float>();

    // Pre-scale C by beta (single pass avoids a temp buffer).
    if (beta != 1.0f) {
        for (int64_t i = 0; i < M * N; ++i)
            c[i] *= beta;
    }

    // Accumulate alpha * A * B into C.
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t k = 0; k < K; ++k) {
            const float a_ik = alpha * a[i * K + k];
            for (int64_t j = 0; j < N; ++j)
                c[i * N + j] += a_ik * b[k * N + j];
        }
    }
}

void gemm_fp32_parallel(const Tensor& /*A*/, const Tensor& /*B*/, Tensor& /*C*/,
                        TilingConfig /*cfg*/, int /*n_threads*/, float /*alpha*/, float /*beta*/) {
    throw std::runtime_error("gemm_fp32_parallel: not yet implemented");
}

void gemm_fp32_simd(const Tensor& /*A*/, const Tensor& /*B*/, Tensor& /*C*/, TilingConfig /*cfg*/,
                    int /*n_threads*/, float /*alpha*/, float /*beta*/) {
    throw std::runtime_error("gemm_fp32_simd: not yet implemented");
}

void gemm_int8_fixed(const Tensor& /*A*/, const Tensor& /*B*/, Tensor& /*C*/, QuantParams /*qp*/) {
    throw std::runtime_error("gemm_int8_fixed: not yet implemented");
}

} // namespace kernels
} // namespace ie
