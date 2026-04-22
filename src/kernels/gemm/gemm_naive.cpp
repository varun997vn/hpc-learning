#include "engine/kernels/gemm.hpp"

#include <stdexcept>

namespace ie::kernels {

// Validate that A[M,K] * B[K,N] -> C[M,N].
// All tensors must be rank-2.
static void check_gemm_shapes(const Tensor& A, const Tensor& B, const Tensor& C) {
    if (A.shape().rank != 2 || B.shape().rank != 2 || C.shape().rank != 2)
        throw std::invalid_argument("gemm_fp32_naive: all tensors must be rank 2");

    const int64_t M = A.shape()[0];
    const int64_t K = A.shape()[1];
    const int64_t K2 = B.shape()[0];
    const int64_t N = B.shape()[1];

    if (K != K2)
        throw std::invalid_argument(
            "gemm_fp32_naive: A columns must equal B rows (inner dimension mismatch)");

    if (C.shape()[0] != M || C.shape()[1] != N)
        throw std::invalid_argument("gemm_fp32_naive: C shape must be [M, N] = [A.rows, B.cols]");
}

void gemm_fp32_naive(const Tensor& A, const Tensor& B, Tensor& C, float alpha, float beta) {
    check_gemm_shapes(A, B, C);

    const int64_t M = A.shape()[0];
    const int64_t K = A.shape()[1];
    const int64_t N = B.shape()[1];

    const float* a = A.data<float>();
    const float* b = B.data<float>();
    float* c = C.data<float>();

    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int64_t k = 0; k < K; ++k)
                acc += a[i * K + k] * b[k * N + j];
            c[i * N + j] = alpha * acc + beta * c[i * N + j];
        }
    }
}

void gemm_fp32_tiled(const Tensor& /*A*/, const Tensor& /*B*/, Tensor& /*C*/, TilingConfig /*cfg*/,
                     float /*alpha*/, float /*beta*/) {
    throw std::runtime_error("gemm_fp32_tiled: not implemented");
}

void gemm_fp32_parallel(const Tensor& /*A*/, const Tensor& /*B*/, Tensor& /*C*/,
                        TilingConfig /*cfg*/, int /*n_threads*/, float /*alpha*/, float /*beta*/) {
    throw std::runtime_error("gemm_fp32_parallel: not implemented");
}

void gemm_fp32_simd(const Tensor& /*A*/, const Tensor& /*B*/, Tensor& /*C*/, TilingConfig /*cfg*/,
                    int /*n_threads*/, float /*alpha*/, float /*beta*/) {
    throw std::runtime_error("gemm_fp32_simd: not implemented");
}

} // namespace ie::kernels
