#include "engine/kernels/gemm.hpp"

#include <stdexcept>

namespace ie {
namespace kernels {

// Reference triple-loop GEMM: C = alpha * A * B + beta * C
// No optimizations — used as correctness oracle in tests.
void gemm_fp32_naive(const Tensor& A, const Tensor& B, Tensor& C, float alpha, float beta) {
    const int64_t M = A.shape()[0];
    const int64_t K = A.shape()[1];
    const int64_t N = B.shape()[1];

    if (B.shape()[0] != K)
        throw std::invalid_argument("gemm_fp32_naive: inner dimensions must match");
    if (C.shape()[0] != M || C.shape()[1] != N)
        throw std::invalid_argument("gemm_fp32_naive: C shape must be [M x N]");

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

// Stub — implemented in gemm_tiled.cpp (ENG-302)
void gemm_fp32_tiled(const Tensor& A, const Tensor& B, Tensor& C, TilingConfig cfg, float alpha,
                     float beta) {
    (void)cfg;
    gemm_fp32_naive(A, B, C, alpha, beta);
}

// Stub — implemented in gemm_parallel.cpp (ENG-302)
void gemm_fp32_parallel(const Tensor& A, const Tensor& B, Tensor& C, TilingConfig cfg,
                        int n_threads, float alpha, float beta) {
    (void)cfg;
    (void)n_threads;
    gemm_fp32_naive(A, B, C, alpha, beta);
}

// Stub — INT8 GEMM implemented in gemm_int8.cpp (ENG-404)
void gemm_int8_fixed(const Tensor& A, const Tensor& B, Tensor& C, QuantParams qp) {
    (void)A;
    (void)B;
    (void)C;
    (void)qp;
    throw std::runtime_error("gemm_int8_fixed: not yet implemented");
}

} // namespace kernels
} // namespace ie
