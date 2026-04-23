#include "engine/kernels/gemm.hpp"

#include <stdexcept>

namespace ie::kernels {

static void check_gemm_shapes(const Tensor& A, const Tensor& B, const Tensor& C, int64_t& M,
                              int64_t& K, int64_t& N) {
    if (A.shape().rank != 2 || B.shape().rank != 2 || C.shape().rank != 2)
        throw std::invalid_argument("gemm: all tensors must be rank 2");

    M = A.shape()[0];
    K = A.shape()[1];
    N = B.shape()[1];

    if (K != B.shape()[0])
        throw std::invalid_argument("gemm: A columns must equal B rows (inner dimension mismatch)");

    if (C.shape()[0] != M || C.shape()[1] != N)
        throw std::invalid_argument("gemm: C shape must be [M, N] = [A.rows, B.cols]");
}

void gemm_fp32_naive(const Tensor& A, const Tensor& B, Tensor& C, float alpha, float beta) {
    int64_t M = 0, K = 0, N = 0;
    check_gemm_shapes(A, B, C, M, K, N);

    const float* a = A.data<float>();
    const float* b = B.data<float>();
    float* c = C.data<float>();

    for (int64_t i = 0; i < M; ++i) {
        const float* a_row = a + i * K;
        float* c_row = c + i * N;
        for (int64_t j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int64_t k = 0; k < K; ++k)
                acc += a_row[k] * b[k * N + j];
            c_row[j] = alpha * acc + beta * c_row[j];
        }
    }
}

void gemm_fp32_simd(const Tensor& /*A*/, const Tensor& /*B*/, Tensor& /*C*/, TilingConfig /*cfg*/,
                    int /*n_threads*/, float /*alpha*/, float /*beta*/) {
    throw std::runtime_error("gemm_fp32_simd: not implemented");
}

} // namespace ie::kernels
