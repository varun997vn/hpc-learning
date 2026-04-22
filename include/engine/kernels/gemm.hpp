#pragma once
#include "engine/tensor.hpp"

namespace ie::kernels {

struct TilingConfig {
    int64_t mc = 64;
    int64_t nc = 64;
    int64_t kc = 64;
};

// C = alpha * A * B + beta * C  (row-major, FP32)
// A: [M, K]   B: [K, N]   C: [M, N]
// Throws std::invalid_argument on shape mismatch or non-2D inputs.
void gemm_fp32_naive(const Tensor& A, const Tensor& B, Tensor& C, float alpha = 1.0f,
                     float beta = 0.0f);

void gemm_fp32_tiled(const Tensor& A, const Tensor& B, Tensor& C, TilingConfig cfg = {},
                     float alpha = 1.0f, float beta = 0.0f);

void gemm_fp32_parallel(const Tensor& A, const Tensor& B, Tensor& C, TilingConfig cfg = {},
                        int n_threads = 0, float alpha = 1.0f, float beta = 0.0f);

void gemm_fp32_simd(const Tensor& A, const Tensor& B, Tensor& C, TilingConfig cfg = {},
                    int n_threads = 0, float alpha = 1.0f, float beta = 0.0f);

} // namespace ie::kernels
