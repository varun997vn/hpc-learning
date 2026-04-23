#pragma once
#include "engine/tensor.hpp"

#include <cstdint>

namespace ie {
namespace kernels {

struct TilingConfig {
    int mc = 64; // M-dimension tile size
    int nc = 64; // N-dimension tile size
    int kc = 64; // K-dimension tile size
};

struct QuantParams {
    float scale_a = 1.0f;
    float scale_b = 1.0f;
    float scale_c = 1.0f;
    int32_t zero_c = 0;
};

// C = alpha * A * B + beta * C
// A: [M x K], B: [K x N], C: [M x N]
void gemm_fp32_naive(const Tensor& A, const Tensor& B, Tensor& C, float alpha = 1.0f,
                     float beta = 0.0f);
void gemm_fp32_tiled(const Tensor& A, const Tensor& B, Tensor& C, TilingConfig cfg = {},
                     float alpha = 1.0f, float beta = 0.0f);
void gemm_fp32_parallel(const Tensor& A, const Tensor& B, Tensor& C, TilingConfig cfg = {},
                        int n_threads = 1, float alpha = 1.0f, float beta = 0.0f);
void gemm_fp32_simd(const Tensor& A, const Tensor& B, Tensor& C, TilingConfig cfg = {},
                    int n_threads = 1, float alpha = 1.0f, float beta = 0.0f);
void gemm_int8_fixed(const Tensor& A, const Tensor& B, Tensor& C, QuantParams qp = {});

} // namespace kernels
} // namespace ie
