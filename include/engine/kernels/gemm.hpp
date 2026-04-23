#pragma once
#include "engine/tensor.hpp"

namespace ie {
namespace kernels {

// Default tile sizes chosen so that mc*kc + kc*nc + mc*nc FP32 elements
// fit in a 48 KB L1 cache (3 * 64*64 * 4 B = 48 KB).
struct TilingConfig {
    int mc = 64; // row tile
    int nc = 64; // column tile
    int kc = 64; // reduction tile
};

struct QuantParams {
    float scale_a = 1.0f;
    float scale_b = 1.0f;
    float scale_c = 1.0f;
    int32_t zero_point = 0;
};

// C = alpha * A * B + beta * C
// A: [M, K]  B: [K, N]  C: [M, N]
// All row-major; element [i][j] of [rows, cols] is at offset i*cols + j.

void gemm_fp32_naive(const Tensor& A, const Tensor& B, Tensor& C, float alpha = 1.0f,
                     float beta = 0.0f);

void gemm_fp32_tiled(const Tensor& A, const Tensor& B, Tensor& C, TilingConfig cfg = {},
                     float alpha = 1.0f, float beta = 0.0f);

void gemm_fp32_parallel(const Tensor& A, const Tensor& B, Tensor& C, TilingConfig cfg = {},
                        int n_threads = 1, float alpha = 1.0f, float beta = 0.0f);

void gemm_fp32_simd(const Tensor& A, const Tensor& B, Tensor& C, TilingConfig cfg = {},
                    int n_threads = 1, float alpha = 1.0f, float beta = 0.0f);

void gemm_int8_fixed(const Tensor& A, const Tensor& B, Tensor& C, QuantParams qp);

} // namespace kernels
} // namespace ie
