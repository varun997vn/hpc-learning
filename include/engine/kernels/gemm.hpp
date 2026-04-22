#pragma once
#include "engine/tensor.hpp"

namespace ie::kernels {

// Cache-tiling parameters for the blocked GEMM variants.
// Default tile (mc=nc=kc=64) fits 3 × 64×64 × 4B = 48 KiB in L1 for FP32,
// leaving headroom for the L1 instruction cache on typical micro-architectures.
struct TilingConfig {
    int64_t mc = 64; // rows of A / C processed per outer tile
    int64_t nc = 64; // columns of B / C processed per outer tile
    int64_t kc = 64; // reduction dimension processed per outer tile
};

// Computes  C = alpha * A * B + beta * C  (row-major, FP32).
//
// Shape contract: A is [M, K], B is [K, N], C is [M, N].  All tensors must
// be rank-2 and DType::FP32.  Throws std::invalid_argument on violation.
//
// This is the reference implementation — no tiling, no SIMD.  Use it only
// for correctness checks and as the baseline in benchmarks.
void gemm_fp32_naive(const Tensor& A, const Tensor& B, Tensor& C, float alpha = 1.0f,
                     float beta = 0.0f);

// Cache-blocked variant.  Uses TilingConfig to fit working sets in L1/L2.
// ENG-302: not yet implemented.
void gemm_fp32_tiled(const Tensor& A, const Tensor& B, Tensor& C, TilingConfig cfg = {},
                     float alpha = 1.0f, float beta = 0.0f);

// Tiled + OpenMP static scheduling across the M-dimension.
// n_threads=0 lets OpenMP choose the thread count from OMP_NUM_THREADS.
// ENG-303: not yet implemented.
void gemm_fp32_parallel(const Tensor& A, const Tensor& B, Tensor& C, TilingConfig cfg = {},
                        int n_threads = 0, float alpha = 1.0f, float beta = 0.0f);

// Tiled + OpenMP + AVX2 8×8 micro-kernel (x86) or NEON (ARM).
// ENG-304: not yet implemented.
void gemm_fp32_simd(const Tensor& A, const Tensor& B, Tensor& C, TilingConfig cfg = {},
                    int n_threads = 0, float alpha = 1.0f, float beta = 0.0f);

} // namespace ie::kernels
