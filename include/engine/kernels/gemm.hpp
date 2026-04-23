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
// Tile order: tile_M -> tile_K -> tile_N -> micro m -> micro n -> k.
// ENG-302: not yet implemented (serial tiled path).
void gemm_fp32_tiled(const Tensor& A, const Tensor& B, Tensor& C, TilingConfig cfg = {},
                     float alpha = 1.0f, float beta = 0.0f);

// Cache-blocked variant with OpenMP static scheduling across M-tiles (ENG-302).
//
// Thread model: the outer M-tile loop is distributed with
//   #pragma omp parallel for schedule(static)
// Each thread owns a disjoint strip of rows of C, so there are no write races.
// beta is applied to C before the parallel region so every thread only needs
// to accumulate (+=) into its portion.
//
// n_threads=0  — defer to OMP_NUM_THREADS (recommended for callers that manage
//                their own thread pool externally).
// n_threads>0  — call omp_set_num_threads() for this invocation; only compiled
//                when _OPENMP is defined.
//
// Without OpenMP the function falls back to a correct serial tiled execution.
//
// Numerical accuracy vs gemm_fp32_naive: max abs diff < 1e-4 for FP32 inputs
// in [-1, 1] at sizes up to 256x256 (see tests/unit/test_gemm.cpp).
void gemm_fp32_parallel(const Tensor& A, const Tensor& B, Tensor& C, TilingConfig cfg = {},
                        int n_threads = 0, float alpha = 1.0f, float beta = 0.0f);

// Tiled + OpenMP + AVX2 8x8 micro-kernel (x86) or NEON (ARM).
// ENG-304: not yet implemented.
void gemm_fp32_simd(const Tensor& A, const Tensor& B, Tensor& C, TilingConfig cfg = {},
                    int n_threads = 0, float alpha = 1.0f, float beta = 0.0f);

} // namespace ie::kernels
