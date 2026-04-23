#include "engine/kernels/gemm.hpp"

#include <algorithm>
#include <stdexcept>

#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ie {
namespace kernels {

// ---- Shared tail helper -----------------------------------------------------
//
// Accumulates a single scalar update: c_row[j] += a_scaled * b[k * N + j]
// for j in [j_start, j_end).  Inlined to allow the compiler to hoist
// invariants; used by both the scalar fallback and the AVX2 path to handle
// columns not divisible by 8.
static inline void scalar_tail(float* __restrict__ c_row, const float* __restrict__ b_k,
                               float a_scaled, int64_t j_start, int64_t j_end) {
    for (int64_t j = j_start; j < j_end; ++j)
        c_row[j] += a_scaled * b_k[j];
}

// ---- Scalar tiled fallback --------------------------------------------------
//
// Six-loop tiled GEMM with no SIMD.  Compiled only when __AVX2__ is absent so
// that the unused-function warning (-Werror) is not triggered on AVX2 builds.

#ifndef __AVX2__
static void gemm_fp32_tiled_scalar(const float* __restrict__ a, const float* __restrict__ b,
                                   float* __restrict__ c, int64_t M, int64_t N, int64_t K,
                                   float alpha, float beta, const TilingConfig& cfg) {
    for (int64_t idx = 0; idx < M * N; ++idx)
        c[idx] *= beta;

    const int64_t mc = cfg.mc;
    const int64_t nc = cfg.nc;
    const int64_t kc = cfg.kc;

    for (int64_t i0 = 0; i0 < M; i0 += mc) {
        const int64_t ib = std::min(mc, M - i0);
        for (int64_t k0 = 0; k0 < K; k0 += kc) {
            const int64_t kb = std::min(kc, K - k0);
            for (int64_t j0 = 0; j0 < N; j0 += nc) {
                const int64_t jb = std::min(nc, N - j0);
                for (int64_t i = i0; i < i0 + ib; ++i) {
                    for (int64_t k = k0; k < k0 + kb; ++k) {
                        const float a_s = alpha * a[i * K + k];
                        scalar_tail(c + i * N, b + k * N, a_s, j0, j0 + jb);
                    }
                }
            }
        }
    }
}
#endif // !__AVX2__

// ---- AVX2 tiled 8x1 micro-kernel -------------------------------------------
//
// Tile structure: mc x kc x nc outer loops, then per-row per-k inner step.
//
// Per inner-k step register usage:
//   a_vec  — YMM broadcast of alpha * A[i,k]         (1 register)
//   b_vec  — YMM load of 8 consecutive B columns      (1 register)
//   c_vec  — YMM load/store of C[i, j..j+7]           (1 register)
//   Total  — 3 YMM, well within the 16-register limit.
//
// Tail: scalar_tail() handles the residual < 8 columns at the right edge of
// each N-tile, sharing the same helper as the scalar fallback path.
//
// Threading: #pragma omp parallel for on the outer M-tile loop with static
// scheduling.  Each thread owns a disjoint M-tile stripe so C writes
// never race.

#ifdef __AVX2__
// clang-format off
static void gemm_fp32_avx2(const float* __restrict__ a,
                            const float* __restrict__ b,
                            float* __restrict__ c,
                            int64_t M, int64_t N, int64_t K,
                            float alpha, float beta,
                            const TilingConfig& cfg,
                            int n_threads) {
    const int64_t total = M * N;
    for (int64_t idx = 0; idx < total; ++idx)
        c[idx] *= beta;

    const int64_t mc = cfg.mc;
    const int64_t nc = cfg.nc;
    const int64_t kc = cfg.kc;

#ifdef _OPENMP
    if (n_threads > 1)
        omp_set_num_threads(n_threads);
#pragma omp parallel for schedule(static)
#endif
    for (int64_t i0 = 0; i0 < M; i0 += mc) {
        const int64_t ib = std::min(mc, M - i0);

        for (int64_t k0 = 0; k0 < K; k0 += kc) {
            const int64_t kb = std::min(kc, K - k0);

            for (int64_t j0 = 0; j0 < N; j0 += nc) {
                const int64_t jb  = std::min(nc, N - j0);
                const int64_t jb8 = (jb / 8) * 8; // full 8-wide steps

                for (int64_t i = i0; i < i0 + ib; ++i) {
                    const float* a_row = a + i * K;
                    float*       c_row = c + i * N;

                    // Prefetch the C row for the next i iteration into L1 so
                    // the read-modify-write in the k loop hits cache.
                    if (i + 1 < i0 + ib)
                        __builtin_prefetch(c + (i + 1) * N + j0, 1, 3);

                    for (int64_t k = k0; k < k0 + kb; ++k) {
                        // Fold alpha into the broadcast — inner loop is:
                        //   c[j] += a_scaled * b[k][j]
                        const float  a_s   = alpha * a_row[k];
                        const __m256 a_vec = _mm256_set1_ps(a_s);

                        // Prefetch the B row 8 k-steps ahead into L2 to hide
                        // the stride-N access latency before it enters the FMA loop.
                        constexpr int64_t kPrefDist = 8;
                        if (k + kPrefDist < k0 + kb)
                            __builtin_prefetch(b + (k + kPrefDist) * N + j0, 0, 1);

                        for (int64_t j = j0; j < j0 + jb8; j += 8) {
                            __m256 c_vec = _mm256_loadu_ps(c_row + j);
                            __m256 b_vec = _mm256_loadu_ps(b + k * N + j);
                            c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                            _mm256_storeu_ps(c_row + j, c_vec);
                        }

                        // Scalar tail shared with non-AVX2 path
                        scalar_tail(c_row, b + k * N, a_s, j0 + jb8, j0 + jb);
                    }
                }
            }
        }
    }
}
// clang-format on
#endif // __AVX2__

// ---- Public entry point -----------------------------------------------------

void gemm_fp32_simd(const Tensor& A, const Tensor& B, Tensor& C, TilingConfig cfg, int n_threads,
                    float alpha, float beta) {
    const int64_t M = A.shape()[0];
    const int64_t K = A.shape()[1];
    const int64_t N = B.shape()[1];

    if (B.shape()[0] != K)
        throw std::invalid_argument("gemm_fp32_simd: inner dimensions must match");
    if (C.shape()[0] != M || C.shape()[1] != N)
        throw std::invalid_argument("gemm_fp32_simd: C shape must be [M x N]");

    const float* a = A.data<float>();
    const float* b = B.data<float>();
    float* c = C.data<float>();

#ifdef __AVX2__
    gemm_fp32_avx2(a, b, c, M, N, K, alpha, beta, cfg, n_threads);
#else
    (void)n_threads;
    gemm_fp32_tiled_scalar(a, b, c, M, N, K, alpha, beta, cfg);
#endif
}

} // namespace kernels
} // namespace ie
