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

// ---- Scalar tiled fallback --------------------------------------------------
//
// Six-loop tiled GEMM, no SIMD.  Used as the fallback when __AVX2__ is not
// defined.  Produces numerically identical output to gemm_fp32_naive modulo
// floating-point reassociation (difference < 1e-4 for [-1,1] inputs).

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
                        const float a_ik = alpha * a[i * K + k];
                        for (int64_t j = j0; j < j0 + jb; ++j)
                            c[i * N + j] += a_ik * b[k * N + j];
                    }
                }
            }
        }
    }
}
#endif // !__AVX2__

// ---- AVX2 tiled 8×1 micro-kernel --------------------------------------------
//
// For each row i in the M-tile:
//   For each k in the K-tile:
//     Broadcast alpha*A[i,k] into a YMM register.
//     FMA with 8 consecutive B columns using _mm256_fmadd_ps.
//     Scalar tail for the residual columns not divisible by 8.
//
// Register usage per inner-k step:
//   - 1 YMM accumulator broadcast  (a_vec)
//   - 1 YMM load from B             (b_vec)
//   - 1 YMM load/store from C       (c_vec)
//   Total: 3 YMM — well within the 16-register budget.
//
// Threading: the outer M-tile loop is parallelised with OpenMP static
// scheduling.  Each thread owns a disjoint set of M-tile rows, so writes to
// C[i][j] never race.

#ifdef __AVX2__
// clang-format off
static void gemm_fp32_avx2(const float* __restrict__ a,
                            const float* __restrict__ b,
                            float* __restrict__ c,
                            int64_t M, int64_t N, int64_t K,
                            float alpha, float beta,
                            const TilingConfig& cfg,
                            int n_threads) {
    // Scale C by beta once before accumulation
    {
        const int64_t total = M * N;
        for (int64_t idx = 0; idx < total; ++idx)
            c[idx] *= beta;
    }

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
                const int64_t jb8 = (jb / 8) * 8; // number of full 8-wide steps

                for (int64_t i = i0; i < i0 + ib; ++i) {
                    const float* a_row = a + i * K;
                    float*       c_row = c + i * N;

                    for (int64_t k = k0; k < k0 + kb; ++k) {
                        // Fold alpha into the broadcast so the inner loop is
                        // purely: c[j] += a_scaled * b[k][j]
                        const float  a_s   = alpha * a_row[k];
                        const __m256 a_vec = _mm256_set1_ps(a_s);

                        // Vectorised: 8 columns per step
                        for (int64_t j = j0; j < j0 + jb8; j += 8) {
                            __m256 c_vec = _mm256_loadu_ps(c_row + j);
                            __m256 b_vec = _mm256_loadu_ps(b + k * N + j);
                            c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                            _mm256_storeu_ps(c_row + j, c_vec);
                        }

                        // Scalar tail: remaining columns in [j0+jb8, j0+jb)
                        for (int64_t j = j0 + jb8; j < j0 + jb; ++j)
                            c_row[j] += a_s * b[k * N + j];
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
