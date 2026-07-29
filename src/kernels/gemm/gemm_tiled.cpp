#include "engine/kernels/gemm.hpp"
#include "gemm_internal.hpp"

#include <algorithm>
#include <cstring>

namespace ie {
namespace kernels {

// 6-loop cache-blocked GEMM: C = alpha * A * B + beta * C
//
// Tile order: tile_M -> tile_N -> tile_K -> inner_i -> inner_k -> inner_j
// This ordering keeps the B panel (kc x nc) resident across the inner loops
// and reuses the A strip (mc x kc) for each nc column tile.
//
// Pre-scaling C by beta in a single pass before the tile loops avoids a
// temporary buffer while keeping the inner loops free of beta arithmetic.
void gemm_fp32_tiled(const Tensor& A, const Tensor& B, Tensor& C, TilingConfig cfg, float alpha,
                     float beta) {
    detail::check_gemm_shapes("gemm_fp32_tiled", A, B, C);

    const int64_t M = A.shape()[0];
    const int64_t K = A.shape()[1];
    const int64_t N = B.shape()[1];

    const float* a = A.data<float>();
    const float* b = B.data<float>();
    float* c = C.data<float>();

    const int mc = cfg.mc;
    const int nc = cfg.nc;
    const int kc = cfg.kc;

    // Distance (in rows) to prefetch ahead. 4 rows × 64 cols × 4B = 1 KB,
    // enough to hide ~100-cycle DRAM latency at typical FP32 bandwidths.
    static constexpr int64_t kPrefetchDist = 4;

    // Pre-scale C by beta so the accumulation loop only needs += alpha*a*b.
    if (beta == 0.0f) {
        std::memset(c, 0, static_cast<size_t>(M * N) * sizeof(float));
    } else if (beta != 1.0f) {
        for (int64_t idx = 0; idx < M * N; ++idx)
            c[idx] *= beta;
    }

    // Outer tile loops.
    for (int64_t i0 = 0; i0 < M; i0 += mc) {
        const int64_t i_end = std::min(i0 + static_cast<int64_t>(mc), M);
        for (int64_t j0 = 0; j0 < N; j0 += nc) {
            const int64_t j_end = std::min(j0 + static_cast<int64_t>(nc), N);
            for (int64_t k0 = 0; k0 < K; k0 += kc) {
                const int64_t k_end = std::min(k0 + static_cast<int64_t>(kc), K);

                // Inner micro-tile: accumulate alpha * A[i0..i_end, k0..k_end]
                //                               * B[k0..k_end, j0..j_end]
                //                    into C[i0..i_end, j0..j_end]
                for (int64_t i = i0; i < i_end; ++i) {
#ifdef __GNUC__
                    // Prefetch next A row into L1 while current row computes.
                    if (i + kPrefetchDist < M)
                        __builtin_prefetch(&a[(i + kPrefetchDist) * K + k0], 0, 1);
#endif
                    for (int64_t k = k0; k < k_end; ++k) {
                        const float a_ik = alpha * a[i * K + k];
#ifdef __GNUC__
                        // Prefetch next B row into L1.
                        if (k + kPrefetchDist < K)
                            __builtin_prefetch(&b[(k + kPrefetchDist) * N + j0], 0, 1);
#endif
                        for (int64_t j = j0; j < j_end; ++j)
                            c[i * N + j] += a_ik * b[k * N + j];
                    }
                }
            }
        }
    }
}

} // namespace kernels
} // namespace ie
