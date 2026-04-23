#include "engine/kernels/gemm.hpp"

#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ie::kernels {

// ---- Internal helpers -------------------------------------------------------

static void check_shapes(const Tensor& A, const Tensor& B, const Tensor& C, int64_t& M, int64_t& K,
                         int64_t& N, const char* fn) {
    if (A.shape().rank != 2 || B.shape().rank != 2 || C.shape().rank != 2)
        throw std::invalid_argument(std::string(fn) + ": all tensors must be rank 2");

    M = A.shape()[0];
    K = A.shape()[1];
    N = B.shape()[1];

    if (K != B.shape()[0])
        throw std::invalid_argument(std::string(fn) +
                                    ": A columns must equal B rows (inner dimension mismatch)");
    if (C.shape()[0] != M || C.shape()[1] != N)
        throw std::invalid_argument(std::string(fn) +
                                    ": C shape must be [M, N] = [A.rows, B.cols]");
}

// ---- gemm_fp32_parallel -----------------------------------------------------

void gemm_fp32_parallel(const Tensor& A, const Tensor& B, Tensor& C, TilingConfig cfg,
                        int n_threads, float alpha, float beta) {
    int64_t M = 0, K = 0, N = 0;
    check_shapes(A, B, C, M, K, N, "gemm_fp32_parallel");

    const float* a = A.data<float>();
    const float* b = B.data<float>();
    float* c = C.data<float>();

    const int64_t mc = cfg.mc;
    const int64_t nc = cfg.nc;
    const int64_t kc = cfg.kc;

#ifdef _OPENMP
    // Only override the thread count when the caller requests a specific value.
    // n_threads=0 defers to OMP_NUM_THREADS (or the OpenMP runtime default).
    if (n_threads > 0)
        omp_set_num_threads(n_threads);
#else
    (void)n_threads;
#endif

    // Pre-scale C by beta outside the parallel region.  Each output element is
    // touched exactly once here, then only added to inside the tiled loops.
    // This pattern avoids read-modify-write races on the beta contribution when
    // multiple K-tiles accumulate into the same C element.
    if (beta == 0.0f) {
        for (int64_t idx = 0; idx < M * N; ++idx)
            c[idx] = 0.0f;
    } else if (beta != 1.0f) {
        for (int64_t idx = 0; idx < M * N; ++idx)
            c[idx] *= beta;
    }

    // Outer M-tile loop is distributed across threads with static scheduling.
    // Threads own non-overlapping row-strips of C; no synchronisation needed.
    //
    // Tile order: tile_M (parallel) -> tile_K -> tile_N -> micro m -> micro n -> k
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t i0 = 0; i0 < M; i0 += mc) {
        const int64_t i_end = (i0 + mc < M) ? i0 + mc : M;

        for (int64_t k0 = 0; k0 < K; k0 += kc) {
            const int64_t k_end = (k0 + kc < K) ? k0 + kc : K;

            for (int64_t j0 = 0; j0 < N; j0 += nc) {
                const int64_t j_end = (j0 + nc < N) ? j0 + nc : N;

                for (int64_t i = i0; i < i_end; ++i) {
                    for (int64_t j = j0; j < j_end; ++j) {
                        float acc = 0.0f;
                        for (int64_t k = k0; k < k_end; ++k)
                            acc += a[i * K + k] * b[k * N + j];
                        c[i * N + j] += alpha * acc;
                    }
                }
            }
        }
    }
}

} // namespace ie::kernels
