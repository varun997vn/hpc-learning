#include "engine/kernels/gemm.hpp"

#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ie::kernels {

// Shape validation shared by all kernel variants.
static void check_gemm_shapes_parallel(const Tensor& A, const Tensor& B, const Tensor& C,
                                       int64_t& M, int64_t& K, int64_t& N) {
    if (A.shape().rank != 2 || B.shape().rank != 2 || C.shape().rank != 2)
        throw std::invalid_argument("gemm_fp32_parallel: all tensors must be rank 2");

    M = A.shape()[0];
    K = A.shape()[1];
    N = B.shape()[1];

    if (K != B.shape()[0])
        throw std::invalid_argument(
            "gemm_fp32_parallel: A columns must equal B rows (inner dimension mismatch)");

    if (C.shape()[0] != M || C.shape()[1] != N)
        throw std::invalid_argument(
            "gemm_fp32_parallel: C shape must be [M, N] = [A.rows, B.cols]");
}

void gemm_fp32_parallel(const Tensor& A, const Tensor& B, Tensor& C, TilingConfig cfg,
                        int n_threads, float alpha, float beta) {
    int64_t M = 0, K = 0, N = 0;
    check_gemm_shapes_parallel(A, B, C, M, K, N);

    const float* a = A.data<float>();
    const float* b = B.data<float>();
    float* c = C.data<float>();

    const int64_t mc = cfg.mc;
    const int64_t nc = cfg.nc;
    const int64_t kc = cfg.kc;

#ifdef _OPENMP
    // n_threads=0 means "let OpenMP pick from OMP_NUM_THREADS"; only override
    // when the caller explicitly requests a specific count.
    if (n_threads > 0)
        omp_set_num_threads(n_threads);
#else
    (void)n_threads;
#endif

    // Pre-scale C by beta before entering the parallel region.
    // Doing this outside the tiled loops avoids read-modify-write races when
    // multiple threads accumulate into the same output tile (not possible with
    // the row-tile parallelism below, but it also avoids each thread having to
    // load and scale C twice for any tile spanning a tile boundary).
    if (beta == 0.0f) {
        for (int64_t idx = 0; idx < M * N; ++idx)
            c[idx] = 0.0f;
    } else if (beta != 1.0f) {
        for (int64_t idx = 0; idx < M * N; ++idx)
            c[idx] *= beta;
    }
    // beta == 1.0f: no-op, C already holds its initial contribution.

    // Parallelize over M-tiles.  Each thread owns a contiguous strip of rows of
    // C, so there are no write-sharing races.  The k-tile and n-tile loops are
    // serial inside each thread.
    //
    // Loop order: tile_M (parallel) -> tile_K -> tile_N -> micro m -> micro n -> k
    //
    // After the pre-scaling pass above the inner accumulation always uses
    // beta=1 (i.e., plain addition into c), which is why the micro-kernel
    // below does c[i*N+j] += ... rather than c[i*N+j] = alpha*acc + beta*old.
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t i0 = 0; i0 < M; i0 += mc) {
        const int64_t i_end = (i0 + mc < M) ? i0 + mc : M;

        for (int64_t k0 = 0; k0 < K; k0 += kc) {
            const int64_t k_end = (k0 + kc < K) ? k0 + kc : K;

            for (int64_t j0 = 0; j0 < N; j0 += nc) {
                const int64_t j_end = (j0 + nc < N) ? j0 + nc : N;

                // Micro-kernel: accumulate the contribution of A[i0:i_end, k0:k_end]
                // * B[k0:k_end, j0:j_end] into C[i0:i_end, j0:j_end].
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
