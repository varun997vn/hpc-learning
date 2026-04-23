#include "engine/kernels/gemm.hpp"

#include <stdexcept>

namespace ie {
namespace kernels {

// ENG-303: AVX2-accelerated FP32 GEMM — stub (Red phase)
void gemm_fp32_simd(const Tensor& A, const Tensor& B, Tensor& C, TilingConfig cfg, int n_threads,
                    float alpha, float beta) {
    (void)A;
    (void)B;
    (void)C;
    (void)cfg;
    (void)n_threads;
    (void)alpha;
    (void)beta;
    throw std::runtime_error("gemm_fp32_simd: not yet implemented");
}

} // namespace kernels
} // namespace ie
