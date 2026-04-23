#include "engine/kernels/gemm.hpp"
#include "engine/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <random>

using namespace ie;
using namespace ie::kernels;

// ---- Helpers ---------------------------------------------------------------

static Tensor make_random_fp32(int64_t rows, int64_t cols, uint32_t seed = 42) {
    auto t = Tensor::create(make_shape(rows, cols), DType::FP32);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    float* p = t.data<float>();
    for (int64_t i = 0; i < rows * cols; ++i)
        p[i] = dist(rng);
    return t;
}

static Tensor make_zeros_fp32(int64_t rows, int64_t cols) {
    auto t = Tensor::create(make_shape(rows, cols), DType::FP32);
    float* p = t.data<float>();
    for (int64_t i = 0; i < rows * cols; ++i)
        p[i] = 0.0f;
    return t;
}

static float max_abs_diff(const Tensor& a, const Tensor& b) {
    const int64_t n = a.numel();
    const float* pa = a.data<float>();
    const float* pb = b.data<float>();
    float mx = 0.0f;
    for (int64_t i = 0; i < n; ++i)
        mx = std::max(mx, std::abs(pa[i] - pb[i]));
    return mx;
}

// ---- gemm_fp32_simd tests --------------------------------------------------

// Correctness against naive at square sizes
TEST(GemmSimd, MatchesNaive_Square) {
    for (int sz : {64, 128, 256}) {
        auto A = make_random_fp32(sz, sz, 1);
        auto B = make_random_fp32(sz, sz, 2);
        auto C_ref = make_zeros_fp32(sz, sz);
        auto C_got = make_zeros_fp32(sz, sz);

        gemm_fp32_naive(A, B, C_ref);
        gemm_fp32_simd(A, B, C_got);

        EXPECT_LT(max_abs_diff(C_ref, C_got), 1e-3f)
            << "size=" << sz << " max_abs_diff=" << max_abs_diff(C_ref, C_got);
    }
}

// alpha / beta scaling
TEST(GemmSimd, AlphaBeta) {
    const int sz = 64;
    auto A = make_random_fp32(sz, sz, 10);
    auto B = make_random_fp32(sz, sz, 11);

    // Initialise C to a known non-zero value so beta term is exercised
    auto C_ref = make_random_fp32(sz, sz, 12);
    auto C_got = C_ref.clone();

    const float alpha = 0.5f;
    const float beta = 2.0f;

    gemm_fp32_naive(A, B, C_ref, alpha, beta);
    gemm_fp32_simd(A, B, C_got, {}, 1, alpha, beta);

    EXPECT_LT(max_abs_diff(C_ref, C_got), 1e-3f)
        << "alpha/beta test failed, max_abs_diff=" << max_abs_diff(C_ref, C_got);
}

// Non-square dimensions exercise different code paths
TEST(GemmSimd, NonSquare) {
    const int M = 37, K = 53, N = 71;
    auto A = make_random_fp32(M, K, 20);
    auto B = make_random_fp32(K, N, 21);
    auto C_ref = make_zeros_fp32(M, N);
    auto C_got = make_zeros_fp32(M, N);

    gemm_fp32_naive(A, B, C_ref);
    gemm_fp32_simd(A, B, C_got);

    EXPECT_LT(max_abs_diff(C_ref, C_got), 1e-3f)
        << "NonSquare 37x53x71 max_abs_diff=" << max_abs_diff(C_ref, C_got);
}

// Dimensions that are not multiples of 8 stress the scalar tail loop
TEST(GemmSimd, TailHandling) {
    for (int sz : {63, 65, 127, 129}) {
        auto A = make_random_fp32(sz, sz, static_cast<uint32_t>(sz));
        auto B = make_random_fp32(sz, sz, static_cast<uint32_t>(sz + 1));
        auto C_ref = make_zeros_fp32(sz, sz);
        auto C_got = make_zeros_fp32(sz, sz);

        gemm_fp32_naive(A, B, C_ref);
        gemm_fp32_simd(A, B, C_got);

        EXPECT_LT(max_abs_diff(C_ref, C_got), 1e-3f)
            << "TailHandling size=" << sz << " max_abs_diff=" << max_abs_diff(C_ref, C_got);
    }
}
