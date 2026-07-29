// Correctness harness: all four GEMM variants vs gemm_fp32_naive at large sizes.
// Tier: nightly (not run on every commit — slow). See CLAUDE.md testing tiers.
// Tolerance: 1e-4 for tiled/parallel (exact same arithmetic), 1e-3 for SIMD
// (FMA reassociation can shift results by a few ULPs).
#include "engine/kernels/gemm.hpp"
#include "engine/tensor.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <random>

using namespace ie;
using namespace ie::kernels;

static Tensor rand_fp32(int64_t r, int64_t c, uint32_t seed) {
    auto t = Tensor::create(make_shape(r, c), DType::FP32);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    float* p = t.data<float>();
    for (int64_t i = 0; i < r * c; ++i)
        p[i] = dist(rng);
    return t;
}

static float max_abs_diff(const Tensor& a, const Tensor& b) {
    float mx = 0.0f;
    const float* pa = a.data<float>();
    const float* pb = b.data<float>();
    for (int64_t i = 0; i < a.numel(); ++i)
        mx = std::max(mx, std::abs(pa[i] - pb[i]));
    return mx;
}

static void run_naive(const Tensor& A, const Tensor& B, Tensor& C) {
    gemm_fp32_naive(A, B, C, 1.0f, 0.0f);
}

// ---- Tiled ------------------------------------------------------------------

class GemmCorrectnessTiled : public ::testing::TestWithParam<int64_t> {};

TEST_P(GemmCorrectnessTiled, MatchesNaive) {
    const int64_t N = GetParam();
    auto A = rand_fp32(N, N, 1);
    auto B = rand_fp32(N, N, 2);
    auto C_ref = Tensor::create(make_shape(N, N), DType::FP32);
    auto C_tiled = Tensor::create(make_shape(N, N), DType::FP32);
    run_naive(A, B, C_ref);
    gemm_fp32_tiled(A, B, C_tiled, {}, 1.0f, 0.0f);
    EXPECT_LT(max_abs_diff(C_ref, C_tiled), 1e-4f) << "N=" << N;
}

INSTANTIATE_TEST_SUITE_P(Sizes, GemmCorrectnessTiled,
                         ::testing::Values(256, 512, 1024));

// ---- Parallel ---------------------------------------------------------------

class GemmCorrectnessParallel : public ::testing::TestWithParam<int64_t> {};

TEST_P(GemmCorrectnessParallel, MatchesNaive) {
    const int64_t N = GetParam();
    auto A = rand_fp32(N, N, 3);
    auto B = rand_fp32(N, N, 4);
    auto C_ref = Tensor::create(make_shape(N, N), DType::FP32);
    auto C_par = Tensor::create(make_shape(N, N), DType::FP32);
    run_naive(A, B, C_ref);
    gemm_fp32_parallel(A, B, C_par, {}, 4, 1.0f, 0.0f);
    EXPECT_LT(max_abs_diff(C_ref, C_par), 1e-4f) << "N=" << N;
}

INSTANTIATE_TEST_SUITE_P(Sizes, GemmCorrectnessParallel,
                         ::testing::Values(256, 512));

// ---- SIMD -------------------------------------------------------------------

class GemmCorrectnessSimd : public ::testing::TestWithParam<int64_t> {};

TEST_P(GemmCorrectnessSimd, MatchesNaive) {
    const int64_t N = GetParam();
    auto A = rand_fp32(N, N, 5);
    auto B = rand_fp32(N, N, 6);
    auto C_ref = Tensor::create(make_shape(N, N), DType::FP32);
    auto C_simd = Tensor::create(make_shape(N, N), DType::FP32);
    run_naive(A, B, C_ref);
    gemm_fp32_simd(A, B, C_simd, {}, 1, 1.0f, 0.0f);
    // Wider tolerance for SIMD: FMA reassociation shifts results by ~1e-5 per op.
    EXPECT_LT(max_abs_diff(C_ref, C_simd), 1e-3f) << "N=" << N;
}

INSTANTIATE_TEST_SUITE_P(Sizes, GemmCorrectnessSimd,
                         ::testing::Values(256, 511, 512, 513));
