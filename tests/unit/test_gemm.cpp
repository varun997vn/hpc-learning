#include "engine/kernels/gemm.hpp"
#include "engine/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <random>

using namespace ie;
using namespace ie::kernels;

// ---- Helpers ----------------------------------------------------------------

static Tensor make_fp32(int rows, int cols) {
    return Tensor::create(make_shape(rows, cols), DType::FP32);
}

// Fill with uniform random floats using a fixed seed for reproducibility.
static void fill_random(Tensor& t, float lo = -1.0f, float hi = 1.0f, uint32_t seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    float* p = t.data<float>();
    for (int64_t i = 0; i < t.numel(); ++i)
        p[i] = dist(rng);
}

static float max_abs_diff(const Tensor& a, const Tensor& b) {
    const float* pa = a.data<float>();
    const float* pb = b.data<float>();
    float d = 0.0f;
    for (int64_t i = 0; i < a.numel(); ++i)
        d = std::max(d, std::abs(pa[i] - pb[i]));
    return d;
}

// ---- gemm_fp32_naive tests --------------------------------------------------

TEST(GemmNaive, TwoByTwo) {
    // A = [[1,2],[3,4]]  B = [[5,6],[7,8]]
    // C = A*B = [[19,22],[43,50]]
    auto A = make_fp32(2, 2);
    auto B = make_fp32(2, 2);
    auto C = make_fp32(2, 2);

    float a_data[] = {1, 2, 3, 4};
    float b_data[] = {5, 6, 7, 8};
    std::copy(a_data, a_data + 4, A.data<float>());
    std::copy(b_data, b_data + 4, B.data<float>());
    std::fill_n(C.data<float>(), 4, 0.0f);

    gemm_fp32_naive(A, B, C);

    EXPECT_FLOAT_EQ(C.data<float>()[0], 19.0f);
    EXPECT_FLOAT_EQ(C.data<float>()[1], 22.0f);
    EXPECT_FLOAT_EQ(C.data<float>()[2], 43.0f);
    EXPECT_FLOAT_EQ(C.data<float>()[3], 50.0f);
}

TEST(GemmNaive, AlphaScales) {
    auto A = make_fp32(2, 2);
    auto B = make_fp32(2, 2);
    auto C = make_fp32(2, 2);

    float a_data[] = {1, 0, 0, 1};
    float b_data[] = {3, 0, 0, 3};
    std::copy(a_data, a_data + 4, A.data<float>());
    std::copy(b_data, b_data + 4, B.data<float>());
    std::fill_n(C.data<float>(), 4, 0.0f);

    gemm_fp32_naive(A, B, C, 2.0f, 0.0f);

    EXPECT_FLOAT_EQ(C.data<float>()[0], 6.0f);
    EXPECT_FLOAT_EQ(C.data<float>()[3], 6.0f);
}

TEST(GemmNaive, BetaAccumulates) {
    auto A = make_fp32(2, 2);
    auto B = make_fp32(2, 2);
    auto C = make_fp32(2, 2);

    float a_data[] = {1, 0, 0, 1};
    float b_data[] = {1, 0, 0, 1};
    std::copy(a_data, a_data + 4, A.data<float>());
    std::copy(b_data, b_data + 4, B.data<float>());
    C.data<float>()[0] = 10.0f;
    C.data<float>()[3] = 10.0f;
    C.data<float>()[1] = 0.0f;
    C.data<float>()[2] = 0.0f;

    // C = 1*(I) + 0.5*C_old → [1+5, 0, 0, 1+5]
    gemm_fp32_naive(A, B, C, 1.0f, 0.5f);

    EXPECT_FLOAT_EQ(C.data<float>()[0], 6.0f);
    EXPECT_FLOAT_EQ(C.data<float>()[3], 6.0f);
}

// ---- gemm_fp32_tiled tests --------------------------------------------------

TEST(GemmTiled, MatchesNaive_2x2) {
    auto A = make_fp32(2, 2);
    auto B = make_fp32(2, 2);
    auto C_naive = make_fp32(2, 2);
    auto C_tiled = make_fp32(2, 2);

    float a_data[] = {1, 2, 3, 4};
    float b_data[] = {5, 6, 7, 8};
    std::copy(a_data, a_data + 4, A.data<float>());
    std::copy(b_data, b_data + 4, B.data<float>());
    std::fill_n(C_naive.data<float>(), 4, 0.0f);
    std::fill_n(C_tiled.data<float>(), 4, 0.0f);

    gemm_fp32_naive(A, B, C_naive);
    gemm_fp32_tiled(A, B, C_tiled);

    EXPECT_FLOAT_EQ(max_abs_diff(C_naive, C_tiled), 0.0f);
}

class GemmTiledSquare : public ::testing::TestWithParam<int> {};

TEST_P(GemmTiledSquare, MatchesNaive) {
    const int N = GetParam();
    auto A = make_fp32(N, N);
    auto B = make_fp32(N, N);
    auto C_naive = make_fp32(N, N);
    auto C_tiled = make_fp32(N, N);

    fill_random(A, -1.0f, 1.0f, 1);
    fill_random(B, -1.0f, 1.0f, 2);
    std::fill_n(C_naive.data<float>(), N * N, 0.0f);
    std::fill_n(C_tiled.data<float>(), N * N, 0.0f);

    gemm_fp32_naive(A, B, C_naive);
    gemm_fp32_tiled(A, B, C_tiled);

    EXPECT_LT(max_abs_diff(C_naive, C_tiled), 1e-4f) << "max diff exceeded at N=" << N;
}

INSTANTIATE_TEST_SUITE_P(Sizes, GemmTiledSquare, ::testing::Values(64, 128, 256));

TEST(GemmTiled, AlphaBeta) {
    const int N = 32;
    auto A = make_fp32(N, N);
    auto B = make_fp32(N, N);
    auto C_init = make_fp32(N, N);
    auto C_naive = make_fp32(N, N);
    auto C_tiled = make_fp32(N, N);

    fill_random(A, -1.0f, 1.0f, 3);
    fill_random(B, -1.0f, 1.0f, 4);
    fill_random(C_init, -1.0f, 1.0f, 5);

    // Copy same initial C into both outputs.
    std::copy_n(C_init.data<float>(), N * N, C_naive.data<float>());
    std::copy_n(C_init.data<float>(), N * N, C_tiled.data<float>());

    const float alpha = 2.0f;
    const float beta = 0.5f;

    gemm_fp32_naive(A, B, C_naive, alpha, beta);
    gemm_fp32_tiled(A, B, C_tiled, {}, alpha, beta);

    EXPECT_LT(max_abs_diff(C_naive, C_tiled), 1e-4f)
        << "alpha/beta mismatch between naive and tiled";
}

TEST(GemmTiled, NonSquare_PrimeSizes) {
    // 37 × 71 × 53 — hits boundary conditions in all three tile loops
    const int M = 37, K = 71, N = 53;
    auto A = make_fp32(M, K);
    auto B = make_fp32(K, N);
    auto C_naive = make_fp32(M, N);
    auto C_tiled = make_fp32(M, N);

    fill_random(A, -1.0f, 1.0f, 10);
    fill_random(B, -1.0f, 1.0f, 11);
    std::fill_n(C_naive.data<float>(), M * N, 0.0f);
    std::fill_n(C_tiled.data<float>(), M * N, 0.0f);

    gemm_fp32_naive(A, B, C_naive);
    gemm_fp32_tiled(A, B, C_tiled);

    EXPECT_LT(max_abs_diff(C_naive, C_tiled), 1e-4f) << "non-square prime-size mismatch";
}

TEST(GemmTiled, CustomTiling) {
    // Use a smaller tile config to exercise the tile boundary logic independently.
    const int N = 256;
    auto A = make_fp32(N, N);
    auto B = make_fp32(N, N);
    auto C_naive = make_fp32(N, N);
    auto C_tiled = make_fp32(N, N);

    fill_random(A, -1.0f, 1.0f, 20);
    fill_random(B, -1.0f, 1.0f, 21);
    std::fill_n(C_naive.data<float>(), N * N, 0.0f);
    std::fill_n(C_tiled.data<float>(), N * N, 0.0f);

    gemm_fp32_naive(A, B, C_naive);
    gemm_fp32_tiled(A, B, C_tiled, TilingConfig{32, 32, 32});

    EXPECT_LT(max_abs_diff(C_naive, C_tiled), 1e-4f) << "custom tiling config {32,32,32} mismatch";
}
