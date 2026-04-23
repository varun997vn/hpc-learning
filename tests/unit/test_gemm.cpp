// Unit tests for gemm_fp32_naive and gemm_fp32_parallel.
// Reference values for 2x2 and non-square cases computed by hand; no kernel
// logic is duplicated here.
#include "engine/kernels/gemm.hpp"
#include "engine/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <random>
#include <stdexcept>
#include <thread>

using namespace ie;
using namespace ie::kernels;

// ---- Helpers ----------------------------------------------------------------

static Tensor make_fp32(std::initializer_list<int64_t> dims, std::initializer_list<float> vals) {
    Shape s;
    s.rank = static_cast<int>(dims.size());
    int i = 0;
    for (auto d : dims)
        s.dims[static_cast<size_t>(i++)] = d;

    auto t = Tensor::create(s, DType::FP32);
    float* p = t.data<float>();
    i = 0;
    for (auto v : vals)
        p[i++] = v;
    return t;
}

static void fill_fp32(Tensor& t, float val) {
    float* p = t.data<float>();
    for (int64_t i = 0; i < t.numel(); ++i)
        p[i] = val;
}

// Fills tensor with values from mt19937 seeded at 42.
static void fill_random(Tensor& t, uint32_t seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    float* p = t.data<float>();
    for (int64_t i = 0; i < t.numel(); ++i)
        p[i] = dist(rng);
}

static float max_abs_diff(const Tensor& a, const Tensor& b) {
    const float* pa = a.data<float>();
    const float* pb = b.data<float>();
    float worst = 0.0f;
    for (int64_t i = 0; i < a.numel(); ++i)
        worst = std::max(worst, std::abs(pa[i] - pb[i]));
    return worst;
}

// ---- GemmNaive: 1x1 identity multiply ----------------------------------------

TEST(GemmNaive, OneByOne) {
    auto A = make_fp32({1, 1}, {2.0f});
    auto B = make_fp32({1, 1}, {3.0f});
    auto C = Tensor::create(make_shape(1, 1), DType::FP32);
    fill_fp32(C, 0.0f);

    gemm_fp32_naive(A, B, C);

    EXPECT_FLOAT_EQ(C.data<float>()[0], 6.0f);
}

// ---- GemmNaive: 2x2 known result ---------------------------------------------

TEST(GemmNaive, TwoByTwo) {
    auto A = make_fp32({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    auto B = make_fp32({2, 2}, {5.0f, 6.0f, 7.0f, 8.0f});
    auto C = Tensor::create(make_shape(2, 2), DType::FP32);
    fill_fp32(C, 0.0f);

    gemm_fp32_naive(A, B, C);

    const float* c = C.data<float>();
    EXPECT_FLOAT_EQ(c[0], 19.0f);
    EXPECT_FLOAT_EQ(c[1], 22.0f);
    EXPECT_FLOAT_EQ(c[2], 43.0f);
    EXPECT_FLOAT_EQ(c[3], 50.0f);
}

// ---- GemmNaive: 2x3 x 3x4 = 2x4 ---------------------------------------------

TEST(GemmNaive, NonSquare2x3Times3x4) {
    auto A = make_fp32({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto B = make_fp32(
        {3, 4}, {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f});
    auto C = Tensor::create(make_shape(2, 4), DType::FP32);
    fill_fp32(C, 0.0f);

    gemm_fp32_naive(A, B, C);

    const float* c = C.data<float>();
    EXPECT_FLOAT_EQ(c[0], 74.0f);
    EXPECT_FLOAT_EQ(c[1], 80.0f);
    EXPECT_FLOAT_EQ(c[2], 86.0f);
    EXPECT_FLOAT_EQ(c[3], 92.0f);
    EXPECT_FLOAT_EQ(c[4], 173.0f);
    EXPECT_FLOAT_EQ(c[5], 188.0f);
    EXPECT_FLOAT_EQ(c[6], 203.0f);
    EXPECT_FLOAT_EQ(c[7], 218.0f);
}

// ---- GemmNaive: alpha/beta ---------------------------------------------------

TEST(GemmNaive, AlphaScaling) {
    auto A = make_fp32({2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});
    auto B = make_fp32({2, 2}, {3.0f, 4.0f, 5.0f, 6.0f});
    auto C = Tensor::create(make_shape(2, 2), DType::FP32);
    fill_fp32(C, 0.0f);

    gemm_fp32_naive(A, B, C, 2.0f, 0.0f);

    const float* c = C.data<float>();
    EXPECT_FLOAT_EQ(c[0], 6.0f);
    EXPECT_FLOAT_EQ(c[1], 8.0f);
    EXPECT_FLOAT_EQ(c[2], 10.0f);
    EXPECT_FLOAT_EQ(c[3], 12.0f);
}

TEST(GemmNaive, BetaAccumulate) {
    auto A = make_fp32({2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});
    auto B = make_fp32({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    auto C = Tensor::create(make_shape(2, 2), DType::FP32);
    float* c = C.data<float>();
    c[0] = 10.0f;
    c[1] = 20.0f;
    c[2] = 30.0f;
    c[3] = 40.0f;

    gemm_fp32_naive(A, B, C, 1.0f, 1.0f);

    EXPECT_FLOAT_EQ(c[0], 11.0f);
    EXPECT_FLOAT_EQ(c[1], 22.0f);
    EXPECT_FLOAT_EQ(c[2], 33.0f);
    EXPECT_FLOAT_EQ(c[3], 44.0f);
}

TEST(GemmNaive, BetaZeroWritesOverC) {
    auto A = make_fp32({1, 1}, {5.0f});
    auto B = make_fp32({1, 1}, {4.0f});
    auto C = Tensor::create(make_shape(1, 1), DType::FP32);
    C.data<float>()[0] = 999.0f;

    gemm_fp32_naive(A, B, C, 1.0f, 0.0f);

    EXPECT_FLOAT_EQ(C.data<float>()[0], 20.0f);
}

// ---- GemmNaive: shape mismatch throws ----------------------------------------

TEST(GemmNaive, ShapeMismatchABThrows) {
    auto A = Tensor::create(make_shape(2, 3), DType::FP32);
    auto B = Tensor::create(make_shape(4, 2), DType::FP32);
    auto C = Tensor::create(make_shape(2, 2), DType::FP32);
    EXPECT_THROW(gemm_fp32_naive(A, B, C), std::invalid_argument);
}

TEST(GemmNaive, ShapeMismatchCThrows) {
    auto A = Tensor::create(make_shape(2, 3), DType::FP32);
    auto B = Tensor::create(make_shape(3, 4), DType::FP32);
    auto C = Tensor::create(make_shape(2, 3), DType::FP32);
    EXPECT_THROW(gemm_fp32_naive(A, B, C), std::invalid_argument);
}

TEST(GemmNaive, WrongRankThrows) {
    auto A = Tensor::create(make_shape(2, 3, 1), DType::FP32);
    auto B = Tensor::create(make_shape(3, 4), DType::FP32);
    auto C = Tensor::create(make_shape(2, 4), DType::FP32);
    EXPECT_THROW(gemm_fp32_naive(A, B, C), std::invalid_argument);
}

// ============================================================================
// gemm_fp32_parallel tests (ENG-302)
// ============================================================================

// Compares parallel vs naive for a square N x N matrix.
// Uses mt19937 seeded at 42 so the test is deterministic.
static void run_parallel_vs_naive(int64_t N, int n_threads, float tol = 1e-4f) {
    auto A = Tensor::create(make_shape(N, N), DType::FP32);
    auto B = Tensor::create(make_shape(N, N), DType::FP32);
    auto C_naive = Tensor::create(make_shape(N, N), DType::FP32);
    auto C_par = Tensor::create(make_shape(N, N), DType::FP32);

    fill_random(A, 42);
    fill_random(B, 43);
    fill_fp32(C_naive, 0.0f);
    fill_fp32(C_par, 0.0f);

    gemm_fp32_naive(A, B, C_naive, 1.0f, 0.0f);
    gemm_fp32_parallel(A, B, C_par, TilingConfig{}, n_threads, 1.0f, 0.0f);

    EXPECT_LE(max_abs_diff(C_naive, C_par), tol) << "N=" << N << " n_threads=" << n_threads;
}

// Square sizes: 64, 128, 256 (small enough for a unit test budget).
TEST(GemmParallel, MatchesNaive_Square_64) {
    run_parallel_vs_naive(64, 0);
}
TEST(GemmParallel, MatchesNaive_Square_128) {
    run_parallel_vs_naive(128, 0);
}
TEST(GemmParallel, MatchesNaive_Square_256) {
    run_parallel_vs_naive(256, 0);
}

// alpha=3.0, beta=0.25: C = 3*A*B + 0.25*C_init
TEST(GemmParallel, AlphaBeta) {
    const int64_t N = 64;
    auto A = Tensor::create(make_shape(N, N), DType::FP32);
    auto B = Tensor::create(make_shape(N, N), DType::FP32);
    auto C_naive = Tensor::create(make_shape(N, N), DType::FP32);
    auto C_par = Tensor::create(make_shape(N, N), DType::FP32);

    fill_random(A, 42);
    fill_random(B, 43);
    // Both C tensors start with the same non-zero initial values so beta is exercised.
    fill_random(C_naive, 7);
    {
        // Copy C_naive initial state into C_par before it gets overwritten.
        const float* src = C_naive.data<float>();
        float* dst = C_par.data<float>();
        for (int64_t i = 0; i < N * N; ++i)
            dst[i] = src[i];
    }

    gemm_fp32_naive(A, B, C_naive, 3.0f, 0.25f);
    gemm_fp32_parallel(A, B, C_par, TilingConfig{}, 0, 3.0f, 0.25f);

    EXPECT_LE(max_abs_diff(C_naive, C_par), 1e-4f);
}

// Non-square: 37x53 * 53x71 -> 37x71
TEST(GemmParallel, NonSquare) {
    auto A = Tensor::create(make_shape(37, 53), DType::FP32);
    auto B = Tensor::create(make_shape(53, 71), DType::FP32);
    auto C_naive = Tensor::create(make_shape(37, 71), DType::FP32);
    auto C_par = Tensor::create(make_shape(37, 71), DType::FP32);

    fill_random(A, 42);
    fill_random(B, 43);
    fill_fp32(C_naive, 0.0f);
    fill_fp32(C_par, 0.0f);

    gemm_fp32_naive(A, B, C_naive, 1.0f, 0.0f);
    gemm_fp32_parallel(A, B, C_par, TilingConfig{}, 0, 1.0f, 0.0f);

    EXPECT_LE(max_abs_diff(C_naive, C_par), 1e-4f);
}

// n_threads=1 should give identical results to naive within fp tolerance.
TEST(GemmParallel, SingleThread) {
    run_parallel_vs_naive(128, 1);
}

// Multi-thread: up to 4 threads (or however many are available).
TEST(GemmParallel, MultiThread) {
    const int hw = static_cast<int>(std::thread::hardware_concurrency());
    const int n_threads = std::max(1, std::min(4, hw));
    run_parallel_vs_naive(256, n_threads);
}

// Shape mismatch must throw just like the naive variant.
TEST(GemmParallel, ShapeMismatchThrows) {
    auto A = Tensor::create(make_shape(2, 3), DType::FP32);
    auto B = Tensor::create(make_shape(4, 2), DType::FP32);
    auto C = Tensor::create(make_shape(2, 2), DType::FP32);
    EXPECT_THROW(gemm_fp32_parallel(A, B, C), std::invalid_argument);
}
