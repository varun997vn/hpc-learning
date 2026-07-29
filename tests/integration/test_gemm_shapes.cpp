// Integration tests: all variants, representative sizes, shape error paths.
// Budget: < 30s total. Sizes chosen to stay in cache for speed.
#include "engine/kernels/gemm.hpp"
#include "engine/tensor.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <random>
#include <stdexcept>

using namespace ie;
using namespace ie::kernels;

static Tensor rand_fp32(int64_t r, int64_t c, uint32_t seed = 7) {
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

// ---- All variants produce non-NaN output ------------------------------------

TEST(GemmIntegration, AllVariants_64) {
    auto A = rand_fp32(64, 64, 1);
    auto B = rand_fp32(64, 64, 2);
    auto C_ref = Tensor::create(make_shape(64, 64), DType::FP32);
    auto C_t   = Tensor::create(make_shape(64, 64), DType::FP32);
    auto C_p   = Tensor::create(make_shape(64, 64), DType::FP32);
    auto C_s   = Tensor::create(make_shape(64, 64), DType::FP32);
    gemm_fp32_naive(A, B, C_ref, 1.0f, 0.0f);
    gemm_fp32_tiled(A, B, C_t, {}, 1.0f, 0.0f);
    gemm_fp32_parallel(A, B, C_p, {}, 1, 1.0f, 0.0f);
    gemm_fp32_simd(A, B, C_s, {}, 1, 1.0f, 0.0f);
    EXPECT_LT(max_abs_diff(C_ref, C_t), 1e-4f);
    EXPECT_LT(max_abs_diff(C_ref, C_p), 1e-4f);
    EXPECT_LT(max_abs_diff(C_ref, C_s), 1e-3f);
}

TEST(GemmIntegration, AllVariants_128) {
    auto A = rand_fp32(128, 128, 3);
    auto B = rand_fp32(128, 128, 4);
    auto C_ref = Tensor::create(make_shape(128, 128), DType::FP32);
    auto C_t   = Tensor::create(make_shape(128, 128), DType::FP32);
    auto C_p   = Tensor::create(make_shape(128, 128), DType::FP32);
    auto C_s   = Tensor::create(make_shape(128, 128), DType::FP32);
    gemm_fp32_naive(A, B, C_ref, 1.0f, 0.0f);
    gemm_fp32_tiled(A, B, C_t, {}, 1.0f, 0.0f);
    gemm_fp32_parallel(A, B, C_p, {}, 2, 1.0f, 0.0f);
    gemm_fp32_simd(A, B, C_s, {}, 1, 1.0f, 0.0f);
    EXPECT_LT(max_abs_diff(C_ref, C_t), 1e-4f);
    EXPECT_LT(max_abs_diff(C_ref, C_p), 1e-4f);
    EXPECT_LT(max_abs_diff(C_ref, C_s), 1e-3f);
}

TEST(GemmIntegration, NonPowerOfTwo_384x768) {
    auto A = rand_fp32(384, 128, 5);
    auto B = rand_fp32(128, 768, 6);
    auto C_ref = Tensor::create(make_shape(384, 768), DType::FP32);
    auto C_t   = Tensor::create(make_shape(384, 768), DType::FP32);
    gemm_fp32_naive(A, B, C_ref, 1.0f, 0.0f);
    gemm_fp32_tiled(A, B, C_t, {}, 1.0f, 0.0f);
    EXPECT_LT(max_abs_diff(C_ref, C_t), 1e-4f);
}

// ---- Shape error paths -------------------------------------------------------

TEST(GemmIntegration, MismatchedShapes_Throws) {
    auto A = rand_fp32(4, 3);
    auto B = rand_fp32(4, 4);  // inner dim mismatch
    auto C = Tensor::create(make_shape(4, 4), DType::FP32);
    EXPECT_THROW(gemm_fp32_naive(A, B, C), std::invalid_argument);
    EXPECT_THROW(gemm_fp32_tiled(A, B, C), std::invalid_argument);
    EXPECT_THROW(gemm_fp32_parallel(A, B, C), std::invalid_argument);
    EXPECT_THROW(gemm_fp32_simd(A, B, C), std::invalid_argument);
}

TEST(GemmIntegration, WrongOutputShape_Throws) {
    auto A = rand_fp32(4, 4);
    auto B = rand_fp32(4, 4);
    auto C = Tensor::create(make_shape(3, 4), DType::FP32);  // C should be [4,4]
    EXPECT_THROW(gemm_fp32_naive(A, B, C), std::invalid_argument);
    EXPECT_THROW(gemm_fp32_tiled(A, B, C), std::invalid_argument);
}
