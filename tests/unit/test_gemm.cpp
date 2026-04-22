// Unit tests for gemm_fp32_naive.
// Reference values for 2×2 and non-square cases computed by hand; no kernel
// logic is duplicated here.
#include "engine/kernels/gemm.hpp"
#include "engine/tensor.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <stdexcept>

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

// ---- 1×1 identity multiply --------------------------------------------------

TEST(GemmNaive, OneByOne) {
    // C = 1*A*B + 0*C  =>  [2] * [3] = [6]
    auto A = make_fp32({1, 1}, {2.0f});
    auto B = make_fp32({1, 1}, {3.0f});
    auto C = Tensor::create(make_shape(1, 1), DType::FP32);
    fill_fp32(C, 0.0f);

    gemm_fp32_naive(A, B, C);

    EXPECT_FLOAT_EQ(C.data<float>()[0], 6.0f);
}

// ---- 2×2 known result -------------------------------------------------------

TEST(GemmNaive, TwoByTwo) {
    // A = [[1,2],[3,4]]   B = [[5,6],[7,8]]
    // C[0][0] = 1*5 + 2*7 = 19
    // C[0][1] = 1*6 + 2*8 = 22
    // C[1][0] = 3*5 + 4*7 = 43
    // C[1][1] = 3*6 + 4*8 = 50
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

// ---- Non-square: 2×3 × 3×4 = 2×4 ------------------------------------------

TEST(GemmNaive, NonSquare2x3Times3x4) {
    // A [2×3]:  row0 = {1,2,3}, row1 = {4,5,6}
    // B [3×4]:  row0 = {7,8,9,10}, row1 = {11,12,13,14}, row2 = {15,16,17,18}
    // C[0][0] = 1*7  + 2*11 + 3*15 = 7  + 22 + 45 = 74
    // C[0][1] = 1*8  + 2*12 + 3*16 = 8  + 24 + 48 = 80
    // C[0][2] = 1*9  + 2*13 + 3*17 = 9  + 26 + 51 = 86
    // C[0][3] = 1*10 + 2*14 + 3*18 = 10 + 28 + 54 = 92
    // C[1][0] = 4*7  + 5*11 + 6*15 = 28 + 55 + 90 = 173
    // C[1][1] = 4*8  + 5*12 + 6*16 = 32 + 60 + 96 = 188
    // C[1][2] = 4*9  + 5*13 + 6*17 = 36 + 65 + 102 = 203
    // C[1][3] = 4*10 + 5*14 + 6*18 = 40 + 70 + 108 = 218
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

// ---- alpha/beta scaling -----------------------------------------------------

TEST(GemmNaive, AlphaScaling) {
    // alpha=2, beta=0: result should be 2 * (A*B)
    auto A = make_fp32({2, 2}, {1.0f, 0.0f, 0.0f, 1.0f}); // identity
    auto B = make_fp32({2, 2}, {3.0f, 4.0f, 5.0f, 6.0f});
    auto C = Tensor::create(make_shape(2, 2), DType::FP32);
    fill_fp32(C, 0.0f);

    gemm_fp32_naive(A, B, C, 2.0f, 0.0f);

    const float* c = C.data<float>();
    EXPECT_FLOAT_EQ(c[0], 6.0f);  // 2*3
    EXPECT_FLOAT_EQ(c[1], 8.0f);  // 2*4
    EXPECT_FLOAT_EQ(c[2], 10.0f); // 2*5
    EXPECT_FLOAT_EQ(c[3], 12.0f); // 2*6
}

TEST(GemmNaive, BetaAccumulate) {
    // alpha=1, beta=1: C = A*B + C_initial
    auto A = make_fp32({2, 2}, {1.0f, 0.0f, 0.0f, 1.0f}); // identity
    auto B = make_fp32({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    auto C = Tensor::create(make_shape(2, 2), DType::FP32);
    // C_initial = [[10, 20], [30, 40]]
    float* c = C.data<float>();
    c[0] = 10.0f;
    c[1] = 20.0f;
    c[2] = 30.0f;
    c[3] = 40.0f;

    gemm_fp32_naive(A, B, C, 1.0f, 1.0f);

    // A*B = [[1,2],[3,4]], plus C_initial
    EXPECT_FLOAT_EQ(c[0], 11.0f);
    EXPECT_FLOAT_EQ(c[1], 22.0f);
    EXPECT_FLOAT_EQ(c[2], 33.0f);
    EXPECT_FLOAT_EQ(c[3], 44.0f);
}

TEST(GemmNaive, BetaZeroWritesOverC) {
    // beta=0 must zero-out C before accumulating (handles uninitialized C)
    auto A = make_fp32({1, 1}, {5.0f});
    auto B = make_fp32({1, 1}, {4.0f});
    auto C = Tensor::create(make_shape(1, 1), DType::FP32);
    C.data<float>()[0] = 999.0f; // garbage initial value

    gemm_fp32_naive(A, B, C, 1.0f, 0.0f);

    EXPECT_FLOAT_EQ(C.data<float>()[0], 20.0f);
}

// ---- Shape mismatch throws --------------------------------------------------

TEST(GemmNaive, ShapeMismatchABThrows) {
    // A [2×3], B [4×2]: inner dimensions don't agree (3 != 4)
    auto A = Tensor::create(make_shape(2, 3), DType::FP32);
    auto B = Tensor::create(make_shape(4, 2), DType::FP32);
    auto C = Tensor::create(make_shape(2, 2), DType::FP32);
    EXPECT_THROW(gemm_fp32_naive(A, B, C), std::invalid_argument);
}

TEST(GemmNaive, ShapeMismatchCThrows) {
    // A [2×3], B [3×4], but C [2×3] (wrong output shape)
    auto A = Tensor::create(make_shape(2, 3), DType::FP32);
    auto B = Tensor::create(make_shape(3, 4), DType::FP32);
    auto C = Tensor::create(make_shape(2, 3), DType::FP32);
    EXPECT_THROW(gemm_fp32_naive(A, B, C), std::invalid_argument);
}

TEST(GemmNaive, WrongRankThrows) {
    // 3-D tensor should be rejected
    auto A = Tensor::create(make_shape(2, 3, 1), DType::FP32);
    auto B = Tensor::create(make_shape(3, 4), DType::FP32);
    auto C = Tensor::create(make_shape(2, 4), DType::FP32);
    EXPECT_THROW(gemm_fp32_naive(A, B, C), std::invalid_argument);
}
