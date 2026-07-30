#include "engine/kernels/gemm.hpp"
#include "engine/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <random>
#include <vector>

using namespace ie;
using namespace ie::kernels;

// ---- Inline symmetric quantization helpers ---------------------------------
//
// These helpers perform per-tensor symmetric INT8 quantization (zero_point=0)
// without depending on ie_quant, keeping the unit tests self-contained.
//
// Scale is chosen so the maximum absolute value maps to ±127:
//   scale = max|x| / 127
// Quantization: q = clamp(round(x / scale), -127, 127)

static float max_abs_fp32(const float *data, int64_t n) {
  float m = 0.0f;
  for (int64_t i = 0; i < n; ++i)
    m = std::max(m, std::abs(data[i]));
  return m;
}

static Tensor quantize_symmetric(int64_t rows, int64_t cols, const float *data,
                                 float scale) {
  auto t = Tensor::create(make_shape(rows, cols), DType::INT8);
  int8_t *q = t.data<int8_t>();
  const int64_t n = rows * cols;
  for (int64_t i = 0; i < n; ++i) {
    float v = std::round(data[i] / scale);
    q[i] = static_cast<int8_t>(std::max(-127.0f, std::min(127.0f, v)));
  }
  return t;
}

// ---- Tests -----------------------------------------------------------------

// Trivial 1×1: INT8 A=[[2]], INT8 B=[[3]], scale=1 → FP32 C=[[6]]
TEST(GemmInt8, Trivial1x1) {
  auto A = Tensor::create(make_shape(1, 1), DType::INT8);
  auto B = Tensor::create(make_shape(1, 1), DType::INT8);
  auto C = Tensor::create(make_shape(1, 1), DType::FP32);

  *A.data<int8_t>() = 2;
  *B.data<int8_t>() = 3;
  *C.data<float>() = 0.0f;

  QuantParams qp; // scale_a=scale_b=scale_c=1.0 by default
  gemm_int8_fixed(A, B, C, qp);

  EXPECT_NEAR(*C.data<float>(), 6.0f, 1e-5f);
}

// 2×2 roundtrip: quantize known FP32 matrices, run INT8 GEMM, compare to
// gemm_fp32_naive within K*(scale_a + scale_b) — one quantization step per
// element of each operand accumulated over K reductions.
TEST(GemmInt8, Roundtrip2x2MatchesFP32) {
  constexpr int64_t M = 2, K = 2, N = 2;
  const float a_data[] = {0.5f, -0.3f, 0.1f, 0.8f};
  const float b_data[] = {0.4f, -0.2f, 0.7f, 0.6f};

  const float scale_a = max_abs_fp32(a_data, M * K) / 127.0f;
  const float scale_b = max_abs_fp32(b_data, K * N) / 127.0f;

  // FP32 reference via naive kernel
  auto A_fp = Tensor::create(make_shape(M, K), DType::FP32);
  auto B_fp = Tensor::create(make_shape(K, N), DType::FP32);
  auto C_ref = Tensor::create(make_shape(M, N), DType::FP32);
  std::copy(a_data, a_data + M * K, A_fp.data<float>());
  std::copy(b_data, b_data + K * N, B_fp.data<float>());
  std::fill(C_ref.data<float>(), C_ref.data<float>() + M * N, 0.0f);
  gemm_fp32_naive(A_fp, B_fp, C_ref);

  // INT8 GEMM
  auto A_q = quantize_symmetric(M, K, a_data, scale_a);
  auto B_q = quantize_symmetric(K, N, b_data, scale_b);
  auto C_got = Tensor::create(make_shape(M, N), DType::FP32);
  std::fill(C_got.data<float>(), C_got.data<float>() + M * N, 0.0f);

  QuantParams qp;
  qp.scale_a = scale_a;
  qp.scale_b = scale_b;
  qp.scale_c = 1.0f;
  gemm_int8_fixed(A_q, B_q, C_got, qp);

  // Tolerance: each operand quantizes with error ≤ scale/2; K reductions
  // accumulate that error on both sides → K*(scale_a + scale_b).
  const float tol = static_cast<float>(K) * (scale_a + scale_b);
  const float *ref = C_ref.data<float>();
  const float *got = C_got.data<float>();
  for (int64_t i = 0; i < M * N; ++i)
    EXPECT_NEAR(got[i], ref[i], tol) << "element " << i;
}

// Non-square [4,8]×[8,4] with random inputs
TEST(GemmInt8, NonSquare4x8x4) {
  constexpr int64_t M = 4, K = 8, N = 4;
  std::mt19937 rng(99);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::vector<float> a_data(M * K), b_data(K * N);
  for (auto &v : a_data)
    v = dist(rng);
  for (auto &v : b_data)
    v = dist(rng);

  const float scale_a = max_abs_fp32(a_data.data(), M * K) / 127.0f;
  const float scale_b = max_abs_fp32(b_data.data(), K * N) / 127.0f;

  auto A_fp = Tensor::create(make_shape(M, K), DType::FP32);
  auto B_fp = Tensor::create(make_shape(K, N), DType::FP32);
  auto C_ref = Tensor::create(make_shape(M, N), DType::FP32);
  std::copy(a_data.begin(), a_data.end(), A_fp.data<float>());
  std::copy(b_data.begin(), b_data.end(), B_fp.data<float>());
  std::fill(C_ref.data<float>(), C_ref.data<float>() + M * N, 0.0f);
  gemm_fp32_naive(A_fp, B_fp, C_ref);

  auto A_q = quantize_symmetric(M, K, a_data.data(), scale_a);
  auto B_q = quantize_symmetric(K, N, b_data.data(), scale_b);
  auto C_got = Tensor::create(make_shape(M, N), DType::FP32);
  std::fill(C_got.data<float>(), C_got.data<float>() + M * N, 0.0f);

  QuantParams qp;
  qp.scale_a = scale_a;
  qp.scale_b = scale_b;
  qp.scale_c = 1.0f;
  gemm_int8_fixed(A_q, B_q, C_got, qp);

  const float tol = static_cast<float>(K) * (scale_a + scale_b);
  const float *ref = C_ref.data<float>();
  const float *got = C_got.data<float>();
  for (int64_t i = 0; i < M * N; ++i)
    EXPECT_NEAR(got[i], ref[i], tol) << "element " << i;
}

// Verify the dequantization formula: C[i][j] = acc * (scale_a * scale_b /
// scale_c). 1×1 case: acc = 4*5 = 20, scale_a=2, scale_b=3, scale_c=6 →
// out_scale=1 → C=20.
TEST(GemmInt8, ScaleCorrect) {
  auto A = Tensor::create(make_shape(1, 1), DType::INT8);
  auto B = Tensor::create(make_shape(1, 1), DType::INT8);
  auto C = Tensor::create(make_shape(1, 1), DType::FP32);
  *A.data<int8_t>() = 4;
  *B.data<int8_t>() = 5;
  *C.data<float>() = 0.0f;

  QuantParams qp;
  qp.scale_a = 2.0f;
  qp.scale_b = 3.0f;
  qp.scale_c = 6.0f; // out_scale = 2*3/6 = 1.0 → output = 20
  gemm_int8_fixed(A, B, C, qp);

  EXPECT_NEAR(*C.data<float>(), 20.0f, 1e-5f);
}

// Shape error: FP32 inputs must throw std::invalid_argument
TEST(GemmInt8, ShapeError_WrongDtype) {
  auto A = Tensor::create(make_shape(2, 2), DType::FP32); // wrong dtype
  auto B = Tensor::create(make_shape(2, 2), DType::INT8);
  auto C = Tensor::create(make_shape(2, 2), DType::FP32);
  EXPECT_THROW(gemm_int8_fixed(A, B, C), std::invalid_argument);
}

// Shape error: mismatched inner dimension must throw std::invalid_argument
TEST(GemmInt8, ShapeError_InnerDim) {
  auto A = Tensor::create(make_shape(2, 3), DType::INT8);
  auto B = Tensor::create(make_shape(4, 2), DType::INT8); // K mismatch: 3 != 4
  auto C = Tensor::create(make_shape(2, 2), DType::FP32);
  EXPECT_THROW(gemm_int8_fixed(A, B, C), std::invalid_argument);
}
