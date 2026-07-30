#include "engine/kernels/gemm.hpp"

#include <stdexcept>

namespace ie {
namespace kernels {

static void validate_int8_gemm_shapes(const Tensor &A, const Tensor &B,
                                      const Tensor &C) {
  if (A.shape().rank != 2 || B.shape().rank != 2 || C.shape().rank != 2)
    throw std::invalid_argument("gemm_int8_fixed: all tensors must be rank-2");
  if (A.dtype() != DType::INT8 || B.dtype() != DType::INT8)
    throw std::invalid_argument("gemm_int8_fixed: A and B must be DType::INT8");
  if (C.dtype() != DType::FP32)
    throw std::invalid_argument("gemm_int8_fixed: C must be DType::FP32");
  if (A.shape()[1] != B.shape()[0])
    throw std::invalid_argument(
        "gemm_int8_fixed: inner dimension mismatch (A.cols != B.rows)");
  if (C.shape()[0] != A.shape()[0] || C.shape()[1] != B.shape()[1])
    throw std::invalid_argument("gemm_int8_fixed: C shape must be [M, N]");
}

void gemm_int8_fixed(const Tensor &A, const Tensor &B, Tensor &C,
                     QuantParams qp) {
  validate_int8_gemm_shapes(A, B, C);

  const int64_t M = A.shape()[0];
  const int64_t K = A.shape()[1];
  const int64_t N = B.shape()[1];

  const int8_t *a = A.data<int8_t>();
  const int8_t *b = B.data<int8_t>();
  float *c = C.data<float>();

  // Dequantization multiplier applied to the int32 accumulator.
  // C[i][j] = acc * (scale_a * scale_b / scale_c)
  const float out_scale = qp.scale_a * qp.scale_b / qp.scale_c;

  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      // Accumulate in int32 — never widen to int64 or float per step,
      // matching the fixed-point precision contract in CLAUDE.md.
      int32_t acc = 0;
      for (int64_t k = 0; k < K; ++k)
        acc += static_cast<int32_t>(a[i * K + k]) *
               static_cast<int32_t>(b[k * N + j]);
      c[i * N + j] = static_cast<float>(acc) * out_scale;
    }
  }
}

} // namespace kernels
} // namespace ie
