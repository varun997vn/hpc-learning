#include "engine/kernels/gemm.hpp"

#include <stdexcept>

namespace ie {
namespace kernels {

void gemm_int8_fixed(const Tensor &A, const Tensor &B, Tensor &C,
                     QuantParams qp) {
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
  // ENG-404: computation not yet implemented
  (void)qp;
}

} // namespace kernels
} // namespace ie
