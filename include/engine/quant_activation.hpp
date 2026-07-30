#pragma once
#include "engine/quantization.hpp"
#include "engine/tensor.hpp"

namespace ie {

// EMA-based activation range calibrator for PTQ (Post-Training Quantization).
//
// Maintains a smoothed min/max of activation ranges across multiple observation
// batches using exponential moving average (EMA):
//
//   ema_t = momentum * ema_{t-1} + (1 - momentum) * batch_stat_t
//
// The first batch seeds the EMA directly (no cold-start bias from 0).
// High momentum (e.g., 0.99) keeps history; momentum=0 discards it each batch.
//
// Typical workflow:
//   EmaCalibrator cal;          // default momentum=0.99
//   for (auto& batch : dataset)
//       cal.observe(batch);     // updates EMA min/max
//   auto qp = cal.compute_symmetric();
class EmaCalibrator {
  public:
    explicit EmaCalibrator(float momentum = 0.99f);

    // Observe a batch of FP32 activations (any shape).
    // Scans batch min/max, then updates EMA: first batch seeds directly.
    void observe(const Tensor &activations);

    // Symmetric params: scale = abs_max / 127, zero_point = 0.
    // Handles the all-zero edge case (scale defaults to 1.0).
    QuantizationParams compute_symmetric() const;

    // Asymmetric params: scale = (ema_max - ema_min) / 255.
    // zero_point = clamp(round(-ema_min / scale) - 128, -128, 127).
    // The -128 shifts from the UINT8 zero-point convention to INT8, so the
    // full [-128, 127] range maps to [ema_min, ema_max] without saturation.
    QuantizationParams compute_asymmetric() const;

    void reset();

    float ema_min() const { return ema_min_; }
    float ema_max() const { return ema_max_; }
    int num_batches() const { return num_batches_; }

  private:
    float momentum_;
    float ema_min_ = 0.0f;
    float ema_max_ = 0.0f;
    int num_batches_ = 0;
};

// Quantize FP32 activations to INT8 with per-tensor params.
//
//   dst[i] = clamp(round(src[i] / scale) + zero_point, -128, 127)
//
// When zero_point=0 this is numerically identical to quantize_symmetric.
// Validation (dtype/shape mismatch) throws std::invalid_argument; the hot
// loop is exception-free.
void quantize_activation(const Tensor &src, Tensor &dst,
                         const QuantizationParams &qp);

// Dequantize INT8 activations to FP32.
//
//   dst[i] = (src[i] - zero_point) * scale
//
// This is the exact inverse of quantize_activation; roundtrip error is
// bounded by scale/2 for values within the calibrated range.
void dequantize_activation(const Tensor &src, Tensor &dst,
                           const QuantizationParams &qp);

} // namespace ie
