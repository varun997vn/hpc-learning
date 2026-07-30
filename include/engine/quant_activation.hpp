#pragma once
#include "engine/quantization.hpp"
#include "engine/tensor.hpp"

namespace ie {

// EMA-based activation range calibrator.
// Tracks a smoothed min/max across multiple observation batches using
// exponential moving average: ema = momentum * ema + (1 - momentum) * new_val
class EmaCalibrator {
  public:
    explicit EmaCalibrator(float momentum = 0.99f);

    // Observe a batch of activations (FP32 tensor, any shape).
    // Updates EMA min/max. First call initialises from the batch statistics.
    void observe(const Tensor &activations);

    // Return symmetric quant params from current EMA range.
    QuantizationParams compute_symmetric() const;

    // Return asymmetric quant params from current EMA range.
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

// Quantize FP32 activations to INT8 with per-tensor asymmetric params.
// dst[i] = clamp(round(src[i] / scale) + zero_point, -128, 127)
// When zero_point=0 this is equivalent to quantize_symmetric.
void quantize_activation(const Tensor &src, Tensor &dst,
                         const QuantizationParams &qp);

// Dequantize INT8 activations to FP32.
// dst[i] = (src[i] - zero_point) * scale
void dequantize_activation(const Tensor &src, Tensor &dst,
                           const QuantizationParams &qp);

} // namespace ie
