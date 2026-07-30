#pragma once
#include "engine/quantization.hpp"
#include "engine/tensor.hpp"

#include <vector>

// ENG-402: Per-channel symmetric weight quantization.
//
// Accuracy rationale
// ------------------
// Weights are quantized per output channel (one scale per row of the weight
// matrix) rather than per tensor.  This is critical for accuracy on
// MobileNetV2-class models: per-tensor weight quantization collapses channels
// with very different magnitude ranges onto a single scale, producing large
// quantization error in low-magnitude channels.  Per-channel quantization keeps
// error within scale[c]/2 for every channel independently, which is the primary
// reason the INT8 accuracy drop target of <1% top-1 on ImageNet is achievable.
//
// Quantization scheme
// -------------------
// Symmetric (zero_point = 0):  q = clamp(round(w / scale), -128, 127)
//                               w' = q * scale
// INT8 range: [-128, 127]  (signed, full 256 values).
// scale[c] = max(|w|) over channel c / 127.0f.
//
// Typical usage
// -------------
//   PerChannelCalibrator cal;
//   cal.observe(weight_tensor);          // call once per weight layer
//   auto params = cal.compute_symmetric();
//   auto q_weights = Tensor::create(weight_tensor.shape(), DType::INT8);
//   quantize_per_channel(weight_tensor, q_weights, params);
//   // Store params.scales alongside the INT8 weight blob for inference.

namespace ie {

// Symmetric per-channel quantization parameters for a weight tensor.
// One scale per output channel; zero_point is always 0 (symmetric).
struct PerChannelParams {
    std::vector<float> scales;        // one scale per output channel
    std::vector<int32_t> zero_points; // one zero_point per output channel (0 for symmetric)
};

// Accumulates per-channel (per-row) abs-max statistics from FP32 weight tensors.
// Designed for [out_channels, in_features] shaped weight tensors.
//
// Multiple calls to observe() perform a running maximum, so the calibrator
// handles the case of streaming weight shards or re-observing the same tensor
// (idempotent for repeated identical data).
class PerChannelCalibrator {
  public:
    // Observe a rank-2 FP32 weight tensor [out_channels, in_features].
    // Updates per-row abs_max via running maximum across multiple calls.
    // Throws std::invalid_argument on dtype or rank mismatch, and on shape
    // mismatch with a previous observation.
    void observe(const Tensor& weights);

    // Compute symmetric per-channel params: scale[c] = abs_max[c] / 127.
    // If abs_max[c] == 0 (all-zero channel), scale[c] = 1.0f to avoid
    // division by zero during quantization.  zero_point is always 0.
    PerChannelParams compute_symmetric() const;

    // Clear all accumulated state for reuse on a different weight tensor.
    void reset();

    int num_channels() const { return static_cast<int>(channel_abs_max_.size()); }

  private:
    std::vector<float> channel_abs_max_;
    int in_features_ = 0;
};

// Quantize a 2D FP32 weight tensor [out_channels, in_features] to INT8
// using per-channel symmetric params.  dst must be pre-allocated INT8 with
// the same shape as src.
// Formula: dst[c][i] = clamp(round(src[c][i] / scales[c]), -128, 127).
// Throws std::invalid_argument on shape/dtype mismatch or channel count
// mismatch between src and params.
void quantize_per_channel(const Tensor& src, Tensor& dst, const PerChannelParams& params);

// Dequantize INT8 [out_channels, in_features] to FP32 using per-channel scales.
// Formula: dst[c][i] = src[c][i] * scales[c].
// Throws std::invalid_argument on shape/dtype mismatch or channel count
// mismatch between src and params.
void dequantize_per_channel(const Tensor& src, Tensor& dst, const PerChannelParams& params);

} // namespace ie
