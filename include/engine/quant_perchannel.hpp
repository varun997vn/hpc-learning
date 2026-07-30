#pragma once
#include "engine/quantization.hpp"
#include "engine/tensor.hpp"

#include <vector>

namespace ie {

// Symmetric per-channel quantization parameters for a weight tensor.
// One scale per output channel; zero_point is always 0 (symmetric).
struct PerChannelParams {
    std::vector<float> scales;        // one scale per output channel
    std::vector<int32_t> zero_points; // one zero_point per output channel (0 for symmetric)
};

// Accumulates per-channel (per-row) abs-max statistics from FP32 weight tensors.
// Designed for [out_channels, in_features] shaped weight tensors.
class PerChannelCalibrator {
  public:
    // Observe a rank-2 FP32 weight tensor [out_channels, in_features].
    // Updates per-row abs_max via running maximum across multiple calls.
    // Throws std::invalid_argument on dtype or rank mismatch.
    void observe(const Tensor& weights);

    // Compute symmetric per-channel params: scale[c] = abs_max[c] / 127.
    // If abs_max[c] == 0, scale[c] = 1.0f.  zero_point is always 0.
    PerChannelParams compute_symmetric() const;

    // Clear all accumulated state.
    void reset();

    int num_channels() const { return static_cast<int>(channel_abs_max_.size()); }

  private:
    std::vector<float> channel_abs_max_;
    int in_features_ = 0;
};

// Quantize a 2D FP32 weight tensor [out_channels, in_features] to INT8
// using per-channel symmetric params. dst must be pre-allocated INT8, same shape.
// Throws std::invalid_argument on shape/dtype mismatch.
void quantize_per_channel(const Tensor& src, Tensor& dst, const PerChannelParams& params);

// Dequantize INT8 [out_channels, in_features] to FP32 using per-channel scales.
// Throws std::invalid_argument on shape/dtype mismatch.
void dequantize_per_channel(const Tensor& src, Tensor& dst, const PerChannelParams& params);

} // namespace ie
