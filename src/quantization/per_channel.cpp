#include "engine/quant_perchannel.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ie {

void PerChannelCalibrator::observe(const Tensor& weights) {
    if (weights.dtype() != DType::FP32) {
        throw std::invalid_argument("PerChannelCalibrator::observe: tensor must be FP32");
    }
    if (weights.shape().rank != 2) {
        throw std::invalid_argument("PerChannelCalibrator::observe: tensor must be rank-2");
    }

    int out_channels = static_cast<int>(weights.shape()[0]);
    int in_features = static_cast<int>(weights.shape()[1]);

    // First observation: initialise per-channel state.
    if (channel_abs_max_.empty()) {
        channel_abs_max_.assign(static_cast<size_t>(out_channels), 0.0f);
        in_features_ = in_features;
    } else if (static_cast<int>(channel_abs_max_.size()) != out_channels ||
               in_features_ != in_features) {
        throw std::invalid_argument(
            "PerChannelCalibrator::observe: shape mismatch with previous observation");
    }

    const float* d = weights.data<float>();
    for (int c = 0; c < out_channels; ++c) {
        for (int f = 0; f < in_features; ++f) {
            float abs_val = std::abs(d[c * in_features + f]);
            channel_abs_max_[static_cast<size_t>(c)] =
                std::max(channel_abs_max_[static_cast<size_t>(c)], abs_val);
        }
    }
}

PerChannelParams PerChannelCalibrator::compute_symmetric() const {
    PerChannelParams p;
    int n = static_cast<int>(channel_abs_max_.size());
    p.scales.resize(static_cast<size_t>(n));
    p.zero_points.assign(static_cast<size_t>(n), 0); // symmetric: always 0

    for (int c = 0; c < n; ++c) {
        float abs_max = channel_abs_max_[static_cast<size_t>(c)];
        // Guard against degenerate all-zero channel weight rows.
        p.scales[static_cast<size_t>(c)] = (abs_max == 0.0f) ? 1.0f : (abs_max / 127.0f);
    }
    return p;
}

void PerChannelCalibrator::reset() {
    channel_abs_max_.clear();
    in_features_ = 0;
}

void quantize_per_channel(const Tensor& src, Tensor& dst, const PerChannelParams& params) {
    if (src.dtype() != DType::FP32) {
        throw std::invalid_argument("quantize_per_channel: src must be FP32");
    }
    if (dst.dtype() != DType::INT8) {
        throw std::invalid_argument("quantize_per_channel: dst must be INT8");
    }
    if (src.shape() != dst.shape()) {
        throw std::invalid_argument("quantize_per_channel: shape mismatch between src and dst");
    }
    if (src.shape().rank != 2) {
        throw std::invalid_argument("quantize_per_channel: tensor must be rank-2");
    }

    int out_channels = static_cast<int>(src.shape()[0]);
    int in_features = static_cast<int>(src.shape()[1]);

    if (static_cast<int>(params.scales.size()) != out_channels) {
        throw std::invalid_argument("quantize_per_channel: params.scales length != out_channels");
    }

    const float* s = src.data<float>();
    int8_t* d = dst.data<int8_t>();

    for (int c = 0; c < out_channels; ++c) {
        // Multiply by reciprocal once per channel to avoid per-element division.
        float inv_scale = 1.0f / params.scales[static_cast<size_t>(c)];
        for (int f = 0; f < in_features; ++f) {
            int32_t q = static_cast<int32_t>(std::round(s[c * in_features + f] * inv_scale));
            d[c * in_features + f] = static_cast<int8_t>(std::clamp(q, -128, 127));
        }
    }
}

void dequantize_per_channel(const Tensor& src, Tensor& dst, const PerChannelParams& params) {
    if (src.dtype() != DType::INT8) {
        throw std::invalid_argument("dequantize_per_channel: src must be INT8");
    }
    if (dst.dtype() != DType::FP32) {
        throw std::invalid_argument("dequantize_per_channel: dst must be FP32");
    }
    if (src.shape() != dst.shape()) {
        throw std::invalid_argument("dequantize_per_channel: shape mismatch between src and dst");
    }
    if (src.shape().rank != 2) {
        throw std::invalid_argument("dequantize_per_channel: tensor must be rank-2");
    }

    int out_channels = static_cast<int>(src.shape()[0]);
    int in_features = static_cast<int>(src.shape()[1]);

    if (static_cast<int>(params.scales.size()) != out_channels) {
        throw std::invalid_argument("dequantize_per_channel: params.scales length != out_channels");
    }

    const int8_t* s = src.data<int8_t>();
    float* d = dst.data<float>();

    for (int c = 0; c < out_channels; ++c) {
        float scale = params.scales[static_cast<size_t>(c)];
        for (int f = 0; f < in_features; ++f) {
            d[c * in_features + f] = static_cast<float>(s[c * in_features + f]) * scale;
        }
    }
}

} // namespace ie
