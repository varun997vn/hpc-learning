#include "engine/quant_activation.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ie {

EmaCalibrator::EmaCalibrator(float momentum) : momentum_(momentum) {}

void EmaCalibrator::observe(const Tensor &activations) {
    if (activations.dtype() != DType::FP32)
        throw std::invalid_argument(
            "EmaCalibrator::observe: tensor must be FP32");

    const float *p = activations.data<float>();
    const int64_t n = activations.numel();
    if (n == 0)
        return;

    float batch_min = p[0];
    float batch_max = p[0];
    for (int64_t i = 1; i < n; ++i) {
        if (p[i] < batch_min)
            batch_min = p[i];
        if (p[i] > batch_max)
            batch_max = p[i];
    }

    // First batch seeds the EMA directly to avoid a cold-start bias from 0.
    if (num_batches_ == 0) {
        ema_min_ = batch_min;
        ema_max_ = batch_max;
    } else {
        const float alpha = 1.0f - momentum_;
        ema_min_ = momentum_ * ema_min_ + alpha * batch_min;
        ema_max_ = momentum_ * ema_max_ + alpha * batch_max;
    }
    ++num_batches_;
}

QuantizationParams EmaCalibrator::compute_symmetric() const {
    const float abs_max = std::max(std::abs(ema_min_), std::abs(ema_max_));
    const float scale = abs_max > 0.0f ? abs_max / 127.0f : 1.0f;
    return {scale, 0};
}

QuantizationParams EmaCalibrator::compute_asymmetric() const {
    const float range = ema_max_ - ema_min_;
    const float scale = range > 0.0f ? range / 255.0f : 1.0f;
    // Subtract 128 to convert from the UINT8 zero-point convention (range
    // [0,255]) to the INT8 convention (range [-128,127]), ensuring that the
    // quantized range
    // [-128, 127] maps exactly to [ema_min, ema_max] with no saturation.
    const int32_t zp = std::max(
        -128,
        std::min(127,
                 static_cast<int32_t>(std::round(-ema_min_ / scale)) - 128));
    return {scale, zp};
}

void EmaCalibrator::reset() {
    ema_min_ = 0.0f;
    ema_max_ = 0.0f;
    num_batches_ = 0;
}

// ---------------------------------------------------------------------------
// quantize_activation / dequantize_activation
// ---------------------------------------------------------------------------

static inline int8_t saturate_int8(float v) {
    return static_cast<int8_t>(
        std::max(-128.0f, std::min(127.0f, std::round(v))));
}

void quantize_activation(const Tensor &src, Tensor &dst,
                         const QuantizationParams &qp) {
    if (src.dtype() != DType::FP32)
        throw std::invalid_argument("quantize_activation: src must be FP32");
    if (dst.dtype() != DType::INT8)
        throw std::invalid_argument("quantize_activation: dst must be INT8");
    if (src.shape() != dst.shape())
        throw std::invalid_argument(
            "quantize_activation: src and dst shapes must match");

    const float inv_scale = qp.scale > 0.0f ? 1.0f / qp.scale : 1.0f;
    const float zp_f = static_cast<float>(qp.zero_point);
    const float *s = src.data<float>();
    int8_t *d = dst.data<int8_t>();
    const int64_t n = src.numel();

    for (int64_t i = 0; i < n; ++i)
        d[i] = saturate_int8(s[i] * inv_scale + zp_f);
}

void dequantize_activation(const Tensor &src, Tensor &dst,
                           const QuantizationParams &qp) {
    if (src.dtype() != DType::INT8)
        throw std::invalid_argument("dequantize_activation: src must be INT8");
    if (dst.dtype() != DType::FP32)
        throw std::invalid_argument("dequantize_activation: dst must be FP32");
    if (src.shape() != dst.shape())
        throw std::invalid_argument(
            "dequantize_activation: src and dst shapes must match");

    const int8_t *s = src.data<int8_t>();
    float *d = dst.data<float>();
    const int64_t n = src.numel();

    for (int64_t i = 0; i < n; ++i)
        d[i] = (static_cast<float>(s[i]) - static_cast<float>(qp.zero_point)) *
               qp.scale;
}

} // namespace ie
