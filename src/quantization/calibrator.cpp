#include "engine/quantization.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ie {

void Calibrator::observe(const Tensor& t) {
    if (t.dtype() != DType::FP32)
        throw std::invalid_argument("Calibrator::observe: tensor must be FP32");

    const float* p = t.data<float>();
    const int64_t n = t.numel();

    for (int64_t i = 0; i < n; ++i) {
        const float v = p[i];
        if (!has_data_) {
            stats_.min_val = stats_.max_val = v;
            has_data_ = true;
        } else {
            if (v < stats_.min_val) stats_.min_val = v;
            if (v > stats_.max_val) stats_.max_val = v;
        }
    }
    stats_.abs_max = std::max(std::abs(stats_.min_val), std::abs(stats_.max_val));
}

QuantizationParams Calibrator::compute_symmetric() const {
    const float scale = stats_.abs_max > 0.0f ? stats_.abs_max / 127.0f : 1.0f;
    return {scale, 0};
}

QuantizationParams Calibrator::compute_asymmetric() const {
    const float range = stats_.max_val - stats_.min_val;
    const float scale = range > 0.0f ? range / 255.0f : 1.0f;
    const int32_t zp = std::max(-128, std::min(127,
        static_cast<int32_t>(std::round(-stats_.min_val / scale))));
    return {scale, zp};
}

void Calibrator::reset() {
    stats_ = {};
    has_data_ = false;
}

static inline int8_t clamp_int8(float v) {
    return static_cast<int8_t>(
        std::max(-128.0f, std::min(127.0f, std::round(v))));
}

void quantize_symmetric(const Tensor& src, Tensor& dst, const QuantizationParams& qp) {
    if (src.dtype() != DType::FP32)
        throw std::invalid_argument("quantize_symmetric: src must be FP32");
    if (dst.dtype() != DType::INT8)
        throw std::invalid_argument("quantize_symmetric: dst must be INT8");
    if (src.shape() != dst.shape())
        throw std::invalid_argument("quantize_symmetric: src and dst shapes must match");

    const float inv_scale = qp.scale > 0.0f ? 1.0f / qp.scale : 1.0f;
    const float* s = src.data<float>();
    int8_t* d = dst.data<int8_t>();
    const int64_t n = src.numel();

    for (int64_t i = 0; i < n; ++i)
        d[i] = clamp_int8(s[i] * inv_scale);
}

void dequantize_symmetric(const Tensor& src, Tensor& dst, const QuantizationParams& qp) {
    if (src.dtype() != DType::INT8)
        throw std::invalid_argument("dequantize_symmetric: src must be INT8");
    if (dst.dtype() != DType::FP32)
        throw std::invalid_argument("dequantize_symmetric: dst must be FP32");
    if (src.shape() != dst.shape())
        throw std::invalid_argument("dequantize_symmetric: src and dst shapes must match");

    const int8_t* s = src.data<int8_t>();
    float* d = dst.data<float>();
    const int64_t n = src.numel();

    for (int64_t i = 0; i < n; ++i)
        d[i] = static_cast<float>(s[i]) * qp.scale;
}

} // namespace ie
