#pragma once
#include "engine/tensor.hpp"
#include <cstdint>

namespace ie {

struct QuantizationParams {
    float scale = 1.0f;
    int32_t zero_point = 0;
};

struct CalibStats {
    float min_val = 0.0f;
    float max_val = 0.0f;
    float abs_max = 0.0f;
};

class Calibrator {
public:
    void observe(const Tensor& t);

    // scale = abs_max / 127, zero_point = 0
    QuantizationParams compute_symmetric() const;

    // scale = (max - min) / 255, zero_point = clamp(round(-min/scale), -128, 127)
    QuantizationParams compute_asymmetric() const;

    void reset();
    const CalibStats& stats() const { return stats_; }

private:
    CalibStats stats_;
    bool has_data_ = false;
};

// FP32 → INT8  (symmetric): dst[i] = clamp(round(src[i] / scale), -128, 127)
void quantize_symmetric(const Tensor& src, Tensor& dst, const QuantizationParams& qp);

// INT8 → FP32  (symmetric): dst[i] = src[i] * scale
void dequantize_symmetric(const Tensor& src, Tensor& dst, const QuantizationParams& qp);

} // namespace ie
