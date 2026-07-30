#include "engine/quantization.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ie {

void Calibrator::update(const Tensor& t) {
    if (t.dtype() != DType::FP32) {
        throw std::invalid_argument("Calibrator::update: tensor must be FP32");
    }
    const float* d = t.data<float>();
    int64_t n = t.numel();
    for (int64_t i = 0; i < n; ++i) {
        stats_.running_min = std::min(stats_.running_min, d[i]);
        stats_.running_max = std::max(stats_.running_max, d[i]);
    }
}

QuantizationParams Calibrator::finalize() const {
    float mn = stats_.running_min;
    float mx = stats_.running_max;

    // Edge case: all-zero tensor (or no data observed)
    if (mn == 0.0f && mx == 0.0f) {
        return {1.0f, 0};
    }
    // Handle edge: running_min/max never updated (still at sentinel values)
    if (mn == std::numeric_limits<float>::max()) {
        return {1.0f, 0};
    }

    if (mode_ == Mode::Symmetric) {
        float abs_max = std::max(std::abs(mn), std::abs(mx));
        float scale = abs_max / 127.0f;
        return {scale, 0};
    }

    // Asymmetric
    float scale = (mx - mn) / 255.0f;
    if (scale == 0.0f)
        scale = 1.0f;
    int32_t zero_point = static_cast<int32_t>(std::round(-mn / scale)) - 128;
    return {scale, zero_point};
}

void Calibrator::reset() {
    stats_ = CalibStats{};
}

} // namespace ie
