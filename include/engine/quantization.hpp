#pragma once
#include "engine/tensor.hpp"
#include "engine/types.hpp"

#include <limits>

namespace ie {

// Per-tensor quantization parameters produced by the Calibrator.
struct QuantizationParams {
    float scale = 1.0f;
    int32_t zero_point = 0;
};

// Running min/max statistics collected during calibration.
struct CalibStats {
    float running_min = std::numeric_limits<float>::max();
    float running_max = std::numeric_limits<float>::lowest();
};

// Per-tensor activation calibrator.
// Streams FP32 tensors through update(), then calls finalize() once.
class Calibrator {
  public:
    enum class Mode { Symmetric, Asymmetric };

    explicit Calibrator(Mode mode = Mode::Symmetric) : mode_(mode) {}

    // Observe a FP32 tensor; updates running min/max element-wise.
    void update(const Tensor& t);

    // Compute QuantizationParams from accumulated statistics.
    // Symmetric:  scale = max(|min|, |max|) / 127, zero_point = 0.
    // Asymmetric: scale = (max - min) / 255,        zero_point = round(-min/scale) - 128.
    // Edge case (all-zero): scale = 1.0, zero_point = 0.
    QuantizationParams finalize() const;

    // Clear accumulated statistics for reuse.
    void reset();

    const CalibStats& stats() const { return stats_; }

  private:
    Mode mode_;
    CalibStats stats_;
};

} // namespace ie
