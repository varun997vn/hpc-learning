#include "engine/quant_perchannel.hpp"

#include <stdexcept>

namespace ie {

// Stubs — will be replaced in the Green commit (ENG-402).

void PerChannelCalibrator::observe(const Tensor& weights) {
    if (weights.dtype() != DType::FP32) {
        throw std::invalid_argument("PerChannelCalibrator::observe: tensor must be FP32");
    }
    if (weights.shape().rank != 2) {
        throw std::invalid_argument("PerChannelCalibrator::observe: tensor must be rank-2");
    }
    // TODO: implement
    (void)weights;
}

PerChannelParams PerChannelCalibrator::compute_symmetric() const {
    // TODO: implement
    return {};
}

void PerChannelCalibrator::reset() {
    channel_abs_max_.clear();
    in_features_ = 0;
}

void quantize_per_channel(const Tensor& src, Tensor& dst, const PerChannelParams& params) {
    // TODO: implement
    (void)src;
    (void)dst;
    (void)params;
}

void dequantize_per_channel(const Tensor& src, Tensor& dst, const PerChannelParams& params) {
    // TODO: implement
    (void)src;
    (void)dst;
    (void)params;
}

} // namespace ie
