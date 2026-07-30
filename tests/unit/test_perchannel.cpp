#include "engine/quant_perchannel.hpp"
#include "engine/tensor.hpp"
#include "engine/types.hpp"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>

using namespace ie;

// ---------------------------------------------------------------------------
// PerChannelCalibrator tests
// ---------------------------------------------------------------------------

// 2x4 weight tensor: verify per-channel abs_max and resulting scale.
TEST(PerChannelCalibrator, TwoChannels) {
    auto weights = Tensor::create(make_shape(2, 4), DType::FP32);
    float* d = weights.data<float>();
    // Channel 0: abs_max = 2.0
    d[0] = 1.0f;
    d[1] = -2.0f;
    d[2] = 0.5f;
    d[3] = -1.5f;
    // Channel 1: abs_max = 0.3
    d[4] = 0.1f;
    d[5] = -0.3f;
    d[6] = 0.2f;
    d[7] = 0.0f;

    PerChannelCalibrator cal;
    cal.observe(weights);

    ASSERT_EQ(cal.num_channels(), 2);
    auto params = cal.compute_symmetric();
    ASSERT_EQ(static_cast<int>(params.scales.size()), 2);
    EXPECT_NEAR(params.scales[0], 2.0f / 127.0f, 1e-6f);
    EXPECT_NEAR(params.scales[1], 0.3f / 127.0f, 1e-6f);
    EXPECT_EQ(params.zero_points[0], 0);
    EXPECT_EQ(params.zero_points[1], 0);
}

// Channels with very different magnitude ranges.
TEST(PerChannelCalibrator, AsymmetricChannels) {
    auto weights = Tensor::create(make_shape(3, 4), DType::FP32);
    float* d = weights.data<float>();
    // Channel 0: abs_max = 10.0
    d[0] = 10.0f;
    d[1] = -5.0f;
    d[2] = 0.0f;
    d[3] = 3.0f;
    // Channel 1: abs_max = 0.1
    d[4] = 0.1f;
    d[5] = -0.05f;
    d[6] = 0.0f;
    d[7] = 0.08f;
    // Channel 2: abs_max = 100.0
    d[8] = 100.0f;
    d[9] = -50.0f;
    d[10] = 0.0f;
    d[11] = 30.0f;

    PerChannelCalibrator cal;
    cal.observe(weights);

    ASSERT_EQ(cal.num_channels(), 3);
    auto params = cal.compute_symmetric();
    ASSERT_EQ(static_cast<int>(params.scales.size()), 3);
    EXPECT_NEAR(params.scales[0], 10.0f / 127.0f, 1e-5f);
    EXPECT_NEAR(params.scales[1], 0.1f / 127.0f, 1e-7f);
    EXPECT_NEAR(params.scales[2], 100.0f / 127.0f, 1e-4f);
}

// Observe the same tensor twice: abs_max is idempotent (running max doesn't grow).
TEST(PerChannelCalibrator, MultiObserve) {
    auto weights = Tensor::create(make_shape(2, 4), DType::FP32);
    float* d = weights.data<float>();
    d[0] = 1.0f;
    d[1] = -2.0f;
    d[2] = 0.5f;
    d[3] = -1.5f;
    d[4] = 0.1f;
    d[5] = -0.3f;
    d[6] = 0.2f;
    d[7] = 0.0f;

    PerChannelCalibrator cal;
    cal.observe(weights);
    auto params1 = cal.compute_symmetric();
    ASSERT_EQ(static_cast<int>(params1.scales.size()), 2);

    cal.observe(weights); // second observation — abs_max should not change
    auto params2 = cal.compute_symmetric();
    ASSERT_EQ(static_cast<int>(params2.scales.size()), 2);

    EXPECT_NEAR(params1.scales[0], params2.scales[0], 1e-7f);
    EXPECT_NEAR(params1.scales[1], params2.scales[1], 1e-7f);
}

// reset() clears all accumulated channel data.
TEST(PerChannelCalibrator, Reset) {
    auto weights = Tensor::create(make_shape(2, 4), DType::FP32);
    float* d = weights.data<float>();
    d[0] = 1.0f;
    d[1] = -2.0f;
    d[2] = 0.5f;
    d[3] = -1.5f;
    d[4] = 0.1f;
    d[5] = -0.3f;
    d[6] = 0.2f;
    d[7] = 0.0f;

    PerChannelCalibrator cal;
    cal.observe(weights);
    ASSERT_EQ(cal.num_channels(), 2);

    cal.reset();
    EXPECT_EQ(cal.num_channels(), 0);
}

// Known values: scale[c] = abs_max[c] / 127, zero_point = 0 for all channels.
TEST(PerChannelCalibrator, SymmetricScales) {
    auto weights = Tensor::create(make_shape(2, 3), DType::FP32);
    float* d = weights.data<float>();
    // Channel 0: abs_max = 4.0
    d[0] = 4.0f;
    d[1] = -3.0f;
    d[2] = 2.0f;
    // Channel 1: abs_max = 6.0
    d[3] = -6.0f;
    d[4] = 5.0f;
    d[5] = 1.0f;

    PerChannelCalibrator cal;
    cal.observe(weights);
    auto params = cal.compute_symmetric();

    ASSERT_EQ(static_cast<int>(params.scales.size()), 2);
    EXPECT_NEAR(params.scales[0], 4.0f / 127.0f, 1e-6f);
    EXPECT_NEAR(params.scales[1], 6.0f / 127.0f, 1e-6f);
    EXPECT_EQ(params.zero_points[0], 0);
    EXPECT_EQ(params.zero_points[1], 0);
}

// ---------------------------------------------------------------------------
// quantize_per_channel / dequantize_per_channel tests
// ---------------------------------------------------------------------------

// Round-trip quantize→dequantize: max error per element < max(scales).
TEST(QuantizePerChannel, Roundtrip) {
    const int C = 4, F = 8;
    auto src = Tensor::create(make_shape(C, F), DType::FP32);
    float* s = src.data<float>();
    // Channels with different magnitude ranges so we exercise a range of scales.
    const float mag[] = {1.0f, 10.0f, 0.1f, 100.0f};
    for (int c = 0; c < C; ++c) {
        for (int f = 0; f < F; ++f) {
            // Values uniformly spaced in [-mag[c], mag[c])
            s[c * F + f] = mag[c] * (static_cast<float>(f - F / 2) / (F / 2));
        }
    }

    PerChannelCalibrator cal;
    cal.observe(src);
    auto params = cal.compute_symmetric();

    ASSERT_EQ(static_cast<int>(params.scales.size()), C);
    ASSERT_FALSE(params.scales.empty());

    auto dst_int8 = Tensor::create(make_shape(C, F), DType::INT8);
    quantize_per_channel(src, dst_int8, params);

    auto dst_fp32 = Tensor::create(make_shape(C, F), DType::FP32);
    dequantize_per_channel(dst_int8, dst_fp32, params);

    const float* orig = src.data<float>();
    const float* rec = dst_fp32.data<float>();
    // Worst-case round-trip error is scale/2 per channel; bound with max(scales).
    float max_scale = *std::max_element(params.scales.begin(), params.scales.end());
    for (int i = 0; i < C * F; ++i) {
        EXPECT_NEAR(orig[i], rec[i], max_scale) << "mismatch at linear index " << i;
    }
}

// Two channels with very different scales — no cross-channel bleeding.
TEST(QuantizePerChannel, ChannelIndependence) {
    const int C = 2, F = 4;
    auto src = Tensor::create(make_shape(C, F), DType::FP32);
    float* d = src.data<float>();
    // Channel 0: large values, abs_max = 100
    d[0] = 100.0f;
    d[1] = -100.0f;
    d[2] = 50.0f;
    d[3] = -50.0f;
    // Channel 1: small values, abs_max = 0.01
    d[4] = 0.01f;
    d[5] = -0.01f;
    d[6] = 0.005f;
    d[7] = -0.005f;

    PerChannelCalibrator cal;
    cal.observe(src);
    auto params = cal.compute_symmetric();

    ASSERT_EQ(static_cast<int>(params.scales.size()), C);

    auto dst_int8 = Tensor::create(make_shape(C, F), DType::INT8);
    quantize_per_channel(src, dst_int8, params);

    const int8_t* q = dst_int8.data<int8_t>();

    // Channel 0: scale = 100/127; 100 / scale = 127 exactly
    EXPECT_EQ(q[0], static_cast<int8_t>(127));
    EXPECT_EQ(q[1], static_cast<int8_t>(-127));

    // Channel 1: scale = 0.01/127; 0.01 / scale = 127 exactly.
    // If channel 1 had accidentally used channel 0's scale, q[4] would be ~0.
    EXPECT_EQ(q[4], static_cast<int8_t>(127));
    EXPECT_EQ(q[5], static_cast<int8_t>(-127));
}
