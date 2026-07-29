#include "engine/quantization.hpp"
#include "engine/tensor.hpp"

#include <cmath>
#include <gtest/gtest.h>

using namespace ie;

static Tensor make_fp32(std::initializer_list<float> vals) {
    auto t = Tensor::create(make_shape(static_cast<int64_t>(vals.size()), 1), DType::FP32);
    float* p = t.data<float>();
    int i = 0;
    for (float v : vals)
        p[i++] = v;
    return t;
}

// ---- Calibrator::observe ----------------------------------------------------

TEST(Calibrator, SingleTensor_StatsCorrect) {
    Calibrator cal;
    auto t = make_fp32({-2.0f, 0.0f, 3.0f, -1.0f});
    cal.observe(t);
    EXPECT_FLOAT_EQ(cal.stats().min_val, -2.0f);
    EXPECT_FLOAT_EQ(cal.stats().max_val, 3.0f);
    EXPECT_FLOAT_EQ(cal.stats().abs_max, 3.0f);
}

TEST(Calibrator, MultiTensor_StatsMerge) {
    Calibrator cal;
    cal.observe(make_fp32({1.0f, 2.0f}));
    cal.observe(make_fp32({-5.0f, 0.5f}));
    EXPECT_FLOAT_EQ(cal.stats().min_val, -5.0f);
    EXPECT_FLOAT_EQ(cal.stats().max_val, 2.0f);
    EXPECT_FLOAT_EQ(cal.stats().abs_max, 5.0f);
}

TEST(Calibrator, Reset_ClearsStats) {
    Calibrator cal;
    cal.observe(make_fp32({10.0f}));
    cal.reset();
    cal.observe(make_fp32({1.0f}));
    EXPECT_FLOAT_EQ(cal.stats().max_val, 1.0f);
}

// ---- compute_symmetric ------------------------------------------------------

TEST(Calibrator, Symmetric_ScaleFromAbsMax) {
    Calibrator cal;
    // abs_max = 8  →  scale = 8/127
    cal.observe(make_fp32({-4.0f, 0.0f, 8.0f}));
    auto qp = cal.compute_symmetric();
    EXPECT_NEAR(qp.scale, 8.0f / 127.0f, 1e-6f);
    EXPECT_EQ(qp.zero_point, 0);
}

// ---- compute_asymmetric -----------------------------------------------------

TEST(Calibrator, Asymmetric_ScaleAndZeroPoint) {
    Calibrator cal;
    // range [-1, 3], scale = 4/255, zp = round(1/scale) ≈ 64
    cal.observe(make_fp32({-1.0f, 3.0f}));
    auto qp = cal.compute_asymmetric();
    EXPECT_NEAR(qp.scale, 4.0f / 255.0f, 1e-6f);
    const int32_t expected_zp = static_cast<int32_t>(
        std::round(1.0f / (4.0f / 255.0f)));
    EXPECT_EQ(qp.zero_point, expected_zp);
}

// ---- quantize_symmetric / dequantize_symmetric ------------------------------

TEST(QuantizeSymmetric, Roundtrip_MaxErrorLeScale) {
    auto src = make_fp32({-1.0f, -0.5f, 0.0f, 0.5f, 1.0f});
    auto dst_q = Tensor::create(src.shape(), DType::INT8);
    auto dst_dq = Tensor::create(src.shape(), DType::FP32);

    QuantizationParams qp{1.0f / 127.0f, 0};
    quantize_symmetric(src, dst_q, qp);
    dequantize_symmetric(dst_q, dst_dq, qp);

    const float* orig = src.data<float>();
    const float* back = dst_dq.data<float>();
    for (int64_t i = 0; i < src.numel(); ++i)
        EXPECT_NEAR(back[i], orig[i], qp.scale) << "i=" << i;
}

TEST(QuantizeSymmetric, Clamping) {
    // Values beyond ±127*scale should clamp to ±127.
    QuantizationParams qp{1.0f, 0};  // scale=1 so INT8 range is [-128,127]
    auto src = make_fp32({-200.0f, 200.0f});
    auto dst = Tensor::create(src.shape(), DType::INT8);
    quantize_symmetric(src, dst, qp);
    EXPECT_EQ(dst.data<int8_t>()[0], -128);
    EXPECT_EQ(dst.data<int8_t>()[1], 127);
}

TEST(QuantizeSymmetric, AllZeros) {
    auto src = make_fp32({0.0f, 0.0f, 0.0f});
    auto dst = Tensor::create(src.shape(), DType::INT8);
    QuantizationParams qp{0.1f, 0};
    quantize_symmetric(src, dst, qp);
    for (int64_t i = 0; i < dst.numel(); ++i)
        EXPECT_EQ(dst.data<int8_t>()[i], 0);
}
