#include "engine/quant_activation.hpp"
#include "engine/quantization.hpp"
#include "engine/tensor.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>

using namespace ie;

// ---------------------------------------------------------------------------
// EmaCalibrator tests
// ---------------------------------------------------------------------------

TEST(EmaCalibrator, FirstBatch_SetsRange) {
    EmaCalibrator cal(0.9f);
    auto t = Tensor::create(make_shape(4), DType::FP32);
    float *p = t.data<float>();
    p[0] = -3.0f;
    p[1] = 1.0f;
    p[2] = 2.0f;
    p[3] = 5.0f;

    cal.observe(t);

    EXPECT_FLOAT_EQ(cal.ema_min(), -3.0f);
    EXPECT_FLOAT_EQ(cal.ema_max(), 5.0f);
    EXPECT_EQ(cal.num_batches(), 1);
}

TEST(EmaCalibrator, SecondBatch_Smoothed) {
    // momentum = 0.5 for easy manual calculation
    EmaCalibrator cal(0.5f);

    auto t1 = Tensor::create(make_shape(2), DType::FP32);
    t1.data<float>()[0] = -4.0f;
    t1.data<float>()[1] = 4.0f;
    cal.observe(t1); // ema_min=-4, ema_max=4

    auto t2 = Tensor::create(make_shape(2), DType::FP32);
    t2.data<float>()[0] = -2.0f;
    t2.data<float>()[1] = 2.0f;
    cal.observe(t2);
    // ema_min = 0.5*(-4) + 0.5*(-2) = -3
    // ema_max = 0.5*4  + 0.5*2  =  3
    EXPECT_FLOAT_EQ(cal.ema_min(), -3.0f);
    EXPECT_FLOAT_EQ(cal.ema_max(), 3.0f);
    EXPECT_EQ(cal.num_batches(), 2);
}

TEST(EmaCalibrator, Momentum099_SlowUpdate) {
    // With high momentum the range barely moves toward the new batch.
    EmaCalibrator cal(0.99f);

    auto t1 = Tensor::create(make_shape(1), DType::FP32);
    t1.data<float>()[0] = 1.0f;
    cal.observe(t1); // ema_min=1, ema_max=1

    // Reset-like: push ema toward 0 via momentum
    auto t2 = Tensor::create(make_shape(1), DType::FP32);
    t2.data<float>()[0] = -10.0f;
    cal.observe(t2);
    // ema_min = 0.99*1 + 0.01*(-10) = 0.99 - 0.10 = 0.89
    EXPECT_NEAR(cal.ema_min(), 0.89f, 1e-4f);
    // ema_max = 0.99*1 + 0.01*(-10) = 0.89
    EXPECT_NEAR(cal.ema_max(), 0.89f, 1e-4f);
}

TEST(EmaCalibrator, Momentum0_NoMemory) {
    // With momentum=0, every batch fully replaces the EMA.
    EmaCalibrator cal(0.0f);

    auto t1 = Tensor::create(make_shape(2), DType::FP32);
    t1.data<float>()[0] = -10.0f;
    t1.data<float>()[1] = 10.0f;
    cal.observe(t1);

    auto t2 = Tensor::create(make_shape(2), DType::FP32);
    t2.data<float>()[0] = -1.0f;
    t2.data<float>()[1] = 1.0f;
    cal.observe(t2);

    EXPECT_FLOAT_EQ(cal.ema_min(), -1.0f);
    EXPECT_FLOAT_EQ(cal.ema_max(), 1.0f);
}

TEST(EmaCalibrator, Reset_ClearsState) {
    EmaCalibrator cal(0.9f);
    auto t = Tensor::create(make_shape(2), DType::FP32);
    t.data<float>()[0] = -5.0f;
    t.data<float>()[1] = 5.0f;
    cal.observe(t);

    cal.reset();

    EXPECT_EQ(cal.num_batches(), 0);
    EXPECT_FLOAT_EQ(cal.ema_min(), 0.0f);
    EXPECT_FLOAT_EQ(cal.ema_max(), 0.0f);

    // After reset, next observe behaves like first batch.
    auto t2 = Tensor::create(make_shape(2), DType::FP32);
    t2.data<float>()[0] = -2.0f;
    t2.data<float>()[1] = 3.0f;
    cal.observe(t2);
    EXPECT_FLOAT_EQ(cal.ema_min(), -2.0f);
    EXPECT_FLOAT_EQ(cal.ema_max(), 3.0f);
    EXPECT_EQ(cal.num_batches(), 1);
}

TEST(EmaCalibrator, Symmetric_Params) {
    // ema_min=-6, ema_max=4  → abs_max=6 → scale=6/127
    EmaCalibrator cal(0.0f);
    auto t = Tensor::create(make_shape(2), DType::FP32);
    t.data<float>()[0] = -6.0f;
    t.data<float>()[1] = 4.0f;
    cal.observe(t);

    auto qp = cal.compute_symmetric();

    EXPECT_NEAR(qp.scale, 6.0f / 127.0f, 1e-6f);
    EXPECT_EQ(qp.zero_point, 0);
}

TEST(EmaCalibrator, Asymmetric_Params) {
    // ema_min=-2, ema_max=6
    // scale = (6 - (-2)) / 255 = 8/255
    // zp = clamp(round(-ema_min / scale) - 128, -128, 127)
    //    = clamp(round(63.75) - 128, -128, 127) = clamp(-64, ...) = -64
    // The -128 offset converts from the UINT8 zero-point convention to INT8,
    // ensuring the full [-128, 127] range maps to [ema_min, ema_max] with no
    // saturation and minimal roundtrip error.
    EmaCalibrator cal(0.0f);
    auto t = Tensor::create(make_shape(2), DType::FP32);
    t.data<float>()[0] = -2.0f;
    t.data<float>()[1] = 6.0f;
    cal.observe(t);

    auto qp = cal.compute_asymmetric();
    const float expected_scale = 8.0f / 255.0f;
    // zp = round(2/(8/255)) - 128 = 64 - 128 = -64
    const int32_t expected_zp = -64;

    EXPECT_NEAR(qp.scale, expected_scale, 1e-6f);
    EXPECT_EQ(qp.zero_point, expected_zp);
}

// ---------------------------------------------------------------------------
// quantize_activation / dequantize_activation tests
// ---------------------------------------------------------------------------

TEST(QuantizeActivation, SymmetricZeroPoint) {
    // zero_point=0 must produce the same result as quantize_symmetric
    auto src = Tensor::create(make_shape(4), DType::FP32);
    float *p = src.data<float>();
    p[0] = 1.27f;
    p[1] = -1.27f;
    p[2] = 0.0f;
    p[3] = 0.635f;

    QuantizationParams qp{0.01f, 0}; // scale=0.01, zp=0

    auto dst_asym = Tensor::create(make_shape(4), DType::INT8);
    auto dst_sym = Tensor::create(make_shape(4), DType::INT8);

    quantize_activation(src, dst_asym, qp);
    quantize_symmetric(src, dst_sym, qp);

    for (int i = 0; i < 4; ++i)
        EXPECT_EQ(dst_asym.data<int8_t>()[i], dst_sym.data<int8_t>()[i])
            << "mismatch at i=" << i;
}

TEST(QuantizeActivation, AsymmetricRoundtrip) {
    // Quantize then dequantize; max absolute error must be < scale.
    const float scale = 8.0f / 255.0f;
    const int32_t zp =
        -64; // as computed in Asymmetric_Params test (INT8 convention)
    QuantizationParams qp{scale, zp};

    const int N = 16;
    auto src = Tensor::create(make_shape(N), DType::FP32);
    float *ps = src.data<float>();
    // Values in [-2, 6]
    for (int i = 0; i < N; ++i)
        ps[i] = -2.0f + static_cast<float>(i) * (8.0f / (N - 1));

    auto quant = Tensor::create(make_shape(N), DType::INT8);
    auto recon = Tensor::create(make_shape(N), DType::FP32);

    quantize_activation(src, quant, qp);
    dequantize_activation(quant, recon, qp);

    const float *pr = recon.data<float>();
    for (int i = 0; i < N; ++i) {
        float err = std::abs(pr[i] - ps[i]);
        EXPECT_LT(err, scale) << "roundtrip error too large at i=" << i
                              << " original=" << ps[i] << " recon=" << pr[i];
    }
}

TEST(QuantizeActivation, Saturation_ClampsBelowMinus128) {
    QuantizationParams qp{1.0f, 0};
    auto src = Tensor::create(make_shape(1), DType::FP32);
    src.data<float>()[0] = -1000.0f;

    auto dst = Tensor::create(make_shape(1), DType::INT8);
    quantize_activation(src, dst, qp);

    EXPECT_EQ(dst.data<int8_t>()[0], static_cast<int8_t>(-128));
}

TEST(QuantizeActivation, Saturation_ClampsAbove127) {
    QuantizationParams qp{1.0f, 0};
    auto src = Tensor::create(make_shape(1), DType::FP32);
    src.data<float>()[0] = 1000.0f;

    auto dst = Tensor::create(make_shape(1), DType::INT8);
    quantize_activation(src, dst, qp);

    EXPECT_EQ(dst.data<int8_t>()[0], static_cast<int8_t>(127));
}

TEST(DequantizeActivation, CorrectFormula) {
    // dst[i] = (src[i] - zero_point) * scale
    const float scale = 0.1f;
    const int32_t zp = 10;
    QuantizationParams qp{scale, zp};

    auto src = Tensor::create(make_shape(3), DType::INT8);
    int8_t *ps = src.data<int8_t>();
    ps[0] = 10; // (10 - 10) * 0.1 = 0
    ps[1] = 20; // (20 - 10) * 0.1 = 1
    ps[2] = 0;  // (0  - 10) * 0.1 = -1

    auto dst = Tensor::create(make_shape(3), DType::FP32);
    dequantize_activation(src, dst, qp);

    EXPECT_NEAR(dst.data<float>()[0], 0.0f, 1e-6f);
    EXPECT_NEAR(dst.data<float>()[1], 1.0f, 1e-6f);
    EXPECT_NEAR(dst.data<float>()[2], -1.0f, 1e-6f);
}
