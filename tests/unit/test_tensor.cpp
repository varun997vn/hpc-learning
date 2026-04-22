#include "engine/tensor.hpp"
#include <gtest/gtest.h>
#include <cstdint>
#include <stdexcept>

using namespace ie;

// ---- Creation and shape --------------------------------------------------

TEST(TensorCreate, BasicFP32) {
    auto t = Tensor::create(make_shape(2, 3), DType::FP32);
    ASSERT_TRUE(t.valid());
    EXPECT_EQ(t.shape().rank, 2);
    EXPECT_EQ(t.shape()[0], 2);
    EXPECT_EQ(t.shape()[1], 3);
    EXPECT_EQ(t.dtype(), DType::FP32);
    EXPECT_EQ(t.numel(), 6);
}

TEST(TensorCreate, Shapes1DThrough4D) {
    EXPECT_EQ(Tensor::create(make_shape(4),          DType::FP32).numel(), 4);
    EXPECT_EQ(Tensor::create(make_shape(2, 4),       DType::FP32).numel(), 8);
    EXPECT_EQ(Tensor::create(make_shape(2, 3, 4),    DType::FP32).numel(), 24);
    EXPECT_EQ(Tensor::create(make_shape(2, 3, 4, 5), DType::FP32).numel(), 120);
}

// ---- 64-byte alignment ---------------------------------------------------

TEST(TensorAlignment, OwnedFP32Is64ByteAligned) {
    auto t = Tensor::create(make_shape(1024), DType::FP32);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(t.data<float>()) % 64, 0u);
}

TEST(TensorAlignment, SmallTensorStillAligned) {
    auto t = Tensor::create(make_shape(1), DType::FP32);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(t.data<float>()) % 64, 0u);
}

TEST(TensorAlignment, INT8TensorAligned) {
    auto t = Tensor::create(make_shape(128), DType::INT8);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(t.data<int8_t>()) % 64, 0u);
}

// ---- DType ---------------------------------------------------------------

TEST(TensorDType, AllDtypesCreatable) {
    EXPECT_EQ(Tensor::create(make_shape(4), DType::FP32).dtype(),  DType::FP32);
    EXPECT_EQ(Tensor::create(make_shape(4), DType::INT8).dtype(),  DType::INT8);
    EXPECT_EQ(Tensor::create(make_shape(4), DType::INT32).dtype(), DType::INT32);
}

TEST(TensorDType, ReadWriteFP32) {
    auto t = Tensor::create(make_shape(4), DType::FP32);
    float* p = t.data<float>();
    p[0] = 1.0f; p[1] = 2.0f; p[2] = 3.0f; p[3] = 4.0f;
    EXPECT_FLOAT_EQ(t.data<float>()[0], 1.0f);
    EXPECT_FLOAT_EQ(t.data<float>()[3], 4.0f);
}

TEST(TensorDType, ReadWriteINT8) {
    auto t = Tensor::create(make_shape(4), DType::INT8);
    int8_t* p = t.data<int8_t>();
    p[0] = -1; p[1] = 127;
    EXPECT_EQ(t.data<int8_t>()[0], -1);
    EXPECT_EQ(t.data<int8_t>()[1], 127);
}

// ---- Move semantics ------------------------------------------------------

TEST(TensorMove, MovePreservesDataPointer) {
    auto t = Tensor::create(make_shape(1024, 1024), DType::FP32);
    const void* original = t.data<float>();
    auto t2 = std::move(t);
    EXPECT_EQ(static_cast<const void*>(t2.data<float>()), original);
    EXPECT_FALSE(t.valid());
}

TEST(TensorMove, MoveAssignmentWorks) {
    auto t = Tensor::create(make_shape(4), DType::FP32);
    t.data<float>()[0] = 99.0f;
    Tensor t2;
    t2 = std::move(t);
    EXPECT_FLOAT_EQ(t2.data<float>()[0], 99.0f);
    EXPECT_FALSE(t.valid());
}

// ---- Clone ---------------------------------------------------------------

TEST(TensorClone, CloneIsDeepCopy) {
    auto t = Tensor::create(make_shape(4), DType::FP32);
    t.data<float>()[0] = 7.0f;
    auto c = t.clone();
    // Mutation to clone does not affect original
    c.data<float>()[0] = 99.0f;
    EXPECT_FLOAT_EQ(t.data<float>()[0], 7.0f);
    EXPECT_FLOAT_EQ(c.data<float>()[0], 99.0f);
}

TEST(TensorClone, CloneMetadataMatches) {
    auto t = Tensor::create(make_shape(3, 4), DType::INT8);
    auto c = t.clone();
    EXPECT_EQ(c.shape(), t.shape());
    EXPECT_EQ(c.dtype(), t.dtype());
    EXPECT_EQ(c.ownership(), Tensor::Ownership::Owned);
}

TEST(TensorClone, CloneIsAligned) {
    auto t = Tensor::create(make_shape(7), DType::FP32);
    auto c = t.clone();
    EXPECT_EQ(reinterpret_cast<uintptr_t>(c.data<float>()) % 64, 0u);
}

// ---- View ----------------------------------------------------------------

TEST(TensorView, ViewSharesUnderlyingData) {
    auto parent = Tensor::create(make_shape(6), DType::FP32);
    parent.data<float>()[0] = 42.0f;
    parent.data<float>()[2] = 13.0f;

    auto v = Tensor::view(parent, make_shape(3), 0);
    EXPECT_FLOAT_EQ(v.data<float>()[0], 42.0f);
    EXPECT_FLOAT_EQ(v.data<float>()[2], 13.0f);
}

TEST(TensorView, ViewWithOffsetCorrect) {
    auto parent = Tensor::create(make_shape(6), DType::FP32);
    for (int i = 0; i < 6; ++i)
        parent.data<float>()[i] = static_cast<float>(i);

    auto v = Tensor::view(parent, make_shape(3), 3);
    EXPECT_FLOAT_EQ(v.data<float>()[0], 3.0f);
    EXPECT_FLOAT_EQ(v.data<float>()[2], 5.0f);
}

TEST(TensorView, MutationThroughViewVisibleInParent) {
    auto parent = Tensor::create(make_shape(4), DType::FP32);
    auto v = Tensor::view(parent, make_shape(2), 1);
    v.data<float>()[0] = 55.0f;
    EXPECT_FLOAT_EQ(parent.data<float>()[1], 55.0f);
}

TEST(TensorView, OwnershipIsView) {
    auto parent = Tensor::create(make_shape(4), DType::FP32);
    auto v = Tensor::view(parent, make_shape(2), 0);
    EXPECT_EQ(v.ownership(), Tensor::Ownership::View);
}

TEST(TensorView, ParentDestroyedViewDataStillValid) {
    // shared_ptr keeps data alive even after the parent Tensor object is gone
    auto v = Tensor::view(
        Tensor::create(make_shape(4), DType::FP32),
        make_shape(2), 0);
    EXPECT_TRUE(v.valid()); // data stays alive via shared_ptr ref-count
}

// ---- External ------------------------------------------------------------

TEST(TensorExternal, DoesNotOwnMemory) {
    float buf[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto t = Tensor::external(buf, make_shape(4), DType::FP32);
    EXPECT_EQ(t.ownership(), Tensor::Ownership::External);
    EXPECT_EQ(static_cast<void*>(t.data<float>()), static_cast<void*>(buf));
}

TEST(TensorExternal, MutationVisibleInOriginalBuffer) {
    float buf[2] = {0.0f, 0.0f};
    auto t = Tensor::external(buf, make_shape(2), DType::FP32);
    t.data<float>()[0] = 77.0f;
    EXPECT_FLOAT_EQ(buf[0], 77.0f);
}

// ---- Reshape -------------------------------------------------------------

TEST(TensorReshape, FlattenSucceeds) {
    auto t = Tensor::create(make_shape(4, 3), DType::FP32);
    auto r = t.reshape(make_shape(12));
    EXPECT_EQ(r.numel(), 12);
    EXPECT_EQ(r.shape().rank, 1);
}

TEST(TensorReshape, ElementCountMismatchThrows) {
    auto t = Tensor::create(make_shape(4, 3), DType::FP32);
    EXPECT_THROW(t.reshape(make_shape(10)), std::invalid_argument);
}

TEST(TensorReshape, DataPreservedAfterReshape) {
    auto t = Tensor::create(make_shape(6), DType::FP32);
    for (int i = 0; i < 6; ++i)
        t.data<float>()[i] = static_cast<float>(i);
    auto r = t.reshape(make_shape(2, 3));
    EXPECT_FLOAT_EQ(r.data<float>()[0], 0.0f);
    EXPECT_FLOAT_EQ(r.data<float>()[5], 5.0f);
}
