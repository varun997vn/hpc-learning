#include "engine/tensor.hpp"
#include <cstring>
#include <stdexcept>

static_assert(ie::kDefaultAlignment >= 64,
              "Alignment must be at least 64 bytes for AVX-512 and cache-line safety");

namespace ie {

Tensor Tensor::create(Shape shape, DType dtype) {
    Tensor t;
    t.shape_     = shape;
    t.dtype_     = dtype;
    t.ownership_ = Ownership::Owned;
    t.offset_    = 0;
    t.data_      = aligned_alloc_shared(
        kDefaultAlignment,
        static_cast<size_t>(shape.numel()) * dtype_size(dtype));
    return t;
}

Tensor Tensor::view(const Tensor& parent, Shape shape, int64_t offset) {
    if (shape.numel() + offset > parent.numel())
        throw std::out_of_range("Tensor::view: shape + offset exceeds parent numel");
    Tensor t;
    t.data_      = parent.data_;
    t.offset_    = parent.offset_ + offset;
    t.shape_     = shape;
    t.dtype_     = parent.dtype_;
    t.ownership_ = Ownership::View;
    return t;
}

Tensor Tensor::external(void* ptr, Shape shape, DType dtype) {
    Tensor t;
    t.data_      = {ptr, [](void*) {}}; // no-op deleter — caller owns
    t.offset_    = 0;
    t.shape_     = shape;
    t.dtype_     = dtype;
    t.ownership_ = Ownership::External;
    return t;
}

Tensor Tensor::clone() const {
    auto result = Tensor::create(shape_, dtype_);
    size_t nbytes = static_cast<size_t>(numel()) * dtype_size(dtype_);
    std::memcpy(result.data<uint8_t>(), data<uint8_t>(), nbytes);
    return result;
}

Tensor Tensor::reshape(Shape new_shape) const {
    if (new_shape.numel() != numel())
        throw std::invalid_argument("Tensor::reshape: element count mismatch");
    Tensor t;
    t.data_      = data_;
    t.offset_    = offset_;
    t.shape_     = new_shape;
    t.dtype_     = dtype_;
    t.ownership_ = Ownership::View;
    return t;
}

} // namespace ie
