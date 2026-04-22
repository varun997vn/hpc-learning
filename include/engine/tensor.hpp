#pragma once
#include "engine/allocator.hpp"
#include "engine/types.hpp"
#include <memory>
#include <stdexcept>

namespace ie {

class Tensor {
public:
    enum class Ownership { Owned, View, External };

    // --- Factory methods ---
    static Tensor create(Shape shape, DType dtype);
    static Tensor view(const Tensor& parent, Shape shape, int64_t offset = 0);
    static Tensor external(void* ptr, Shape shape, DType dtype);

    // --- Rule of 5 ---
    Tensor() = default;
    ~Tensor() = default;
    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(Tensor&&) noexcept = default;
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    // Deep copy
    Tensor clone() const;
    // Returns a view with a new shape; throws std::invalid_argument if numel differs
    Tensor reshape(Shape new_shape) const;

    // --- Data access ---
    template<typename T>
    T* data() {
        return reinterpret_cast<T*>(
            static_cast<char*>(data_.get()) +
            offset_ * static_cast<int64_t>(dtype_size(dtype_)));
    }

    template<typename T>
    const T* data() const {
        return reinterpret_cast<const T*>(
            static_cast<const char*>(data_.get()) +
            offset_ * static_cast<int64_t>(dtype_size(dtype_)));
    }

    // --- Metadata ---
    const Shape& shape() const noexcept { return shape_; }
    DType dtype() const noexcept { return dtype_; }
    int64_t numel() const noexcept { return shape_.numel(); }
    Ownership ownership() const noexcept { return ownership_; }
    bool valid() const noexcept { return data_ != nullptr; }

private:
    std::shared_ptr<void> data_;
    int64_t offset_ = 0; // element offset into data_ (not bytes)
    Shape shape_;
    DType dtype_ = DType::FP32;
    Ownership ownership_ = Ownership::Owned;
};

} // namespace ie
