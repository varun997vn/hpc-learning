#pragma once
#include <array>
#include <cstdint>

namespace ie {

enum class DType { FP32, FP16, INT8, INT32 };

inline size_t dtype_size(DType dt) noexcept {
    switch (dt) {
        case DType::FP32:  return 4;
        case DType::FP16:  return 2;
        case DType::INT8:  return 1;
        case DType::INT32: return 4;
    }
    return 0;
}

struct Shape {
    static constexpr int kMaxDims = 8;
    std::array<int64_t, kMaxDims> dims{};
    int rank = 0;

    int64_t operator[](int i) const noexcept { return dims[static_cast<size_t>(i)]; }
    int64_t& operator[](int i) noexcept { return dims[static_cast<size_t>(i)]; }

    int64_t numel() const noexcept {
        if (rank == 0) return 0;
        int64_t n = 1;
        for (int i = 0; i < rank; ++i)
            n *= dims[static_cast<size_t>(i)];
        return n;
    }

    bool operator==(const Shape& o) const noexcept {
        if (rank != o.rank) return false;
        for (int i = 0; i < rank; ++i)
            if (dims[static_cast<size_t>(i)] != o.dims[static_cast<size_t>(i)]) return false;
        return true;
    }
    bool operator!=(const Shape& o) const noexcept { return !(*this == o); }
};

template<typename... Dims>
Shape make_shape(Dims... d) {
    static_assert(sizeof...(d) <= static_cast<size_t>(Shape::kMaxDims), "Too many dimensions");
    Shape s;
    s.rank = static_cast<int>(sizeof...(d));
    int i = 0;
    ((s.dims[static_cast<size_t>(i++)] = static_cast<int64_t>(d)), ...);
    return s;
}

} // namespace ie
