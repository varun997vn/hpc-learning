#pragma once
#include <cstdlib>
#include <memory>
#include <new>

namespace ie {

constexpr size_t kDefaultAlignment = 64; // cache-line + AVX-512 compatible

struct AlignedDeleter {
    void operator()(void* ptr) noexcept { free(ptr); }
};

// Allocates nbytes with at least `alignment`-byte alignment.
// alignment must be a power of two and size will be rounded up to a multiple of it.
inline std::shared_ptr<void> aligned_alloc_shared(size_t alignment, size_t nbytes) {
    if (nbytes == 0) nbytes = 1;
    // std::aligned_alloc requires size to be a multiple of alignment
    size_t alloc_size = (nbytes + alignment - 1) & ~(alignment - 1);
    void* ptr = std::aligned_alloc(alignment, alloc_size);
    if (!ptr) throw std::bad_alloc();
    return {ptr, AlignedDeleter{}};
}

} // namespace ie
