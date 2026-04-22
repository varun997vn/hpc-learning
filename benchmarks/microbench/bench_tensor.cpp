#include "engine/tensor.hpp"

#include <benchmark/benchmark.h>

// Benchmark Tensor::create for various square matrix sizes.
// Measures the cost of 64-byte-aligned allocation + shape bookkeeping.
static void BM_TensorCreate(benchmark::State& state) {
    const int64_t n = state.range(0);
    for (auto _ : state) {
        auto t = ie::Tensor::create(ie::make_shape(n, n), ie::DType::FP32);
        benchmark::DoNotOptimize(t.data<float>());
    }
    state.SetBytesProcessed(state.iterations() * n * n * 4);
}
BENCHMARK(BM_TensorCreate)->Range(64, 4096)->Unit(benchmark::kMicrosecond);

// Benchmark Tensor::clone — allocate + memcpy of an n×n FP32 matrix.
static void BM_TensorClone(benchmark::State& state) {
    const int64_t n = state.range(0);
    auto src = ie::Tensor::create(ie::make_shape(n, n), ie::DType::FP32);
    for (auto _ : state) {
        auto dst = src.clone();
        benchmark::DoNotOptimize(dst.data<float>());
    }
    state.SetBytesProcessed(state.iterations() * n * n * 4);
}
BENCHMARK(BM_TensorClone)->Range(64, 4096)->Unit(benchmark::kMicrosecond);
