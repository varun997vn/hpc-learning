#include "engine/tensor.hpp"

#include <algorithm>
#include <benchmark/benchmark.h>

// Sequential-read bandwidth probe.
// Uses a Tensor-owned buffer so the measurement captures the same allocator
// path that real kernels use.  Buffer sizes from 1 MiB (fits in L3) to 128 MiB
// (exceeds typical LLC → pure DRAM read).
static void BM_MemBandwidth(benchmark::State& state) {
    const int64_t n = state.range(0);
    auto t = ie::Tensor::create(ie::make_shape(n), ie::DType::FP32);
    float* p = t.data<float>();
    std::fill(p, p + n, 1.0f);
    volatile float sink = 0;
    for (auto _ : state) {
        float sum = 0;
        for (int64_t i = 0; i < n; ++i)
            sum += p[i];
        sink = sum;
        benchmark::DoNotOptimize(sink);
    }
    state.SetBytesProcessed(state.iterations() * n * static_cast<int64_t>(sizeof(float)));
    state.counters["GB/s"] =
        benchmark::Counter(static_cast<double>(state.bytes_processed()),
                           benchmark::Counter::kIsRate, benchmark::Counter::kIs1024);
}
BENCHMARK(BM_MemBandwidth)->Range(1 << 20, 1 << 27)->Unit(benchmark::kMillisecond);
