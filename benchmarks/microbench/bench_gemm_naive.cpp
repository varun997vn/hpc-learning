#include "engine/kernels/gemm.hpp"
#include "engine/tensor.hpp"

#include <benchmark/benchmark.h>
#include <random>

using namespace ie;
using namespace ie::kernels;

static void fill_random(Tensor& t, uint32_t seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    float* p = t.data<float>();
    for (int64_t i = 0; i < t.numel(); ++i)
        p[i] = dist(rng);
}

// Benchmark sizes: {64,128,256,512,1024,2048,4096} + {384,768} non-power-of-2.
// We skip 2048 and 4096 for naive to avoid multi-minute runs in CI;
// they are present in the tiled benchmark.
static void BM_GemmNaive_Square(benchmark::State& state) {
    const int N = static_cast<int>(state.range(0));
    auto A = Tensor::create(make_shape(N, N), DType::FP32);
    auto B = Tensor::create(make_shape(N, N), DType::FP32);
    auto C = Tensor::create(make_shape(N, N), DType::FP32);
    fill_random(A, 1);
    fill_random(B, 2);
    fill_random(C, 3);

    for (auto _ : state) {
        gemm_fp32_naive(A, B, C, 1.0f, 0.0f);
        benchmark::ClobberMemory();
    }

    const double flops =
        2.0 * static_cast<double>(N) * static_cast<double>(N) * static_cast<double>(N);
    state.counters["GFLOPS"] =
        benchmark::Counter(flops, benchmark::Counter::kIsRate, benchmark::Counter::kIs1000);
    state.counters["N"] = static_cast<double>(N);
}
BENCHMARK(BM_GemmNaive_Square)
    ->Args({64})
    ->Args({128})
    ->Args({256})
    ->Args({384})
    ->Args({512})
    ->Args({768})
    ->Unit(benchmark::kMillisecond);

static void BM_GemmTiled_Square(benchmark::State& state) {
    const int N = static_cast<int>(state.range(0));
    auto A = Tensor::create(make_shape(N, N), DType::FP32);
    auto B = Tensor::create(make_shape(N, N), DType::FP32);
    auto C = Tensor::create(make_shape(N, N), DType::FP32);
    fill_random(A, 1);
    fill_random(B, 2);
    fill_random(C, 3);

    // Default TilingConfig {mc=64, nc=64, kc=64} — fits 3*64*64*4 = 48 KB in L1.
    const TilingConfig cfg{};

    for (auto _ : state) {
        gemm_fp32_tiled(A, B, C, cfg, 1.0f, 0.0f);
        benchmark::ClobberMemory();
    }

    const double flops =
        2.0 * static_cast<double>(N) * static_cast<double>(N) * static_cast<double>(N);
    state.counters["GFLOPS"] =
        benchmark::Counter(flops, benchmark::Counter::kIsRate, benchmark::Counter::kIs1000);
    state.counters["N"] = static_cast<double>(N);
}
BENCHMARK(BM_GemmTiled_Square)
    ->Args({64})
    ->Args({128})
    ->Args({256})
    ->Args({384})
    ->Args({512})
    ->Args({768})
    ->Args({1024})
    ->Args({2048})
    ->Args({4096})
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
