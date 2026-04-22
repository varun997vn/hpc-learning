// Microbenchmark for gemm_fp32_naive.
//
// This file is wired into the build when benchmarks/CMakeLists.txt exists
// (added by the build-system agent in a parallel branch).  Until then it
// compiles independently and is not linked by default.
//
// Usage (once the benchmark target is active):
//   cmake --preset release && cmake --build build/release --target bench_gemm_naive
//   build/release/benchmarks/microbench/bench_gemm_naive \
//       --benchmark_min_time=1 --benchmark_repetitions=5
//
// Sizes follow the project mandate: powers-of-two up to 4096 plus the
// non-power-of-two sizes 384 and 768 that exercise non-aligned edge cases.

#include "engine/kernels/gemm.hpp"
#include "engine/tensor.hpp"

#include <benchmark/benchmark.h>
#include <cstdlib>

using namespace ie;
using namespace ie::kernels;

static void fill_random(Tensor& t) {
    float* p = t.data<float>();
    for (int64_t i = 0; i < t.numel(); ++i)
        p[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

// Square GEMM: C[N,N] = A[N,N] * B[N,N]
static void BM_GemmNaive_Square(benchmark::State& state) {
    const int64_t N = state.range(0);
    auto A = Tensor::create(make_shape(N, N), DType::FP32);
    auto B = Tensor::create(make_shape(N, N), DType::FP32);
    auto C = Tensor::create(make_shape(N, N), DType::FP32);
    fill_random(A);
    fill_random(B);
    fill_random(C);

    for (auto _ : state) {
        gemm_fp32_naive(A, B, C, 1.0f, 0.0f);
        benchmark::ClobberMemory();
    }

    // 2*N^3 FLOPs per matrix multiply (N^3 multiplies + N^3 adds)
    const double flops = 2.0 * static_cast<double>(N) * static_cast<double>(N) *
                         static_cast<double>(N) * static_cast<double>(state.iterations());
    state.counters["GFLOPS"] =
        benchmark::Counter(flops, benchmark::Counter::kIsRate, benchmark::Counter::OneK::kIs1000);
    state.counters["GFLOPS"] /= 1e9;
}

// clang-format off
BENCHMARK(BM_GemmNaive_Square)
    ->Arg(64)->Arg(128)->Arg(256)->Arg(512)->Arg(384)->Arg(768)
    ->Unit(benchmark::kMillisecond);
// clang-format on

// Larger sizes are slow for the naive kernel; gate them behind a separate
// registration so nightly runs can opt in with --benchmark_filter=Large.
static void BM_GemmNaive_Square_Large(benchmark::State& state) {
    const int64_t N = state.range(0);
    auto A = Tensor::create(make_shape(N, N), DType::FP32);
    auto B = Tensor::create(make_shape(N, N), DType::FP32);
    auto C = Tensor::create(make_shape(N, N), DType::FP32);
    fill_random(A);
    fill_random(B);
    fill_random(C);

    for (auto _ : state) {
        gemm_fp32_naive(A, B, C, 1.0f, 0.0f);
        benchmark::ClobberMemory();
    }

    const double flops = 2.0 * static_cast<double>(N) * static_cast<double>(N) *
                         static_cast<double>(N) * static_cast<double>(state.iterations());
    state.counters["GFLOPS"] =
        benchmark::Counter(flops, benchmark::Counter::kIsRate, benchmark::Counter::OneK::kIs1000);
    state.counters["GFLOPS"] /= 1e9;
}

// clang-format off
BENCHMARK(BM_GemmNaive_Square_Large)
    ->Arg(1024)->Arg(2048)->Arg(4096)
    ->Unit(benchmark::kMillisecond);
// clang-format on

BENCHMARK_MAIN();
