// Microbenchmarks for gemm_fp32_naive and gemm_fp32_parallel.
//
// Usage (release preset for meaningful numbers):
//   cmake --preset release && cmake --build build/release --target bench_gemm
//   build/release/benchmarks/microbench/bench_gemm \
//       --benchmark_min_time=1 --benchmark_repetitions=5
//
// Sizes follow the project mandate: powers-of-two up to 4096 plus the
// non-power-of-two sizes 384 and 768 that exercise non-aligned edge cases.
//
// Thread-scaling sweep runs 1024x1024 with 1/2/4/8 threads to characterise
// multi-core scaling efficiency.

#include "engine/kernels/gemm.hpp"
#include "engine/tensor.hpp"

#include <benchmark/benchmark.h>
#include <cstdlib>
#include <random>

using namespace ie;
using namespace ie::kernels;

// ---- Helpers ---------------------------------------------------------------

static void fill_random(Tensor& t) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    float* p = t.data<float>();
    for (int64_t i = 0; i < t.numel(); ++i)
        p[i] = dist(rng);
}

static double gflops_per_iter(int64_t N) {
    // 2 * N^3: N^3 multiplies + N^3 adds
    return 2.0 * static_cast<double>(N) * static_cast<double>(N) * static_cast<double>(N);
}

// ---- BM_GemmNaive_Square ----------------------------------------------------

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

    const double flops = gflops_per_iter(N) * static_cast<double>(state.iterations());
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

    const double flops = gflops_per_iter(N) * static_cast<double>(state.iterations());
    state.counters["GFLOPS"] =
        benchmark::Counter(flops, benchmark::Counter::kIsRate, benchmark::Counter::OneK::kIs1000);
    state.counters["GFLOPS"] /= 1e9;
}

// clang-format off
BENCHMARK(BM_GemmNaive_Square_Large)
    ->Arg(1024)->Arg(2048)->Arg(4096)
    ->Unit(benchmark::kMillisecond);
// clang-format on

// ---- BM_GemmParallel_Square -------------------------------------------------
// Benchmarks gemm_fp32_parallel with default TilingConfig and OMP_NUM_THREADS.
// Run across the same full size list as naive for a direct comparison.

static void BM_GemmParallel_Square(benchmark::State& state) {
    const int64_t N = state.range(0);
    auto A = Tensor::create(make_shape(N, N), DType::FP32);
    auto B = Tensor::create(make_shape(N, N), DType::FP32);
    auto C = Tensor::create(make_shape(N, N), DType::FP32);
    fill_random(A);
    fill_random(B);
    fill_random(C);

    for (auto _ : state) {
        gemm_fp32_parallel(A, B, C, TilingConfig{}, 0, 1.0f, 0.0f);
        benchmark::ClobberMemory();
    }

    const double flops = gflops_per_iter(N) * static_cast<double>(state.iterations());
    state.counters["GFLOPS"] =
        benchmark::Counter(flops, benchmark::Counter::kIsRate, benchmark::Counter::OneK::kIs1000);
    state.counters["GFLOPS"] /= 1e9;
}

// clang-format off
BENCHMARK(BM_GemmParallel_Square)
    ->Arg(64)->Arg(128)->Arg(256)->Arg(512)->Arg(384)->Arg(768)
    ->Arg(1024)->Arg(2048)->Arg(4096)
    ->Unit(benchmark::kMillisecond);
// clang-format on

// ---- BM_GemmParallel_ThreadScaling ------------------------------------------
// Sweeps thread counts 1/2/4/8 at a fixed 1024x1024 problem size.
// The n_threads argument is passed as state.range(1); state.range(0) is N.
//
// Rationale for explicit n_threads rather than ->Threads(): Google Benchmark's
// ->Threads() mechanism pins the *fixture* thread count, which conflicts with
// omp_set_num_threads().  We pass the thread count as an Arg so that each
// benchmark iteration calls gemm_fp32_parallel with a deterministic thread
// count independent of the benchmark framework's own threading.

static void BM_GemmParallel_ThreadScaling(benchmark::State& state) {
    const int64_t N = state.range(0);
    const int n_threads = static_cast<int>(state.range(1));

    auto A = Tensor::create(make_shape(N, N), DType::FP32);
    auto B = Tensor::create(make_shape(N, N), DType::FP32);
    auto C = Tensor::create(make_shape(N, N), DType::FP32);
    fill_random(A);
    fill_random(B);
    fill_random(C);

    for (auto _ : state) {
        gemm_fp32_parallel(A, B, C, TilingConfig{}, n_threads, 1.0f, 0.0f);
        benchmark::ClobberMemory();
    }

    const double flops = gflops_per_iter(N) * static_cast<double>(state.iterations());
    state.counters["GFLOPS"] =
        benchmark::Counter(flops, benchmark::Counter::kIsRate, benchmark::Counter::OneK::kIs1000);
    state.counters["GFLOPS"] /= 1e9;
    state.counters["threads"] = static_cast<double>(n_threads);
}

// clang-format off
BENCHMARK(BM_GemmParallel_ThreadScaling)
    ->Args({1024, 1})
    ->Args({1024, 2})
    ->Args({1024, 4})
    ->Args({1024, 8})
    ->Unit(benchmark::kMillisecond);
// clang-format on

BENCHMARK_MAIN();
