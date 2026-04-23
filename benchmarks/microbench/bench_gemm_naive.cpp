#include "engine/kernels/gemm.hpp"
#include "engine/tensor.hpp"

#include <benchmark/benchmark.h>
#include <random>

using namespace ie;
using namespace ie::kernels;

// ---- Helpers ---------------------------------------------------------------

static Tensor make_rand_fp32(int64_t rows, int64_t cols, uint32_t seed = 0) {
    auto t = Tensor::create(make_shape(rows, cols), DType::FP32);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    float* p = t.data<float>();
    for (int64_t i = 0; i < rows * cols; ++i)
        p[i] = dist(rng);
    return t;
}

// ---- gemm_fp32_naive -------------------------------------------------------

static void BM_GemmNaive_Square(benchmark::State& state) {
    const int64_t N = state.range(0);
    auto A = make_rand_fp32(N, N, 1);
    auto B = make_rand_fp32(N, N, 2);
    auto C = Tensor::create(make_shape(N, N), DType::FP32);

    for (auto _ : state) {
        // Reset C each iteration so beta=0 does not accumulate across iterations
        float* cp = C.data<float>();
        for (int64_t i = 0; i < N * N; ++i)
            cp[i] = 0.0f;
        gemm_fp32_naive(A, B, C);
        benchmark::ClobberMemory();
    }

    const double flops =
        2.0 * static_cast<double>(N) * static_cast<double>(N) * static_cast<double>(N);
    state.counters["GFLOPS"] =
        benchmark::Counter(flops, benchmark::Counter::kIsRate, benchmark::Counter::kIs1000);
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(3 * N * N * sizeof(float)));
}

// ---- gemm_fp32_simd --------------------------------------------------------

static void BM_GemmSimd_Square(benchmark::State& state) {
    const int64_t N = state.range(0);
    auto A = make_rand_fp32(N, N, 3);
    auto B = make_rand_fp32(N, N, 4);
    auto C = Tensor::create(make_shape(N, N), DType::FP32);

    for (auto _ : state) {
        float* cp = C.data<float>();
        for (int64_t i = 0; i < N * N; ++i)
            cp[i] = 0.0f;
        gemm_fp32_simd(A, B, C);
        benchmark::ClobberMemory();
    }

    const double flops =
        2.0 * static_cast<double>(N) * static_cast<double>(N) * static_cast<double>(N);
    state.counters["GFLOPS"] =
        benchmark::Counter(flops, benchmark::Counter::kIsRate, benchmark::Counter::kIs1000);
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(3 * N * N * sizeof(float)));
}

// ---- Registration ----------------------------------------------------------
// Standard size range per CLAUDE.md: {64, 128, 256, 512, 1024, 2048, 4096} + {384, 768}

// clang-format off
BENCHMARK(BM_GemmNaive_Square)
    ->Arg(64)->Arg(128)->Arg(256)->Arg(384)->Arg(512)->Arg(768)->Arg(1024)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_GemmSimd_Square)
    ->Arg(64)->Arg(128)->Arg(256)->Arg(384)->Arg(512)->Arg(768)->Arg(1024)
    ->Unit(benchmark::kMillisecond);
// clang-format on

BENCHMARK_MAIN();
