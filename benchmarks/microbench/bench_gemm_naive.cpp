#include <benchmark/benchmark.h>
#include <vector>

// Naive triple-loop GEMM: C = A * B (square, row-major, float).
// This is the reference baseline that all optimized variants must beat.
static void BM_Gemm_Naive(benchmark::State& state) {
    const int N = static_cast<int>(state.range(0));
    std::vector<float> A(N * N, 1.0f);
    std::vector<float> B(N * N, 1.0f);
    std::vector<float> C(N * N, 0.0f);

    for (auto _ : state) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                float acc = 0.0f;
                for (int k = 0; k < N; ++k) {
                    acc += A[i * N + k] * B[k * N + j];
                }
                C[i * N + j] = acc;
            }
        }
        benchmark::DoNotOptimize(C.data());
    }

    const double flops = 2.0 * N * N * N;
    state.counters["GFLOPS"] =
        benchmark::Counter(flops, benchmark::Counter::kIsRate, benchmark::Counter::kIs1000);
}

BENCHMARK(BM_Gemm_Naive)
    ->Args({64})
    ->Args({128})
    ->Args({256})
    ->Args({512})
    ->Args({1024})
    ->Args({2048})
    ->Args({384})
    ->Args({768})
    ->Unit(benchmark::kMillisecond);
