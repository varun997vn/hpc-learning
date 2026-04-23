#include <benchmark/benchmark.h>
#include <cblas.h>
#include <vector>

static void BM_OpenBLAS_Sgemm(benchmark::State& state) {
    const int N = static_cast<int>(state.range(0));
    std::vector<float> A(N * N, 1.0f);
    std::vector<float> B(N * N, 1.0f);
    std::vector<float> C(N * N, 0.0f);

    for (auto _ : state) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0f, A.data(), N, B.data(),
                    N, 0.0f, C.data(), N);
        benchmark::DoNotOptimize(C.data());
    }

    const double flops = 2.0 * N * N * N;
    state.counters["GFLOPS"] =
        benchmark::Counter(flops, benchmark::Counter::kIsRate, benchmark::Counter::kIs1000);
}

BENCHMARK(BM_OpenBLAS_Sgemm)
    ->Args({64})
    ->Args({128})
    ->Args({256})
    ->Args({512})
    ->Args({1024})
    ->Args({2048})
    ->Args({384})
    ->Args({768})
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
