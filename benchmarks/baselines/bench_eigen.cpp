#include <Eigen/Dense>
#include <benchmark/benchmark.h>

static void BM_Eigen_MatMul(benchmark::State& state) {
    const int N = static_cast<int>(state.range(0));
    Eigen::MatrixXf A = Eigen::MatrixXf::Random(N, N);
    Eigen::MatrixXf B = Eigen::MatrixXf::Random(N, N);
    Eigen::MatrixXf C(N, N);

    for (auto _ : state) {
        C.noalias() = A * B;
        benchmark::DoNotOptimize(C.data());
    }

    const double flops = 2.0 * N * N * N;
    state.counters["GFLOPS"] =
        benchmark::Counter(flops, benchmark::Counter::kIsRate, benchmark::Counter::kIs1000);
}

BENCHMARK(BM_Eigen_MatMul)
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
