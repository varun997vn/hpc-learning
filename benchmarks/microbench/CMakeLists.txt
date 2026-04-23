add_executable(bench_gemm bench_gemm_naive.cpp)
target_link_libraries(bench_gemm PRIVATE ie_gemm engine benchmark::benchmark)
target_compile_features(bench_gemm PRIVATE cxx_std_17)
