add_executable(bench_tensor bench_tensor.cpp)
target_link_libraries(bench_tensor PRIVATE
    benchmark::benchmark_main
    engine
)
set_target_properties(bench_tensor PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/benchmarks/microbench"
)

add_executable(bench_gemm bench_gemm_naive.cpp)
target_link_libraries(bench_gemm PRIVATE
    benchmark::benchmark_main
    ie_gemm
)
set_target_properties(bench_gemm PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/benchmarks/microbench"
)
