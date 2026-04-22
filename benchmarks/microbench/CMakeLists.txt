# Microbenchmarks — per-kernel, per-size
# bench_gemm_naive.cpp will be added by the kernel agent; leave room for it here.

add_executable(bench_tensor bench_tensor.cpp)
target_link_libraries(bench_tensor PRIVATE
    benchmark::benchmark_main
    engine
)
set_target_properties(bench_tensor PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/benchmarks/microbench"
)
