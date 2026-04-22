# Hardware probe — measures peak DRAM bandwidth; run once per machine.

add_executable(bench_peak bench_peak.cpp)
target_link_libraries(bench_peak PRIVATE
    benchmark::benchmark_main
    engine
)
set_target_properties(bench_peak PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/benchmarks/hardware_probe"
)
