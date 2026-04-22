# OpenBLAS baseline — only built when OpenBLAS is found on the host system.
if(OpenBLAS_FOUND)
    add_executable(bench_openblas bench_openblas.cpp)
    target_include_directories(bench_openblas PRIVATE ${OpenBLAS_INCLUDE_DIRS})
    target_link_libraries(bench_openblas PRIVATE
        benchmark::benchmark_main
        ${OpenBLAS_LIBRARIES}
    )
    set_target_properties(bench_openblas PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/benchmarks/baselines"
    )
endif()

# Eigen baseline — always built (header-only, no system dep).
add_executable(bench_eigen bench_eigen.cpp)
target_link_libraries(bench_eigen PRIVATE
    benchmark::benchmark_main
    Eigen3::Eigen
)
set_target_properties(bench_eigen PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/benchmarks/baselines"
)
