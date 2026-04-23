add_library(ie_gemm STATIC
    gemm_naive.cpp
    gemm_parallel.cpp
)

target_include_directories(ie_gemm PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(ie_gemm PUBLIC engine)
target_compile_features(ie_gemm PUBLIC cxx_std_17)

if(ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
    target_link_libraries(ie_gemm PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(ie_gemm PUBLIC ENABLE_OPENMP)
endif()
