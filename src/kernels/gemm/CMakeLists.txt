add_library(ie_gemm STATIC
    gemm_naive.cpp
    gemm_tiled.cpp
)

target_include_directories(ie_gemm PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(ie_gemm PUBLIC cxx_std_17)
