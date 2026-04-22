add_library(engine STATIC)

# Subdirectories add sources to the engine target
add_subdirectory(tensor)

target_include_directories(engine PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(engine PUBLIC cxx_std_17)

# Kernel libraries (each is its own static lib depending on engine)
add_subdirectory(kernels/gemm)
