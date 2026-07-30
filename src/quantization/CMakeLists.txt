add_library(ie_quant STATIC
    calibrator.cpp
    per_channel.cpp
    activation_calib.cpp
)

target_include_directories(ie_quant PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(ie_quant PUBLIC engine)
target_compile_features(ie_quant PUBLIC cxx_std_17)
