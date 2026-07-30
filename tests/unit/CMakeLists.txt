add_executable(test_tensor test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE engine GTest::gtest_main)

add_executable(test_gemm test_gemm.cpp)
target_link_libraries(test_gemm PRIVATE ie_gemm GTest::gtest_main)

add_executable(test_calibrator test_calibrator.cpp)
target_link_libraries(test_calibrator PRIVATE ie_quant GTest::gtest_main)

add_executable(test_perchannel test_perchannel.cpp)
target_link_libraries(test_perchannel PRIVATE ie_quant GTest::gtest_main)

add_executable(test_gemm_int8 test_gemm_int8.cpp)
target_link_libraries(test_gemm_int8 PRIVATE ie_gemm GTest::gtest_main)

add_executable(test_activation_quant test_activation_quant.cpp)
target_link_libraries(test_activation_quant PRIVATE ie_quant GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(test_tensor)
gtest_discover_tests(test_gemm)
gtest_discover_tests(test_calibrator)
gtest_discover_tests(test_perchannel)
gtest_discover_tests(test_gemm_int8)
gtest_discover_tests(test_activation_quant)
