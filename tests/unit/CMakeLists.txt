add_executable(test_tensor test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE engine GTest::gtest_main)

add_executable(test_gemm test_gemm.cpp)
target_link_libraries(test_gemm PRIVATE ie_gemm GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(test_tensor)
gtest_discover_tests(test_gemm)
