add_executable(test_gemm_integration test_gemm_shapes.cpp)
target_link_libraries(test_gemm_integration PRIVATE ie_gemm GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(test_gemm_integration PROPERTIES TIMEOUT 30)
