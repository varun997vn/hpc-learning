add_executable(test_tensor test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE engine GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(test_tensor)
