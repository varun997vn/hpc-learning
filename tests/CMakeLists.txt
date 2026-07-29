add_subdirectory(unit)

if(EXISTS "${CMAKE_SOURCE_DIR}/tests/correctness/CMakeLists.txt")
    add_subdirectory(correctness)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/tests/integration/CMakeLists.txt")
    add_subdirectory(integration)
endif()
