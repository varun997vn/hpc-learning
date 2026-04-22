if(NOT DEFINED ENV{ANDROID_NDK_ROOT})
    message(FATAL_ERROR
        "ANDROID_NDK_ROOT is not set.\n"
        "Install NDK r26+ and export ANDROID_NDK_ROOT=/path/to/ndk.")
endif()

set(ANDROID_NDK $ENV{ANDROID_NDK_ROOT})

set(CMAKE_SYSTEM_NAME      Android)
set(CMAKE_SYSTEM_VERSION   26)
set(CMAKE_ANDROID_ARCH_ABI arm64-v8a)
set(CMAKE_ANDROID_NDK      ${ANDROID_NDK})
set(CMAKE_ANDROID_STL_TYPE c++_static)

# arm64-v8a always has NEON/ASIMD; enable it explicitly
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a+simd")

add_compile_definitions(ENABLE_NEON=1)
