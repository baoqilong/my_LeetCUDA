cmake_minimum_required(VERSION 3.16)
project(MATRIX_COPY LANGUAGES CUDA CXX)

find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})

# Main executable (example/demo)
add_executable(copy_main main.cu copy.cu)

# Test executable
add_executable(copy_test test.cpp copy.cu)

# Enable debug symbols for CUDA
set_property(TARGET copy_main PROPERTY CUDA_ARCHITECTURES "75;80;86")
set_property(TARGET copy_test PROPERTY CUDA_ARCHITECTURES "75;80;86")

target_compile_options(copy_main PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g -G>)
target_compile_options(copy_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g -G>)

# Enable C++17 for test
target_compile_features(copy_test PRIVATE cxx_std_17)
