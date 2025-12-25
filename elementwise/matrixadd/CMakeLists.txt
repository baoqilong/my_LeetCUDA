cmake_minimum_required(VERSION 3.16)
project(F32MATRIX_ADD LANGUAGES CUDA CXX)

find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})

# Main executable (example/demo)
add_executable(f32add_main main.cu f32add.cu)

# Test executable
add_executable(f32add_test test.cpp f32add.cu)

# Enable debug symbols for CUDA
set_property(TARGET f32add_main PROPERTY CUDA_ARCHITECTURES "75;80;86")
set_property(TARGET f32add_test PROPERTY CUDA_ARCHITECTURES "75;80;86")

target_compile_options(f32add_main PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g -G>)
target_compile_options(f32add_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g -G>)

# Enable C++17 for test
target_compile_features(f32add_test PRIVATE cxx_std_17)
