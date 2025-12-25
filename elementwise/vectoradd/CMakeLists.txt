cmake_minimum_required(VERSION 3.16)
project(F32VECTOR_ADD LANGUAGES CUDA CXX)

find_package(CUDA REQUIRED)
add_executable(f32add f32add.cu)

# Enable debug symbols for CUDA
set_property(TARGET f32add PROPERTY CUDA_ARCHITECTURES "75;80;86")  # Adjust based on your GPU
target_compile_options(f32add PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g -G -lineinfo>)