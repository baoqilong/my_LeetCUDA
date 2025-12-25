cmake_minimum_required(VERSION 3.16)
project(CUDA_INFO LANGUAGES CUDA CXX)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
add_executable(cuda_info cudainfo.cu)
# target_link_libraries(cuda_info ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY})