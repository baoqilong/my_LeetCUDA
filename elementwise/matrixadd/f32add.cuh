#ifndef F32ADD_CUH
#define F32ADD_CUH

#include <cuda_runtime.h>

// Matrix addition kernel device function
__global__ void add(float* A, float* B, float* C, int rows, int cols);

// Wrapper function for launching the kernel
void cuda_matrix_add(float* d_A, float* d_B, float* d_C, int rows, int cols);

#endif // F32ADD_CUH
