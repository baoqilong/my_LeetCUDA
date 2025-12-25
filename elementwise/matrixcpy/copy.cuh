#ifndef F32COPY_CUH
#define F32COPY_CUH

#include <cuda_runtime.h>

// Matrix addition kernel device function
__global__ void copy(float* A, float* B,int N);

// Wrapper function for launching the kernel
void copy_matrix_kernel(float* d_A, float* d_B,int N);

#endif // F32COPY_CUH
