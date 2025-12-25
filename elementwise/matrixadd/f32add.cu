#include "f32add.cuh"

__global__ void add(float* A, float* B, float* C, int rows, int cols) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < rows && col < cols) {
        C[row * cols + col] = A[row * cols + col] + B[row * cols + col];
    }
}

void cuda_matrix_add(float* d_A, float* d_B, float* d_C, int rows, int cols) {
    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid(
        (cols + threads_per_block.x - 1) / threads_per_block.x,
        (rows + threads_per_block.y - 1) / threads_per_block.y
    );
    add<<<blocks_per_grid, threads_per_block>>>(d_A, d_B, d_C, rows, cols);
}
