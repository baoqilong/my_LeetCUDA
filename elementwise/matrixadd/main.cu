#include <iostream>
#include <cuda_runtime.h>
#include "f32add.cuh"

void cpu_matrix_add(const float* A, const float* B, float* C, int rows, int cols) {
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            C[row * cols + col] = A[row * cols + col] + B[row * cols + col];
        }
    }
}

bool check_result(const float* cpu_res, const float* gpu_res, int N) {
    for (int i = 0; i < N; ++i) {
        if (i % 10 == 0 && i != 0) {
            std::cout << "\n" << cpu_res[i] << " ";
        } else {
            std::cout << cpu_res[i] << " ";
        }
        if (std::fabs(cpu_res[i] - gpu_res[i]) > 1e-6f)
            return false;
    }
    return true;
}

int main() {
    int rows = 12;
    int cols = 10;
    int N = rows * cols;

    float *h_A, *h_B, *h_C, *h_D;
    h_A = new float[N];
    h_B = new float[N];
    h_C = new float[N];
    h_D = new float[N];

    // Initialize data
    for (int i = 0; i < N; ++i) {
        h_A[i] = h_B[i] = 1;
        h_C[i] = h_A[i] + h_B[i];
        h_D[i] = 0;
    }

    // GPU computation
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeof(float) * N);
    cudaMalloc(&d_B, sizeof(float) * N);
    cudaMalloc(&d_C, sizeof(float) * N);

    cudaMemcpy(d_A, h_A, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(float) * N, cudaMemcpyHostToDevice);

    cuda_matrix_add(d_A, d_B, d_C, rows, cols);
    cudaDeviceSynchronize();

    cudaMemcpy(h_D, d_C, sizeof(float) * N, cudaMemcpyDeviceToHost);

    // Check results
    std::cout << "CPU result: ";
    if (check_result(h_C, h_D, N)) {
        std::cout << "\nGPU matrix add equals CPU matrix add\n" << std::endl;
    } else {
        std::cout << "\nGPU matrix add does NOT equal CPU matrix add\n" << std::endl;
    }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_D;

    return 0;
}
