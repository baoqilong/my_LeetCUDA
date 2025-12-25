#include <iostream>
#include <cuda_runtime.h>
#include "copy.cuh"

void cpu_matrix_copy(const float* A, float* B, int N) {
    for (int i = 0; i < N; ++i) {
      B[i] = A[i];
    }
}

bool check_result(const float* cpu_res, const float* gpu_res, int N) {
    for (int i = 0; i < N; ++i) {
        // if (i % 10 == 0 && i != 0) {
        //     std::cout << "\n" << cpu_res[i] << " ";
        // } else {
        //     std::cout << cpu_res[i] << " ";
        // }
        if (std::fabs(cpu_res[i] - gpu_res[i]) > 1e-6f)
            return false;
    }
    return true;
}

int main() {
    int rows = 12;
    int cols = 10;
    int N = rows * cols;

    float *h_A, *h_B, *h_C;
    h_A = new float[N];
    h_B = new float[N];
    h_C = new float[N];

    // Initialize data
    for (int i = 0; i < N; ++i) {
        h_A[i] = h_B[i] = 1;
        h_C[i] = 0;
    }

    // GPU computation
    float *d_A, *d_B;
    cudaMalloc(&d_A, sizeof(float) * N);
    cudaMalloc(&d_B, sizeof(float) * N);

    cudaMemcpy(d_A, h_A, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(float) * N, cudaMemcpyHostToDevice);

    copy_matrix_kernel(d_A, d_B, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_B, sizeof(float) * N, cudaMemcpyDeviceToHost);

    // Check results
    std::cout << "CPU result: ";
    if (check_result(h_B, h_C, N)) {
        std::cout << "\nGPU matrix equals CPU matrix \n" << std::endl;
    } else {
        std::cout << "\nGPU matrix does NOT equal CPU matrix\n" << std::endl;
    }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
