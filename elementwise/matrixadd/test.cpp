#include <iostream>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include "f32add.cuh"

// CPU reference implementation
void cpu_matrix_add(const float* A, const float* B, float* C, int rows, int cols) {
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            C[row * cols + col] = A[row * cols + col] + B[row * cols + col];
        }
    }
}

// Check if two arrays are equal within tolerance
bool check_result(const float* ref, const float* gpu, int N, float tol = 1e-6f, bool verbose = false) {
    float max_error = 0.0f;
    int max_error_idx = 0;
    for (int i = 0; i < N; ++i) {
        float error = std::fabs(ref[i] - gpu[i]);
        if (error > max_error) {
            max_error = error;
            max_error_idx = i;
        }
        if (error > tol) {
            if (verbose) {
                std::cout << "  Mismatch at index " << i << ": ref=" << ref[i]
                          << ", gpu=" << gpu[i] << ", error=" << error << std::endl;
            }
            return false;
        }
    }
    if (verbose) {
        std::cout << "  Max error: " << max_error << " at index " << max_error_idx << std::endl;
    }
    return true;
}

// Test case structure
struct TestCase {
    int rows;
    int cols;
    float value_a;
    float value_b;
    const char* name;
};

// Run a single test case
bool run_test(const TestCase& test, bool verbose = false) {
    int N = test.rows * test.cols;

    // Allocate host memory
    std::vector<float> h_A(N), h_B(N), h_C_cpu(N), h_C_gpu(N, 0.0f);

    // Initialize input data
    for (int i = 0; i < N; ++i) {
        h_A[i] = test.value_a;
        h_B[i] = test.value_b;
    }

    // CPU computation
    cpu_matrix_add(h_A.data(), h_B.data(), h_C_cpu.data(), test.rows, test.cols);

    // GPU computation
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeof(float) * N);
    cudaMalloc(&d_B, sizeof(float) * N);
    cudaMalloc(&d_C, sizeof(float) * N);

    cudaMemcpy(d_A, h_A.data(), sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), sizeof(float) * N, cudaMemcpyHostToDevice);

    // Call the function from f32add.cu
    cuda_matrix_add(d_A, d_B, d_C, test.rows, test.cols);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C_gpu.data(), d_C, sizeof(float) * N, cudaMemcpyDeviceToHost);

    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cout << "[FAIL] " << test.name << " - CUDA error: " << cudaGetErrorString(error) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return false;
    }

    // Verify results
    bool passed = check_result(h_C_cpu.data(), h_C_gpu.data(), N, 1e-6f, verbose);

    if (passed) {
        std::cout << "[PASS] " << test.name;
    } else {
        std::cout << "[FAIL] " << test.name;
    }
    std::cout << " (" << test.rows << "x" << test.cols << ")" << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return passed;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  CUDA Matrix Addition Test Suite" << std::endl;
    std::cout << "========================================" << std::endl;

    // Define test cases
    std::vector<TestCase> tests = {
        {12, 10, 1.0f, 1.0f, "Basic test (1+1)"},
        {16, 16, 2.5f, 3.7f, "Exact block size"},
        {1, 1, 1.0f, 2.0f, "Single element"},
        {100, 100, 0.0f, 0.0f, "Large square (zeros)"},
        {100, 100, 1.0f, -1.0f, "Large square (1-1=0)"},
        {5, 200, 3.14f, 2.86f, "Wide matrix"},
        {200, 5, 1.0f, 1.0f, "Tall matrix"},
        {17, 33, 1.5f, 2.5f, "Non-multiple of 16"},
        {1024, 1024, 1.0f, 1.0f, "Very large"},
    };

    // Run all tests
    int passed = 0;
    int total = tests.size();

    for (const auto& test : tests) {
        if (run_test(test)) {
            passed++;
        }
    }

    std::endl(std::cout);
    std::cout << "========================================" << std::endl;
    std::cout << "  Results: " << passed << "/" << total << " tests passed";
    if (passed == total) {
        std::cout << " - ALL TESTS PASSED!";
    } else {
        std::cout << " - SOME TESTS FAILED!";
    }
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;

    // Run one test with verbose output for debugging
    std::endl(std::cout);
    std::cout << "=== Verbose output for basic test ===" << std::endl;
    run_test({12, 10, 1.0f, 1.0f, "Basic test (verbose)"}, true);

    return (passed == total) ? 0 : 1;
}
