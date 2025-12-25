#include <iostream>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include "copy.cuh"

// CPU reference implementation
void cpu_matrix_copy(const float* A, float* B, int N) {
    for (int i = 0; i < N; ++i) {
        B[i] = A[i];
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
    int N;
    float value;
    const char* name;
};

// Run a single test case
bool run_test(const TestCase& test, bool verbose = false) {
    // Allocate host memory
    std::vector<float> h_A(test.N), h_B_cpu(test.N), h_B_gpu(test.N, 0.0f);

    // Initialize input data
    for (int i = 0; i < test.N; ++i) {
        h_A[i] = test.value;
    }

    // CPU computation
    cpu_matrix_copy(h_A.data(), h_B_cpu.data(), test.N);

    // GPU computation
    float *d_A, *d_B;
    cudaMalloc(&d_A, sizeof(float) * test.N);
    cudaMalloc(&d_B, sizeof(float) * test.N);

    cudaMemcpy(d_A, h_A.data(), sizeof(float) * test.N, cudaMemcpyHostToDevice);

    // Call the function from copy.cu
    copy_matrix_kernel(d_A, d_B, test.N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_B_gpu.data(), d_B, sizeof(float) * test.N, cudaMemcpyDeviceToHost);

    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cout << "[FAIL] " << test.name << " - CUDA error: " << cudaGetErrorString(error) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        return false;
    }

    // Verify results
    bool passed = check_result(h_B_cpu.data(), h_B_gpu.data(), test.N, 1e-6f, verbose);

    if (passed) {
        std::cout << "[PASS] " << test.name;
    } else {
        std::cout << "[FAIL] " << test.name;
    }
    std::cout << " (N=" << test.N << ")" << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);

    return passed;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  CUDA Matrix Copy Test Suite" << std::endl;
    std::cout << "========================================" << std::endl;

    // Define test cases
    std::vector<TestCase> tests = {
        {120, 1.0f, "Basic test (copy 1.0)"},
        {256, 2.5f, "Exact block size"},
        {1, 3.14f, "Single element"},
        {10000, 0.0f, "Large array (zeros)"},
        {10000, -1.5f, "Large array (negative)"},
        {1000, 3.14159f, "Medium array (pi)"},
        {12345, 1.0f, "Non-multiple of 256"},
        {1000000, 42.0f, "Very large array"},
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
    run_test({120, 1.0f, "Basic test (verbose)"}, true);

    return (passed == total) ? 0 : 1;
}
