#include <iostream>
#include <cuda_runtime.h>

__global__ void add(float *A, float *B, float* C,int N){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx<N){
    C[idx] = A[idx] + B[idx];
  }
}

bool checkRes(float *cpu_res, float *gpu_res, int N) {
  for (int i = 0; i < N; ++i ){
    
    if (cpu_res[i] - gpu_res[i] > 1e-6){
       return false;
    }
  }
  return true;
}

int main() {
  int N = 100000000;
  float *h_A, *h_B, *h_C,*h_D;
  h_A = new float[N];
  h_B = new float[N];
  h_C = new float[N];
  h_D = new float[N];
  for (int i = 0; i < N;++i){
    h_A[i] = h_B[i] = 1;
    h_C[i] = h_A[i] + h_B[i];
    h_D[i] = 0;
  }
  float *d_A, *d_B, *d_C;
  cudaMalloc((void **)&d_A, sizeof(float) * N);
  cudaMalloc((void **)&d_B, sizeof(float) * N);
  cudaMalloc((void **)&d_C, sizeof(float) * N);
  cudaMemcpy(d_A, h_A, sizeof(float) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, sizeof(float) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_C,h_C,sizeof(float)*N,cudaMemcpyHostToDevice);
  dim3 thread_per_block(256);
  dim3 block_per_grid((N + thread_per_block.x - 1) / thread_per_block.x);
  add<<<block_per_grid, thread_per_block>>>(d_A, d_B, d_C,N);
  cudaDeviceSynchronize();
  cudaMemcpy(h_D, d_C, sizeof(float) * N, cudaMemcpyDeviceToHost);
  if(checkRes(h_C,h_D,N))
    std::cout << "GPU vector add equals CPU vector add" << std::endl;
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  delete[] h_A;
  delete[] h_B;
  delete[] h_C;
  delete[] h_D;
  return 0;
}