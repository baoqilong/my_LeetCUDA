
// Matrix addition kernel device function
__global__ void copy(float* A, float* B,int N){
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx<N){
    B[idx] = A[idx];
  }
}

// Wrapper function for launching the kernel
void copy_matrix_kernel(float* d_A, float* d_B,int N){
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    copy<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, N);
    cudaDeviceSynchronize();
}