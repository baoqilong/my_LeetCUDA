#ifndef LinearSA
#define LinearSA
#include <cuda_runtime.h>

__global__ void LinearSelfAttention(const float* Q,const float* KtV,const float* deno,float* output,int M,int d);
void solve(const float* Q, const float* K, const float* V, float* output, int M, int d);
#endif