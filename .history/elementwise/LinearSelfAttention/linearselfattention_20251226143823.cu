#include <cuda_runtime.h>

__device__ float phi(float in){
    if(in>0) return in+1;
    return __expf(in);
}

__global__ void sigma_phik(const float* K,float* output,int M,int d){
    int col = blockDim.x*blockIdx.x+threadIdx.x;
    if(col<d){
        float sum = 0.0f;
        for(int row = 0;row<M;++row){
            sum+=phi(K[row*d+col]);
        }
        output[col] = sum;
    }
}

__global__ void denominator(const float* Q,float* sigma_k,float* deno,int M,int d){
    int row = threadIdx.x+blockDim.x*blockIdx.x;
    if(row<M){
        float sum = 0.0f;
        for(int col=0;col<d;++col){
            sum+=phi(Q[row*d+col])*sigma_k[col];
        }
        deno[row] = sum;
    }
}

__global__ void KTV(const float* K,const float* V,float* output,int M,int d){
    int row = threadIdx.y+blockIdx.y*blockDim.y;
    int col = threadIdx.x+blockDim.x*blockIdx.x;
    if(row<d&&col<d){
        float sum = 0.0f;
        for(int i=0;i<M;++i){
            sum+=phi(K[i*d+row])*V[i*d+col];
        }
        output[row*d+col] = sum;
    }
}

__global__ void LinearSelfAttention(const float* Q,const float* KtV,const float* deno,float* output,int M,int d){
    int row = threadIdx.y+blockDim.y*blockIdx.y;
    int col = threadIdx.x+blockDim.x*blockIdx.x;
    if(row<M&&col<d){
        float sum = 0.0f;
        for(int i=0;i<d;++i){
            sum+=phi(Q[row*d+i])*KtV[i*d+col];
        }
        output[row*d+col]=sum/deno[row];
    }
}



// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int d) {
    float * sigmk,*ktv,*denom;
    cudaMalloc(&sigmk,sizeof(float)*d);
    cudaMalloc(&ktv,d*d*sizeof(float));
    cudaMalloc(&denom,M*sizeof(float));
    int threads1=256;
    int blocks1=(d+threads1-1)/threads1;
    sigma_phik<<<blocks1,threads1>>>(K,sigmk,M,d);
    cudaDeviceSynchronize();
    int blocks2=(M+threads1-1)/threads1;
    denominator<<<blocks2,threads1>>>(Q,sigmk,denom,M,d);
    cudaDeviceSynchronize();
    dim3 threads3(16,16);
    dim3 blocks3((d+threads3.x-1)/threads3.x,(d+threads3.y-1)/threads3.y);
    KTV<<<blocks3,threads3>>>(K,V,ktv,M,d);
    cudaDeviceSynchronize();
    dim3 blocks4((d+threads3.x-1)/threads3.x,(M+threads3.y-1)/threads3.y);
    LinearSelfAttention<<<blocks4,threads3>>>(Q,ktv,denom,output,M,d);
    cudaDeviceSynchronize();
    cudaFree(sigmk);
    cudaFree(ktv);
    cudaFree(denom);
}
