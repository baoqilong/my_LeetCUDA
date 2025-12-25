#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    int device_count;
    cudaGetDeviceCount(&device_count);

    printf("============================================\n");
    printf("CUDA Device Information\n");
    printf("============================================\n");
    printf("Number of CUDA devices: %d\n\n", device_count);

    for (int device_id = 0; device_id < device_count; device_id++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device_id);

        printf("Device %d: %s\n", device_id, prop.name);
        printf("--------------------------------------------\n");
        printf("Compute Capability:        %d.%d\n", prop.major, prop.minor);
        printf("\n");

        printf("Memory Information:\n");
        printf("  Total Global Mem:        %.2f GB\n", (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Total Constant Mem:      %.2f KB\n", (double)prop.totalConstMem / 1024.0);
        printf("  Shared Mem Per Block:    %.2f KB\n", (double)prop.sharedMemPerBlock / 1024.0);
        printf("  Memory Bus Width:        %d bits\n", prop.memoryBusWidth);
        printf("  L2 Cache Size:           %d KB\n", prop.l2CacheSize / 1024);
        printf("\n");

        printf("Multiprocessor Information:\n");
        printf("  MultiProcessor Count:    %d\n", prop.multiProcessorCount);
        printf("  Max Threads Per Block:   %d\n", prop.maxThreadsPerBlock);
        printf("  Max Threads Per SM:      %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  Max Threads Dim:         [%d, %d, %d]\n",
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Max Grid Size:           [%d, %d, %d]\n",
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("  Warp Size:               %d\n", prop.warpSize);
        printf("  Registers Per Block:     %d\n", prop.regsPerBlock);
        printf("  Registers Per SM:        %d\n", prop.regsPerMultiprocessor);
        printf("\n");

        printf("Other Information:\n");
        printf("  Integrated:              %s\n", prop.integrated ? "Yes" : "No");
        printf("  Can Map Host Memory:     %s\n", prop.canMapHostMemory ? "Yes" : "No");
        printf("  Unified Addressing:      %s\n", prop.unifiedAddressing ? "Yes" : "No");
        printf("  Managed Memory:          %s\n", prop.managedMemory ? "Yes" : "No");
        printf("  Concurrent Kernels:      %s\n", prop.concurrentKernels ? "Yes" : "No");
        printf("  ECC Enabled:             %s\n", prop.ECCEnabled ? "Yes" : "No");
        printf("  Async Engines Count:     %d\n", prop.asyncEngineCount);
        printf("  PCIe Generation:         %d\n", prop.pciBusID);
        printf("============================================\n\n");
    }

    return 0;
}
