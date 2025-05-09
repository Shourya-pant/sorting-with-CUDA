#include <stdio.h>

// Simple CUDA kernel
__global__ void testKernel() {
    printf("Hello from GPU thread %d in block %d\n", threadIdx.x, blockIdx.x);
}

int main() {
    printf("CUDA Test - Starting...\n");
    
    // Get CUDA device count
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    
    printf("Found %d CUDA devices\n", deviceCount);
    
    // Display device properties
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        
        printf("Device %d: %s\n", i, deviceProp.name);
        printf("  Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("  Total Global Memory: %.2f GB\n", deviceProp.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Max Threads Per Block: %d\n", deviceProp.maxThreadsPerBlock);
    }
    
    if (deviceCount > 0) {
        // Launch a simple kernel (1 block, 5 threads)
        printf("\nLaunching kernel...\n");
        testKernel<<<1, 5>>>();
        
        // Wait for kernel to finish
        cudaDeviceSynchronize();
        
        // Check for errors
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(error));
            return 1;
        }
    } else {
        printf("No CUDA devices found\n");
    }
    
    printf("\nCUDA Test - Complete\n");
    return 0;
} 