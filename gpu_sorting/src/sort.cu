#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include "../include/sort.h"

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Bitonic sort kernel for GPU
__global__ void bitonicSortKernel(int *data, int j, int k) {
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int ixj = i ^ j;
    
    // Only threads with ixj > i are used
    if (ixj > i) {
        // Sort in ascending or descending order
        if ((i & k) == 0) {
            // Ascending
            if (data[i] > data[ixj]) {
                int temp = data[i];
                data[i] = data[ixj];
                data[ixj] = temp;
            }
        } else {
            // Descending
            if (data[i] < data[ixj]) {
                int temp = data[i];
                data[i] = data[ixj];
                data[ixj] = temp;
            }
        }
    }
}

// Float version of bitonic sort kernel for GPU
__global__ void bitonicSortKernelFloat(float *data, int j, int k) {
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int ixj = i ^ j;
    
    // Only threads with ixj > i are used
    if (ixj > i) {
        // Sort in ascending or descending order
        if ((i & k) == 0) {
            // Ascending
            if (data[i] > data[ixj]) {
                float temp = data[i];
                data[i] = data[ixj];
                data[ixj] = temp;
            }
        } else {
            // Descending
            if (data[i] < data[ixj]) {
                float temp = data[i];
                data[i] = data[ixj];
                data[ixj] = temp;
            }
        }
    }
}

// Launch bitonic sort on GPU
void gpuBitonicSort(int *d_data, int n, int threadsPerBlock) {
    // Bitonic sort stages
    for (int k = 2; k <= n; k <<= 1) {
        // Bitonic merge steps
        for (int j = k >> 1; j > 0; j >>= 1) {
            // Calculate grid size based on data size and thread block size
            int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
            
            // Launch kernel
            bitonicSortKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, j, k);
            
            // Check for errors after kernel launch
            CUDA_CHECK(cudaGetLastError());
            
            // Synchronize to ensure kernel completion
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
}

// Float version of bitonic sort - implementation for the bridge application
extern "C" void bitonicSort(float *d_input, float *d_output, size_t n) {
    // Copy input to output for in-place sorting
    CUDA_CHECK(cudaMemcpy(d_output, d_input, n * sizeof(float), cudaMemcpyDeviceToDevice));
    
    int threadsPerBlock = 256;
    
    // Bitonic sort stages
    for (int k = 2; k <= n; k <<= 1) {
        // Bitonic merge steps
        for (int j = k >> 1; j > 0; j >>= 1) {
            // Calculate grid size based on data size and thread block size
            int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
            
            // Launch kernel
            bitonicSortKernelFloat<<<blocksPerGrid, threadsPerBlock>>>(d_output, j, k);
            
            // Check for errors after kernel launch
            CUDA_CHECK(cudaGetLastError());
            
            // Synchronize to ensure kernel completion
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
}

// Verify the sorting results
bool verifySorting(const int *data, int n) {
    for (int i = 0; i < n - 1; i++) {
        if (data[i] > data[i + 1]) {
            printf("Sorting verification failed at index %d: %d > %d\n", 
                   i, data[i], data[i + 1]);
            return false;
        }
    }
    return true;
}

// Print array for debugging
void printArray(const int *arr, int n, const char *label) {
    printf("%s: ", label);
    for (int i = 0; i < (n < 20 ? n : 20); i++) {
        printf("%d ", arr[i]);
    }
    if (n > 20) printf("...");
    printf("\n");
}

// Main function to demonstrate GPU bitonic sort
int runGpuSort(int *data, int n) {
    int *d_data = NULL;
    int size = n * sizeof(int);
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_data, size));
    
    // Copy input data from host to device
    CUDA_CHECK(cudaMemcpy(d_data, data, size, cudaMemcpyHostToDevice));
    
    // Get device properties
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
    
    // Use maximum threads per block (up to 512, which is often optimal)
    int threadsPerBlock = 256;
    if (deviceProp.maxThreadsPerBlock < threadsPerBlock) {
        threadsPerBlock = deviceProp.maxThreadsPerBlock;
    }
    
    // Start timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    
    // Perform the sort
    gpuBitonicSort(d_data, n, threadsPerBlock);
    
    // Stop timing
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(data, d_data, size, cudaMemcpyDeviceToHost));
    
    // Clean up
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_data));
    
    return milliseconds;
} 