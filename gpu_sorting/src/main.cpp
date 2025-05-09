#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>
#include "../include/sort.h"

// Function declaration from sort.cu
extern int runGpuSort(int *data, int n);
extern bool verifySorting(const int *data, int n);
extern void printArray(const int *arr, int n, const char *label);

// CPU sorting function for comparison
void cpuSort(int *data, int n) {
    std::sort(data, data + n);
}

// Generate random data
void generateRandomData(int *data, int n, int max_val = 1000000) {
    for (int i = 0; i < n; i++) {
        data[i] = rand() % max_val;
    }
}

// Helper function to copy array
void copyArray(int *dst, const int *src, int n) {
    for (int i = 0; i < n; i++) {
        dst[i] = src[i];
    }
}

int main(int argc, char *argv[]) {
    // Default size if not specified
    int n = 1 << 20; // 1M elements

    // Use command line argument if provided
    if (argc > 1) {
        n = 1 << atoi(argv[1]);
        printf("Setting array size to 2^%d = %d elements\n", atoi(argv[1]), n);
    } else {
        printf("Using default array size of %d elements\n", n);
        printf("Hint: You can specify the power of 2 for the array size as a command line argument\n");
    }

    // Ensure the size is a power of 2 (required for bitonic sort)
    if ((n & (n - 1)) != 0) {
        printf("Warning: Size %d is not a power of 2, adjusting...\n", n);
        // Find the next highest power of 2
        int power = 0;
        while (n > 0) {
            n >>= 1;
            power++;
        }
        n = 1 << power;
        printf("Adjusted size to %d\n", n);
    }

    // Seed random number generator
    srand(time(NULL));

    // Allocate memory for arrays
    int *original_data = (int *)malloc(n * sizeof(int));
    int *gpu_data = (int *)malloc(n * sizeof(int));
    int *cpu_data = (int *)malloc(n * sizeof(int));

    if (!original_data || !gpu_data || !cpu_data) {
        fprintf(stderr, "Failed to allocate host memory\n");
        exit(EXIT_FAILURE);
    }

    // Generate random data
    printf("Generating random data...\n");
    generateRandomData(original_data, n);

    // Copy data to CPU and GPU arrays
    copyArray(gpu_data, original_data, n);
    copyArray(cpu_data, original_data, n);

    // Print a small sample of the unsorted data
    printArray(original_data, n, "Unsorted");

    // Run GPU sort and measure time
    printf("\nRunning GPU Bitonic Sort...\n");
    float gpu_time = runGpuSort(gpu_data, n);
    printf("GPU sorting time: %.3f ms\n", gpu_time);

    // Run CPU sort and measure time
    printf("\nRunning CPU Sort (std::sort)...\n");
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpuSort(cpu_data, n);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    float cpu_time = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();
    printf("CPU sorting time: %.3f ms\n", cpu_time);

    // Print a small sample of the sorted data
    printArray(gpu_data, n, "GPU Sorted");
    printArray(cpu_data, n, "CPU Sorted");

    // Verify the sorting results
    printf("\nVerifying GPU sort result...\n");
    bool gpu_correct = verifySorting(gpu_data, n);
    printf("GPU sort %s\n", gpu_correct ? "CORRECT" : "FAILED");

    printf("\nVerifying CPU sort result...\n");
    bool cpu_correct = verifySorting(cpu_data, n);
    printf("CPU sort %s\n", cpu_correct ? "CORRECT" : "FAILED");

    // Compare CPU and GPU sorted results
    bool results_match = true;
    for (int i = 0; i < n; i++) {
        if (gpu_data[i] != cpu_data[i]) {
            printf("Results don't match at index %d: GPU = %d, CPU = %d\n", 
                   i, gpu_data[i], cpu_data[i]);
            results_match = false;
            break;
        }
    }
    printf("CPU and GPU results %s\n", results_match ? "MATCH" : "DON'T MATCH");

    // Calculate speedup
    printf("\nSpeedup: %.2fx\n", cpu_time / gpu_time);

    // Free allocated memory
    free(original_data);
    free(gpu_data);
    free(cpu_data);

    return 0;
} 