#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>

// CPU-based implementation of bitonic sort (equivalent to the GPU version)
void bitonicSort(int *data, int n) {
    // Iterate over each stage of the bitonic sort
    for (int k = 2; k <= n; k <<= 1) {
        // Iterate over each step within a stage
        for (int j = k >> 1; j > 0; j >>= 1) {
            // Process all elements in parallel (sequential on CPU)
            for (int i = 0; i < n; i++) {
                int ixj = i ^ j;
                
                // Only process if ixj > i to avoid duplicate swaps
                if (ixj > i) {
                    // Determine sort direction
                    if ((i & k) == 0) {
                        // Ascending
                        if (data[i] > data[ixj]) {
                            // Swap
                            int temp = data[i];
                            data[i] = data[ixj];
                            data[ixj] = temp;
                        }
                    } else {
                        // Descending
                        if (data[i] < data[ixj]) {
                            // Swap
                            int temp = data[i];
                            data[i] = data[ixj];
                            data[ixj] = temp;
                        }
                    }
                }
            }
        }
    }
}

// Verification function
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

// Print array
void printArray(const int *arr, int n, const char *label) {
    printf("%s: ", label);
    for (int i = 0; i < (n < 20 ? n : 20); i++) {
        printf("%d ", arr[i]);
    }
    if (n > 20) printf("...");
    printf("\n");
}

// Generate random data
void generateRandomData(int *data, int n, int max_val = 1000000) {
    for (int i = 0; i < n; i++) {
        data[i] = rand() % max_val;
    }
}

// Main function
int main(int argc, char *argv[]) {
    // Default size if not specified
    int n = 1 << 16; // 65,536 elements

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
    int *data = (int *)malloc(n * sizeof(int));
    int *std_sort_data = (int *)malloc(n * sizeof(int));

    if (!data || !std_sort_data) {
        fprintf(stderr, "Failed to allocate memory\n");
        exit(EXIT_FAILURE);
    }

    // Generate random data
    printf("Generating random data...\n");
    generateRandomData(data, n);

    // Copy data for std::sort comparison
    for (int i = 0; i < n; i++) {
        std_sort_data[i] = data[i];
    }

    // Print a small sample of the unsorted data
    printArray(data, n, "Unsorted");

    // Run bitonic sort and measure time
    printf("\nRunning CPU Bitonic Sort...\n");
    auto bitonic_start = std::chrono::high_resolution_clock::now();
    bitonicSort(data, n);
    auto bitonic_end = std::chrono::high_resolution_clock::now();
    float bitonic_time = std::chrono::duration<float, std::milli>(bitonic_end - bitonic_start).count();
    printf("CPU Bitonic sort time: %.3f ms\n", bitonic_time);

    // Run std::sort and measure time
    printf("\nRunning std::sort...\n");
    auto std_start = std::chrono::high_resolution_clock::now();
    std::sort(std_sort_data, std_sort_data + n);
    auto std_end = std::chrono::high_resolution_clock::now();
    float std_time = std::chrono::duration<float, std::milli>(std_end - std_start).count();
    printf("std::sort time: %.3f ms\n", std_time);

    // Print a small sample of the sorted data
    printArray(data, n, "Bitonic Sorted");
    printArray(std_sort_data, n, "std::sort Sorted");

    // Verify the sorting results
    printf("\nVerifying bitonic sort result...\n");
    bool bitonic_correct = verifySorting(data, n);
    printf("Bitonic sort %s\n", bitonic_correct ? "CORRECT" : "FAILED");

    printf("\nVerifying std::sort result...\n");
    bool std_correct = verifySorting(std_sort_data, n);
    printf("std::sort %s\n", std_correct ? "CORRECT" : "FAILED");

    // Compare results
    bool results_match = true;
    for (int i = 0; i < n; i++) {
        if (data[i] != std_sort_data[i]) {
            printf("Results don't match at index %d: Bitonic = %d, std::sort = %d\n", 
                   i, data[i], std_sort_data[i]);
            results_match = false;
            break;
        }
    }
    printf("Bitonic sort and std::sort results %s\n", results_match ? "MATCH" : "DON'T MATCH");

    // Calculate comparison
    printf("\nPerformance: std::sort is %.2fx %s than Bitonic sort on CPU\n", 
           (bitonic_time > std_time) ? (bitonic_time / std_time) : (std_time / bitonic_time),
           (bitonic_time > std_time) ? "faster" : "slower");

    // Free allocated memory
    free(data);
    free(std_sort_data);

    return 0;
} 