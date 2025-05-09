#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <limits>
#include <algorithm>
#include <cuda_runtime.h>
#include "../include/sort.h"

// For handling large datasets
#define CHUNK_SIZE 1000000 // 1 million elements per chunk for reading

// Function to read numbers from a file, optimized for large files
std::vector<float> readNumbersFromFile(const std::string& filename) {
    std::vector<float> numbers;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return numbers;
    }
    
    std::string line;
    size_t lineCount = 0;
    size_t validNumbers = 0;
    
    // First count lines to pre-allocate memory
    std::ifstream countFile(filename);
    if (countFile.is_open()) {
        while (std::getline(countFile, line)) {
            lineCount++;
        }
        countFile.close();
        
        // Pre-allocate with some overhead (some lines might be empty)
        numbers.reserve(lineCount);
        std::cout << "Pre-allocated space for " << lineCount << " numbers" << std::endl;
    }
    
    // Now read the actual data
    while (std::getline(file, line)) {
        try {
            // Skip empty lines
            if (line.empty() || line.find_first_not_of(" \t\r\n") == std::string::npos) {
                continue;
            }
            
            // Parse number and add to vector
            float num = std::stof(line);
            numbers.push_back(num);
            validNumbers++;
            
            // Report progress for large files
            if (validNumbers % CHUNK_SIZE == 0) {
                std::cout << "Read " << validNumbers << " numbers so far..." << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "Warning: Could not parse number from line: " << line << std::endl;
            // Continue with next line
        }
    }
    
    std::cout << "Read " << numbers.size() << " numbers from file" << std::endl;
    return numbers;
}

// Function to write numbers to a file, optimized for large files
void writeResultsToFile(const std::string& filename, double executionTimeMs, const std::vector<float>& numbers) {
    std::ofstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }
    
    // Write execution time as first line
    file << "Time: " << executionTimeMs << " ms" << std::endl;
    
    // Write sorted numbers in chunks for large datasets
    size_t totalSize = numbers.size();
    for (size_t i = 0; i < totalSize; i += CHUNK_SIZE) {
        size_t chunkSize = std::min<size_t>(CHUNK_SIZE, totalSize - i);
        
        for (size_t j = 0; j < chunkSize; j++) {
            file << numbers[i + j] << std::endl;
        }
        
        // Report progress for large datasets
        if (i > 0 && i % CHUNK_SIZE == 0) {
            std::cout << "Wrote " << i << " of " << totalSize << " numbers to file..." << std::endl;
        }
    }
    
    file.close();
    std::cout << "Wrote " << numbers.size() << " numbers to file" << std::endl;
}

// Check for CUDA errors
void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

// Program entry point
int main(int argc, char* argv[]) {
    std::cout << "Starting GPU sort bridge" << std::endl;
    std::cout << "CUDA version: " << CUDART_VERSION << std::endl;
    
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <output_file>" << std::endl;
        return 1;
    }
    
    std::string inputFile = argv[1];
    std::string outputFile = argv[2];
    
    std::cout << "Input file: " << inputFile << std::endl;
    std::cout << "Output file: " << outputFile << std::endl;
    
    // Read numbers from input file
    std::vector<float> numbers = readNumbersFromFile(inputFile);
    
    if (numbers.empty()) {
        std::cerr << "Error: Input file is empty or invalid" << std::endl;
        return 1;
    }
    
    // Ensure size is a power of 2 (required by bitonic sort)
    size_t originalSize = numbers.size();
    size_t powerOfTwo = 1;
    while (powerOfTwo < originalSize) {
        powerOfTwo *= 2;
    }
    
    std::cout << "Original size: " << originalSize << std::endl;
    std::cout << "Power of two size: " << powerOfTwo << std::endl;
    
    // Check if dataset is too large for GPU memory
    // Most consumer GPUs have 8-16GB of memory
    const size_t MAX_GPU_ELEMENTS = 250000000; // About 1GB of memory (assuming float = 4 bytes)
    
    if (powerOfTwo > MAX_GPU_ELEMENTS) {
        std::cerr << "Warning: Dataset is very large and may exceed GPU memory." << std::endl;
        std::cerr << "Consider using a smaller dataset or processing in chunks." << std::endl;
        
        // Exit with a special code that the Node.js bridge can recognize
        return 2; 
    }
    
    // Pad to power of 2
    numbers.resize(powerOfTwo, std::numeric_limits<float>::max());
    
    // Create a vector for results
    std::vector<float> results(powerOfTwo);
    
    // Allocate device memory
    float* d_input = nullptr;
    float* d_output = nullptr;
    
    cudaError_t err;
    
    err = cudaMalloc(&d_input, powerOfTwo * sizeof(float));
    checkCudaError(err, "Failed to allocate device input array");
    
    err = cudaMalloc(&d_output, powerOfTwo * sizeof(float));
    checkCudaError(err, "Failed to allocate device output array");
    
    // Copy data to device
    err = cudaMemcpy(d_input, numbers.data(), powerOfTwo * sizeof(float), cudaMemcpyHostToDevice);
    checkCudaError(err, "Failed to copy data to device");
    
    std::cout << "Data copied to device, starting sort..." << std::endl;
    
    // Run GPU bitonic sort
    auto start = std::chrono::high_resolution_clock::now();
    
    try {
        bitonicSort(d_input, d_output, powerOfTwo);
        
        // Wait for GPU to finish
        err = cudaDeviceSynchronize();
        checkCudaError(err, "Synchronization failed");
    }
    catch (const std::exception& e) {
        std::cerr << "Exception during sort: " << e.what() << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        return 1;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double executionTimeMs = std::chrono::duration<double, std::milli>(end - start).count();
    
    std::cout << "Sort completed in " << executionTimeMs << " ms" << std::endl;
    
    // Copy results back to host
    err = cudaMemcpy(results.data(), d_output, powerOfTwo * sizeof(float), cudaMemcpyDeviceToHost);
    checkCudaError(err, "Failed to copy results from device");
    
    // Clean up device memory
    cudaFree(d_input);
    cudaFree(d_output);
    
    // Resize back to original size (removing padding)
    results.resize(originalSize);
    
    // Write results to output file
    writeResultsToFile(outputFile, executionTimeMs, results);
    
    std::cout << "Results written to " << outputFile << std::endl;
    
    return 0;
} 