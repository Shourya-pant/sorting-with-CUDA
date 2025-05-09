#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <limits.h>

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

// Round up to next power of 2
int nextPowerOf2(int n) {
    int power = 1;
    while (power < n) {
        power *= 2;
    }
    return power;
}

// Save sorting results to a file
void saveResultsToFile(const int* data, int n, const char* filename) {
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        printf("Error: Could not open file %s for writing\n", filename);
        return;
    }
    
    outFile << "Sorted data:\n";
    for (int i = 0; i < n; i++) {
        outFile << data[i];
        if (i < n-1) outFile << ", ";
        if ((i+1) % 10 == 0) outFile << "\n";
    }
    
    outFile.close();
    printf("Results saved to %s\n", filename);
}

// Process user input for sorting
std::vector<int> getUserInput() {
    std::vector<int> data;
    std::string input;
    
    std::cout << "Enter elements to sort (comma, space, or newline separated), then press Enter twice to finish:\n";
    
    // Collect multi-line input
    std::string line;
    bool emptyLineFound = false;
    
    while (!emptyLineFound) {
        std::getline(std::cin, line);
        if (line.empty() && !input.empty()) {
            emptyLineFound = true;
        } else {
            if (!input.empty()) {
                input += " ";
            }
            input += line;
        }
    }
    
    // Replace commas with spaces for easier parsing
    for (char& c : input) {
        if (c == ',') c = ' ';
    }
    
    // Parse input string into integers
    std::stringstream ss(input);
    int val;
    while (ss >> val) {
        data.push_back(val);
    }
    
    return data;
}

// Main function
int main(int argc, char *argv[]) {
    int n = 0;
    int *data = nullptr;
    int *std_sort_data = nullptr;
    bool useRandomData = false;
    bool useInputFile = false;
    std::string inputFilename;
    
    std::cout << "===== Bitonic Sort Implementation =====\n";
    std::cout << "Select input option:\n";
    std::cout << "1. Enter numbers manually\n";
    std::cout << "2. Generate random numbers\n";
    std::cout << "3. Read from file\n";
    
    int choice;
    std::cin >> choice;
    std::cin.ignore(); // Clear the newline
    
    if (choice == 1) {
        // Get user input
        std::vector<int> userInput = getUserInput();
        
        if (userInput.empty()) {
            std::cout << "No valid input provided.\n";
            return 1;
        }
        
        // Determine size (round up to power of 2)
        n = nextPowerOf2(userInput.size());
        if (n != (int)userInput.size()) {
            std::cout << "Input size " << userInput.size() << " is not a power of 2. Padding to " << n << " elements.\n";
        }
        
        // Allocate memory
        data = (int *)malloc(n * sizeof(int));
        std_sort_data = (int *)malloc(n * sizeof(int));
        
        if (!data || !std_sort_data) {
            fprintf(stderr, "Failed to allocate memory\n");
            exit(EXIT_FAILURE);
        }
        
        // Copy user input and pad with max value if needed
        for (int i = 0; i < n; i++) {
            if (i < (int)userInput.size()) {
                data[i] = userInput[i];
            } else {
                data[i] = INT_MAX; // Pad with maximum value so they end up at the end after sorting
            }
        }
    }
    else if (choice == 2) {
        useRandomData = true;
        
        // Ask for size
        std::cout << "Enter size (will be rounded up to power of 2): ";
        int requestedSize;
        std::cin >> requestedSize;
        
        // Round up to power of 2
        n = nextPowerOf2(requestedSize);
        if (n != requestedSize) {
            std::cout << "Rounding up to " << n << " (next power of 2)\n";
        }
        
        // Allocate memory
        data = (int *)malloc(n * sizeof(int));
        std_sort_data = (int *)malloc(n * sizeof(int));
        
        if (!data || !std_sort_data) {
            fprintf(stderr, "Failed to allocate memory\n");
            exit(EXIT_FAILURE);
        }
        
        // Generate random data
        srand(time(NULL));
        generateRandomData(data, n);
    }
    else if (choice == 3) {
        useInputFile = true;
        
        // Ask for filename
        std::cout << "Enter input filename: ";
        std::cin >> inputFilename;
        
        // Read data from file
        std::ifstream inFile(inputFilename);
        if (!inFile.is_open()) {
            std::cout << "Error: Unable to open file " << inputFilename << std::endl;
            return 1;
        }
        
        std::vector<int> fileData;
        int value;
        while (inFile >> value) {
            fileData.push_back(value);
            // Skip any non-number characters
            if (inFile.peek() == ',' || inFile.peek() == ' ')
                inFile.ignore();
        }
        
        if (fileData.empty()) {
            std::cout << "No valid data found in file.\n";
            return 1;
        }
        
        // Determine size (round up to power of 2)
        n = nextPowerOf2(fileData.size());
        if (n != (int)fileData.size()) {
            std::cout << "Input size " << fileData.size() << " is not a power of 2. Padding to " << n << " elements.\n";
        }
        
        // Allocate memory
        data = (int *)malloc(n * sizeof(int));
        std_sort_data = (int *)malloc(n * sizeof(int));
        
        if (!data || !std_sort_data) {
            fprintf(stderr, "Failed to allocate memory\n");
            exit(EXIT_FAILURE);
        }
        
        // Copy file data and pad with max value if needed
        for (int i = 0; i < n; i++) {
            if (i < (int)fileData.size()) {
                data[i] = fileData[i];
            } else {
                data[i] = INT_MAX; // Pad with maximum value
            }
        }
    }
    else {
        std::cout << "Invalid choice.\n";
        return 1;
    }
    
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
    
    // Ask if user wants to save results
    std::cout << "\nDo you want to save the sorted data to a file? (y/n): ";
    char saveChoice;
    std::cin >> saveChoice;
    
    if (saveChoice == 'y' || saveChoice == 'Y') {
        std::string outputFilename;
        std::cout << "Enter output filename: ";
        std::cin >> outputFilename;
        saveResultsToFile(data, n, outputFilename.c_str());
    }
    
    // Free allocated memory
    free(data);
    free(std_sort_data);
    
    std::cout << "\nPress Enter to exit...";
    std::cin.ignore();
    std::cin.get();
    
    return 0;
} 