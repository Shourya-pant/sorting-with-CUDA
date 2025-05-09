#ifndef SORT_H
#define SORT_H

#ifdef __cplusplus
extern "C" {
#endif

// Function declarations
int runGpuSort(int *data, int n);
bool verifySorting(const int *data, int n);
void printArray(const int *arr, int n, const char *label);

// New function declaration for the bridge application
// This will allow sort_bridge.cpp to call the CUDA function
void bitonicSort(float *d_input, float *d_output, size_t n);

#ifdef __cplusplus
}
#endif

#endif // SORT_H 