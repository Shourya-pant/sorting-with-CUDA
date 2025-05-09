# GPU-Optimized Parallel Sorting with CUDA

This project implements a parallel bitonic sort algorithm on NVIDIA GPUs using CUDA. It compares the performance of GPU-based sorting with standard CPU-based sorting.

## Algorithm

Bitonic sort is a comparison-based sorting algorithm that is well-suited for parallel implementation on GPUs. It's particularly efficient for sorting large arrays on parallel architectures.

## Features

- CUDA-accelerated bitonic sort implementation
- Performance comparison with CPU-based sorting
- Interactive command-line interface for testing
- Web-based interface for visualization and testing

## Requirements

To build and run this project, you'll need:

1. NVIDIA GPU with CUDA support
2. CUDA Toolkit (10.0 or higher recommended)
3. CMake (3.8 or higher)
4. C++ compiler compatible with your CUDA version
5. Node.js (for the web application)

## Web Application

A web interface has been added to make it easier to interact with the sorting algorithm:

### Running the Web Application

#### Option 1: JavaScript-only Implementation (No GPU Required)

1. Simply run the batch file:
   ```
   run_web_app.bat
   ```

2. Or manually:
   ```
   cd gpu_sorting/web
   npm install
   npm start
   ```

#### Option 2: With Native CUDA Support (Requires NVIDIA GPU)

1. Run the GPU-enabled batch file:
   ```
   run_web_with_gpu.bat
   ```

   This will:
   - Build the C++ CUDA bridge application
   - Start the web server with native CUDA support
   - Open your browser to http://localhost:3000

2. On the web interface, you can now click the "Run Native CUDA Sort" button after generating data to see the actual GPU-accelerated performance!

### Web Application Features

- Input numbers manually, generate random data, or upload a file
- Visualize unsorted and sorted data
- Compare JavaScript bitonic sort with standard JavaScript sort
- Compare with actual CUDA GPU acceleration (when using Option 2)
- Download sorted results

### Important Note on Web Performance

**When running Option 1 (JavaScript-only), the web interface uses a JavaScript implementation of the bitonic sort algorithm, NOT the actual CUDA GPU implementation.**

JavaScript is single-threaded and has no direct GPU access, so the bitonic sort algorithm will actually be slower than the native Array.sort() method in the browser. This is expected behavior.

For the true performance benefits of GPU acceleration (typically 50-100x faster than CPU sorting), use Option 2 or run the C++ CUDA executable directly.

## Building the C++ Project

Follow these steps to build the project:

```bash
# Navigate to the project build directory
cd gpu_sorting/build

# Generate the build files
cmake ..

# Build the project
cmake --build .
```

## Running the Program

After building, you can run the program with:

```bash
# Default size (2^20 = 1,048,576 elements)
./gpu_sorting

# Specify size as power of 2 (e.g., 2^22 = 4,194,304 elements)
./gpu_sorting 22
```

## Performance Optimization

This implementation includes several optimizations:
- Efficient memory access patterns to maximize throughput
- Optimal thread block size selection
- Minimized synchronization between kernel launches
- Efficient use of GPU shared memory

## Customizing for Your GPU

You may need to adjust the CUDA architecture flags in the CMakeLists.txt file based on your specific NVIDIA GPU. The current settings support a wide range of GPUs (Compute Capability 5.0 to 8.6).

## Project Structure

- `src/`: Source code for CUDA implementation
- `include/`: Header files
- `web/`: Web application files
- `build/`: Build output directory
- Various batch files for building and running

## Known Limitations

- The current implementation requires the array size to be a power of 2
- Performance may vary based on the specific GPU model and CUDA version
- The web interface's JavaScript implementation will be slower than the CUDA implementation 