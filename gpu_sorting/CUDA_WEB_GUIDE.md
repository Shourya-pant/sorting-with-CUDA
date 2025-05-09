# Guide to Running the GPU Sorting Web Application

This guide provides step-by-step instructions for running the GPU-accelerated bitonic sort web application with CUDA 12.1.

## Prerequisites

1. **NVIDIA GPU** - You must have an NVIDIA GPU with CUDA support
2. **CUDA Toolkit 12.1** - Install from [NVIDIA's website](https://developer.nvidia.com/cuda-12-1-0-download-archive)
3. **Visual Studio 2022 Preview** - With C++ development tools installed
4. **Node.js** - Version 14 or higher

## Step 1: Verify CUDA Installation

1. Make sure CUDA 12.1 is installed at:
   ```
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1
   ```

2. If you installed it elsewhere, you'll need to update the path in:
   - `build_bridge.bat`

## Step 2: Run the Application

1. Open a Command Prompt window
2. Navigate to the project directory:
   ```
   cd path\to\gpu_sorting
   ```
3. Run the web application with GPU support:
   ```
   run_web_with_gpu.bat
   ```
   
   This batch file will:
   - Check for CUDA installation
   - Build the CUDA bridge application
   - Install Node.js dependencies
   - Start the web server
   - Open your browser automatically

4. Your browser should automatically open to `http://localhost:3000`

## Step 3: Using the Web Interface

1. **Input Data**:
   - Choose one of the input methods (manual entry, random generation, or file upload)
   - For best performance comparison, try with 100,000+ random numbers

2. **Sort the Data**:
   - First click "Sort Data (JS)" to run the JavaScript implementation
   - Then click "Run Native CUDA Sort" to process the same data with the GPU
   
3. **Compare Results**:
   - You'll see the timing for both JavaScript implementations and the CUDA GPU implementation
   - The CUDA speedup ratio shows how much faster the GPU is compared to JavaScript

## Troubleshooting

### Common Issues

1. **"CUDA v12.1 not found at expected location"**
   - Make sure CUDA 12.1 is installed
   - If installed at a different location, update the path in `build_bridge.bat`

2. **"Bridge executable not found"**
   - The CUDA bridge failed to build
   - Check the build output for errors
   - Make sure Visual Studio tools are installed

3. **"Error running native CUDA sort"**
   - Open the browser console (F12) for detailed error messages
   - Check the Node.js console output for CUDA errors
   - Try with a smaller dataset first

4. **JSON Parsing Error**
   - This usually indicates the CUDA bridge failed to produce valid output
   - Check the Node.js console for detailed error messages

### Advanced Troubleshooting

If you encounter persistent issues:

1. Run the build_bridge.bat script separately:
   ```
   build_bridge.bat
   ```

2. Try running the bridge executable directly:
   ```
   cd build
   gpu_sort_bridge.exe input.txt output.txt
   ```
   Where input.txt contains numbers separated by newlines

3. Check the console output for any CUDA-related errors

## Understanding the Results

- **JavaScript Bitonic Sort**: Usually slower than standard JavaScript sort due to single-threaded execution
- **Standard JavaScript Sort**: The browser's built-in sorting algorithm
- **Native CUDA GPU Sort**: The true GPU-accelerated implementation

For large datasets, the CUDA implementation should be significantly faster than both JavaScript implementations. 