# GPU Sorting Project - Setup Guide

This guide provides detailed instructions for setting up and building the GPU-optimized parallel sorting project.

## Prerequisites

To build and run this project, you'll need:

1. **NVIDIA GPU** with CUDA support
2. **CUDA Toolkit** (10.0 or higher recommended) - [Download CUDA](https://developer.nvidia.com/cuda-downloads)
3. **Visual Studio** with C++ development tools - [Download Visual Studio](https://visualstudio.microsoft.com/downloads/)
   - During installation, be sure to select the "Desktop development with C++" workload

## Installation Steps

### 1. Install CUDA Toolkit

1. Download CUDA Toolkit from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads)
2. Follow the installation wizard instructions
3. Make sure to select the option to install Visual Studio integration

### 2. Install Visual Studio

1. Download Visual Studio Community (free) or other edition from [Microsoft's website](https://visualstudio.microsoft.com/downloads/)
2. During installation, select the "Desktop development with C++" workload
3. Complete the installation

### 3. Verify CUDA Installation

1. Open a command prompt
2. Run `nvcc --version` to verify CUDA installation
3. If you see version information, CUDA is correctly installed

## Building the Project

### Using Visual Studio (Recommended)

1. Open Visual Studio
2. Select "Open a local folder" and navigate to the `gpu_sorting` folder
3. Right-click on the `CMakeLists.txt` file and select "Configure CMake"
4. Select "Build > Build All" to compile the project
5. Run the executable from the `build` directory

### Using Command Line

If Visual Studio is properly installed and you have the C++ compiler in your PATH:

1. Open a Developer Command Prompt for Visual Studio
   - Search for "Developer Command Prompt" in the Start menu
2. Navigate to the project directory: `cd path\to\gpu_sorting`
3. Run the build script: `build.bat`

## Troubleshooting

### CUDA Compiler (nvcc) Can't Find C++ Compiler (cl.exe)

This happens when the Visual Studio C++ compiler is not in your PATH.

**Solution:**
1. Use the "Developer Command Prompt for Visual Studio" instead of a regular command prompt
2. Or, run the following before building (replace with your Visual Studio version/path):
   ```
   "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
   ```

### "CUDA driver version is insufficient for CUDA runtime version"

This means your GPU driver is older than your CUDA Toolkit.

**Solution:**
1. Update your NVIDIA GPU drivers from [NVIDIA's website](https://www.nvidia.com/Download/index.aspx)

### No CUDA-capable Device is Detected

This means either:
- You don't have an NVIDIA GPU
- Your NVIDIA GPU doesn't support CUDA
- Your NVIDIA drivers are not installed correctly

**Solution:**
1. Verify you have a CUDA-capable NVIDIA GPU
2. Reinstall NVIDIA drivers

## Modifying the Code

The project is structured as follows:

- `src/main.cpp`: Host code for data generation and CPU sorting
- `src/sort.cu`: CUDA implementation of bitonic sort
- `include/sort.h`: Header file with function declarations

To modify the sorting algorithm or optimize performance:

1. Adjust thread and block sizes in `sort.cu`
2. Implement different sorting algorithms (radix sort, merge sort, etc.)
3. Optimize memory access patterns for better performance

## Contact for Support

If you encounter issues not covered in this guide, please contact your instructor or project lead for assistance. 