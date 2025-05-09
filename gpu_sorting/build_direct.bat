@echo off
echo Building GPU Sorting project with NVCC directly...

REM Create build directory if it doesn't exist
if not exist build mkdir build

REM Get CUDA information
echo CUDA Information:
nvcc --version

REM Determine GPU architecture
echo.
echo Checking for NVIDIA GPU...
"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe" -L

REM Compile the project using NVCC with a simple approach
echo.
echo Compiling with command: nvcc -I./include -o build/gpu_sorting src/main.cpp src/sort.cu -std=c++14
echo.

REM First attempt - simple approach
nvcc -I./include -o build/gpu_sorting src/main.cpp src/sort.cu -std=c++14

REM Check if compilation succeeded
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo First compilation attempt failed, trying with explicit path to Visual Studio compiler
    echo.
    
    REM Second attempt - with standard location path
    echo Trying with: nvcc -I./include -o build/gpu_sorting src/main.cpp src/sort.cu -std=c++14 -ccbin="C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.36.32532\bin\Hostx64\x64"
    nvcc -I./include -o build/gpu_sorting src/main.cpp src/sort.cu -std=c++14 -ccbin="C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.36.32532\bin\Hostx64\x64"
    
    if %ERRORLEVEL% NEQ 0 (
        echo.
        echo CUDA compilation failed, will try to compile CPU-only version
        echo.

        REM Try to compile CPU-only version with g++
        echo Compiling CPU-only version: g++ -o build/sort_cpu_only src/sort_cpu_only.cpp -std=c++14
        g++ -o build/sort_cpu_only src/sort_cpu_only.cpp -std=c++14
        
        if %ERRORLEVEL% NEQ 0 (
            echo.
            echo CPU-only compilation also failed.
            echo.
            echo Please make sure you have:
            echo 1. Visual Studio 2022 with Desktop Development with C++ workload installed
            echo 2. Or MinGW/GCC installed for the CPU-only version
            echo.
            echo For Visual Studio: https://visualstudio.microsoft.com/downloads/
            echo For MinGW: https://sourceforge.net/projects/mingw/
        ) else (
            echo.
            echo CPU-only version compiled successfully! You can run it with: build\sort_cpu_only.exe
        )
    ) else (
        echo.
        echo Build successful with explicit compiler path! The executable is located at build/gpu_sorting.exe
        echo.
        echo You can run it with: build\gpu_sorting.exe
    )
) else (
    echo.
    echo Build successful! The executable is located at build/gpu_sorting.exe
    echo.
    echo You can run it with: build\gpu_sorting.exe
)

echo.
echo Press any key to exit...
pause > nul 