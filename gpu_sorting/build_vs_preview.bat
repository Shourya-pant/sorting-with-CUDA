@echo off
echo Building GPU Sorting project with NVCC using Visual Studio 2022 Preview...

REM Create build directory if it doesn't exist
if not exist build mkdir build

REM Get CUDA information
echo CUDA Information:
nvcc --version

REM Determine GPU architecture
echo.
echo Checking for NVIDIA GPU...
"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe" -L 2>nul

REM Compile the project using NVCC with Visual Studio 2022 Preview compiler
echo.
echo Compiling with command: nvcc -I./include -o build/gpu_sorting src/main.cpp src/sort.cu -std=c++14 -ccbin="C:\Program Files\Microsoft Visual Studio\2022\Preview\VC\Tools\MSVC\14.44.34918\bin\Hostx64\x64"
echo.

nvcc -I./include -o build/gpu_sorting src/main.cpp src/sort.cu -std=c++14 -ccbin="C:\Program Files\Microsoft Visual Studio\2022\Preview\VC\Tools\MSVC\14.44.34918\bin\Hostx64\x64"

if %ERRORLEVEL% == 0 (
    echo.
    echo Build successful! The executable is located at build/gpu_sorting.exe
    echo.
    echo You can run it with: build\gpu_sorting.exe
) else (
    echo.
    echo Build failed with error code %ERRORLEVEL%
    echo.
    echo Please check:
    echo 1. Your Visual Studio 2022 Preview installation is complete with C++ development tools
    echo 2. The path to the compiler is correct
    echo 3. Your CUDA installation is compatible with Visual Studio 2022 Preview
)

echo.
echo Press any key to exit...
pause > nul 