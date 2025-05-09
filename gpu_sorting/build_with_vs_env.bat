@echo off
echo Setting up Visual Studio environment and building GPU Sorting project with CUDA 12.6...

REM First, initialize the Visual Studio 2022 Preview environment
call "C:\Program Files\Microsoft Visual Studio\2022\Preview\Common7\Tools\VsDevCmd.bat"

REM Create build directory if it doesn't exist
if not exist build mkdir build

REM Get CUDA information
echo CUDA Information:
nvcc --version

REM Try to get GPU information
echo.
echo Checking for NVIDIA GPU...
"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe" -L 2>nul

REM Compile the project using NVCC
echo.
echo Compiling with command: nvcc -I./include -o build/gpu_sorting src/main.cpp src/sort.cu -std=c++14
echo.

nvcc -I./include -o build/gpu_sorting src/main.cpp src/sort.cu -std=c++14

if %ERRORLEVEL% == 0 (
    echo.
    echo Build successful! The executable is located at build/gpu_sorting.exe
    echo.
    echo Running the program...
    build\gpu_sorting.exe
) else (
    echo.
    echo Build failed with error code %ERRORLEVEL%
    echo.
    echo Please check:
    echo 1. Your Visual Studio 2022 Preview installation is complete with C++ development tools
    echo 2. The path to the compiler is correct
    echo 3. Your CUDA installation is properly set up
)

echo.
echo Press any key to exit...
pause > nul 