@echo off
echo Building GPU Sorting project with NVCC...

REM First setup Visual Studio environment
echo Setting up Visual Studio environment...
call setup_vs_env.bat --no-pause

REM Check if Visual Studio setup was successful
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Visual Studio setup failed. Cannot continue.
    goto :end
)

REM Create build directory if it doesn't exist
if not exist build mkdir build

echo.
echo Using CUDA compiler:
nvcc --version
echo.

REM Compile the project using NVCC with verbose output
echo Compiling with command: nvcc -I./include -o build/gpu_sorting src/main.cpp src/sort.cu -std=c++14
echo.
nvcc -I./include -o build/gpu_sorting src/main.cpp src/sort.cu -std=c++14

if %ERRORLEVEL% == 0 (
    echo.
    echo Build successful! The executable is located at build/gpu_sorting.exe
    echo.
    echo You can run it with: build\gpu_sorting.exe
) else (
    echo.
    echo Build failed with error code %ERRORLEVEL%
    echo.
    echo Please check the error messages above.
    echo.
    echo Troubleshooting steps:
    echo 1. Ensure CUDA path is correctly set in your PATH
    echo 2. Check if you have the right compiler compatible with your CUDA version
    echo 3. Make sure your GPU supports CUDA
)

:end
echo.
echo Press any key to exit...
pause > nul 