@echo off
echo Checking CUDA installation...

where nvcc >nul 2>nul
if %ERRORLEVEL% == 0 (
    echo CUDA Toolkit is installed.
    echo Running nvcc --version to check CUDA version:
    nvcc --version
) else (
    echo CUDA Toolkit is not installed or not in PATH.
    echo.
    echo Please install CUDA Toolkit from https://developer.nvidia.com/cuda-downloads
    echo.
    echo After installation, make sure the CUDA bin directory is in your PATH.
    echo.
    echo Once CUDA is installed, you can build the project with:
    echo   1. cd gpu_sorting\build
    echo   2. cmake ..
    echo   3. cmake --build .
    echo.
    echo Checking if you have an NVIDIA GPU...
    
    wmic path win32_VideoController get name | findstr /i "NVIDIA" >nul
    if %ERRORLEVEL% == 0 (
        echo NVIDIA GPU detected. You can install CUDA Toolkit.
    ) else (
        echo No NVIDIA GPU detected. CUDA requires an NVIDIA GPU.
        echo If you believe this is an error, please check your hardware and drivers.
    )
)

echo.
echo Checking for CMake...
where cmake >nul 2>nul
if %ERRORLEVEL% == 0 (
    echo CMake is installed.
    cmake --version
) else (
    echo CMake is not installed or not in PATH.
    echo Please install CMake from https://cmake.org/download/
)

echo.
echo Press any key to exit...
pause > nul 