@echo off
echo Testing CUDA compilation with Visual Studio 2022 Preview...

REM Create build directory if it doesn't exist
if not exist build mkdir build

echo.
echo Compiling test program...
nvcc -allow-unsupported-compiler -I./include -o build/gpu_sorting src/main.cpp src/sort.cu -ccbin="C:\Program Files\Microsoft Visual Studio\2022\Preview\VC\Tools\MSVC\14.44.34918\bin\Hostx64\x64"

if %ERRORLEVEL% == 0 (
    echo Build successful!
) else (
    echo Build failed with error code %ERRORLEVEL%
)

pause 