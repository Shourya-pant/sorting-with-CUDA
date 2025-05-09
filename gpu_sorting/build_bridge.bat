@echo off
setlocal enabledelayedexpansion

echo ===================================================
echo Building GPU sorting bridge application...
echo ===================================================

:: Set paths for CUDA and Visual Studio with proper quoting
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Preview\VC\Auxiliary\Build\vcvars64.bat"

:: Check if CUDA exists at specified path
if not exist "%CUDA_PATH%" (
    echo ERROR: CUDA not found at %CUDA_PATH%
    echo Please install CUDA v12.1 or update the path in this script.
    exit /b 1
)

:: Check if Visual Studio exists at specified path
if not exist "%VCVARS%" (
    echo ERROR: Visual Studio vcvars64.bat not found at %VCVARS%
    echo Please install Visual Studio 2022 Preview or update the path in this script.
    exit /b 1
)

:: Create build directory if it doesn't exist
if not exist build mkdir build

:: Initialize Visual Studio environment
echo Initializing Visual Studio environment...
call "%VCVARS%"

:: Set compile flags with proper quoting
set "CUDA_INCLUDES=-I"%CUDA_PATH%\include" -I"include""
set "CUDA_LIBS=cudart.lib"
set "CPP_FLAGS=/EHsc /O2 /nologo"

:: Set CUDA optimization flags
set "CUDA_OPT_FLAGS=--use_fast_math -O3"

:: Compile the CUDA file (sort.cu)
echo Compiling CUDA code...
"%CUDA_PATH%\bin\nvcc.exe" -c -o build/sort.obj src/sort.cu !CUDA_INCLUDES! %CUDA_OPT_FLAGS% -Xcompiler "/EHsc /O2 /nologo"

if %errorlevel% neq 0 (
    echo ERROR: CUDA compilation failed
    exit /b 1
)

:: Compile the C++ bridge program
echo Compiling C++ bridge code...
cl.exe /c %CPP_FLAGS% -I"%CUDA_PATH%\include" -I"include" /Fobuild/sort_bridge.obj src/sort_bridge.cpp

if %errorlevel% neq 0 (
    echo ERROR: C++ compilation failed
    exit /b 1
)

:: Link everything together
echo Linking...
link.exe /OUT:build/gpu_sort_bridge.exe build/sort_bridge.obj build/sort.obj /LIBPATH:"%CUDA_PATH%\lib\x64" %CUDA_LIBS% /nologo

if %errorlevel% neq 0 (
    echo ERROR: Linking failed
    exit /b 1
)

echo ===================================================
echo Build completed successfully!
echo Executable: build\gpu_sort_bridge.exe
echo ===================================================

endlocal