@echo off
echo Setting up Visual Studio environment for CUDA...

REM Check for Visual Studio installation paths
set VS_PATHS=^
"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" ^
"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat" ^
"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" ^
"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" ^
"C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvarsall.bat" ^
"C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" ^
"C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsall.bat" ^
"C:\Program Files (x86)\Microsoft Visual Studio\2017\Professional\VC\Auxiliary\Build\vcvarsall.bat" ^
"C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\VC\Auxiliary\Build\vcvarsall.bat"

set VS_PATH_FOUND=0

for %%p in (%VS_PATHS%) do (
    if exist %%p (
        echo Found Visual Studio at: %%p
        echo Setting up environment for x64 architecture...
        call %%p x64
        set VS_PATH_FOUND=1
        goto :found
    )
)

:found
if %VS_PATH_FOUND% == 0 (
    echo ERROR: Could not find Visual Studio.
    echo Please install Visual Studio with C++ development tools.
    echo Download from: https://visualstudio.microsoft.com/downloads/
    exit /b 1
)

echo Visual Studio environment set up successfully.
echo You can now compile CUDA programs with nvcc.

REM Test if cl.exe is in PATH
where cl.exe >nul 2>nul
if %ERRORLEVEL% == 0 (
    echo C++ compiler (cl.exe) found in PATH.
) else (
    echo WARNING: C++ compiler (cl.exe) not found in PATH.
    echo Visual Studio environment setup may have failed.
)

echo.
echo To compile the CUDA test, run: nvcc -o build/cuda_test src/cuda_test.cu
echo.

REM Don't exit automatically if running as a separate script
if "%1"=="--no-pause" goto :eof
echo Press any key to exit...
pause > nul 