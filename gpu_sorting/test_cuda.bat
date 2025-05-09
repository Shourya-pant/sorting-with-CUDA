@echo off
echo Testing CUDA compilation and execution...

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

REM Compile the CUDA test
echo.
echo Compiling CUDA test...
nvcc -o build/cuda_test src/cuda_test.cu

if %ERRORLEVEL% == 0 (
    echo.
    echo Compilation successful! Running CUDA test...
    echo.
    REM Run the CUDA test
    build\cuda_test.exe
) else (
    echo.
    echo Compilation failed with error code %ERRORLEVEL%
)

:end
echo.
echo Press any key to exit...
pause > nul 