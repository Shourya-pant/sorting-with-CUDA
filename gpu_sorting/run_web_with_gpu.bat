@echo off
echo ===================================================
echo Building GPU sorting bridge application...
echo ===================================================

:: Check if CUDA is installed in the expected path
if not exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1" (
    echo ERROR: CUDA v12.1 not found at expected location.
    echo Please install CUDA v12.1 or update the path in build_bridge.bat
    pause
    exit /b 1
)

:: Build the CUDA bridge application
echo Running build_bridge.bat...
call build_bridge.bat

if %errorlevel% neq 0 (
    echo ERROR: Failed to build the CUDA bridge. See above for error messages.
    pause
    exit /b 1
)

echo.
echo ===================================================
echo Starting Bitonic Sort Web Application with native CUDA support...
echo ===================================================

:: Change to web directory
cd web

:: Verify bridge executable exists
if not exist "../build/gpu_sort_bridge.exe" (
    echo ERROR: Bridge executable not found at ../build/gpu_sort_bridge.exe
    echo Build might have failed or output to a different location.
    pause
    exit /b 1
)

:: Install dependencies
echo Installing Node.js dependencies...
call npm install

if %errorlevel% neq 0 (
    echo ERROR: Failed to install Node.js dependencies.
    echo Make sure Node.js is installed and in your PATH.
    pause
    exit /b 1
)

:: Start the web application
echo Starting web server...
call npm run start-browser

echo.
echo ===================================================
echo Web application is running at http://localhost:3000
echo Press Ctrl+C to stop the server when done.
echo =================================================== 