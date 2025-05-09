@echo off
echo Building CPU-only version of Bitonic Sort...

REM Create build directory if it doesn't exist
if not exist build mkdir build

REM Compile the CPU-only program
g++ -o build/sort_cpu_only src/sort_cpu_only.cpp -std=c++14

if %ERRORLEVEL% == 0 (
    echo.
    echo Build successful! The executable is located at build/sort_cpu_only.exe
    echo.
    echo Running program with default settings...
    echo.
    build\sort_cpu_only.exe
) else (
    echo.
    echo Build failed with error code %ERRORLEVEL%
    echo.
    echo If you don't have g++ installed, you can install MinGW or use Visual Studio.
    echo Download MinGW: https://sourceforge.net/projects/mingw/
)

echo.
echo Press any key to exit...
pause > nul 