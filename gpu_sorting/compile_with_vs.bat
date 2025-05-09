@echo off
echo Setting up Visual Studio environment and compiling CPU-only version...

REM First, initialize the Visual Studio 2022 Preview environment
call "C:\Program Files\Microsoft Visual Studio\2022\Preview\Common7\Tools\VsDevCmd.bat"

REM Make build directory
if not exist build mkdir build

REM Compile the CPU-only version
cl /EHsc /std:c++14 /W4 /O2 src/sort_cpu_only.cpp /Fe:build/sort_cpu_only.exe

if %ERRORLEVEL% == 0 (
    echo.
    echo Compilation successful! Running the program...
    echo.
    build\sort_cpu_only.exe
) else (
    echo.
    echo Compilation failed with error code %ERRORLEVEL%
    echo.
)

pause 