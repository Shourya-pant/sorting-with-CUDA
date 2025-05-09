@echo off
echo Building Interactive Bitonic Sort...

REM First, initialize the Visual Studio 2022 Preview environment
call "C:\Program Files\Microsoft Visual Studio\2022\Preview\Common7\Tools\VsDevCmd.bat"

REM Create build directory if it doesn't exist
if not exist build mkdir build

REM Compile the interactive version
cl /EHsc /std:c++14 /W4 /O2 src/sort_with_io.cpp /Fe:build/sort_interactive.exe

if %ERRORLEVEL% == 0 (
    echo.
    echo Compilation successful! Running the interactive program...
    echo.
    build\sort_interactive.exe
) else (
    echo.
    echo Compilation failed with error code %ERRORLEVEL%
    echo.
)

echo.
echo Press any key to exit...
pause > nul 