@echo off
echo ===================================================
echo Cleaning up temporary and build files...
echo ===================================================

:: Clean node_modules
if exist web\node_modules (
    echo Removing web/node_modules...
    rd /s /q web\node_modules
)

:: Clean package-lock.json
if exist web\package-lock.json (
    echo Removing web/package-lock.json...
    del web\package-lock.json
)

:: Clean build directory completely
if exist build (
    echo Removing build directory...
    rd /s /q build
)

:: Clean obj files
echo Removing .obj files...
del /s *.obj

:: Clean executable files
echo Removing .exe files...
del /s *.exe

:: Clean temporary files
echo Removing temporary files...
del /s *.ilk *.pdb *.exp *.lib

:: Clean CUDA temporary files
echo Removing CUDA temporary files...
del /s *.i *.ii *.gpu *.ptx *.cubin *.fatbin

:: Clean VS directories
if exist .vs (
    echo Removing .vs directory...
    rd /s /q .vs
)

if exist out (
    echo Removing out directory...
    rd /s /q out
)

if exist x64 (
    echo Removing x64 directory...
    rd /s /q x64
)

if exist Debug (
    echo Removing Debug directory...
    rd /s /q Debug
)

if exist Release (
    echo Removing Release directory...
    rd /s /q Release
)

echo ===================================================
echo Cleanup complete!
echo =================================================== 