@echo off

set CURDIR=%CD%

set BASEDIR=%~dp0

cd /d %BASEDIR%

:: if reconfiguration is necessary, enable the following line.
:: set EXTRA_CMAKE_OPTION="--fresh"

set TARGET=Build

set CONFIG=%1
if not defined CONFIG (
    set CONFIG=Debug
)

set PLATFORM=x64

set VS="Visual Studio 17 2022"

:: Path to vs tools depends on which version is installed. e.g. "BuildTools", "Community".
set VS_TARGET=%2
if not defined VS_TARGET (
    set VS_TARGET=Community
)

:: Check if path to msbuild. If no, run vs tool bat file.
where msbuild
if not %errorlevel%==0 (
    call "C:\Program Files\Microsoft Visual Studio\2022\%VS_TARGET%\Common7\Tools\VsMSBuildCmd.bat"
)

cmake %EXTRA_CMAKE_OPTION% -S 3rdparty -B 3rdparty\x64 -A %PLATFORM% -DLIBRARY_SUFFIX= -DCMAKE_DEBUG_POSTFIX= || goto error
cmake --build 3rdparty\x64 --config=%CONFIG% -j 4 || goto error

cd /d %CURDIR%

exit /b 0

:error
cd /d %CURDIR%
echo "Error====="
exit /b 1
