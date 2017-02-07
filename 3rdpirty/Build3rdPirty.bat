@echo off

set CURDIR=%CD%

cd /d %~dp0

set MSBUILD="C:\Program Files (x86)\MSBuild\14.0\Bin\MSBuild.exe"

set TARGET=Build
set CONFIG=%1
set PLATFORM=%2

if not defined CONFIG (
    set CONFIG=Debug
)

if not defined PLATFORM (
    set PLATFORM=Win32
)

rem glfw =============================

set BUILD_DIR=glfw\%PLATFORM%
set ROOT=..\..

if not exist %BUILD_DIR% (
    mkdir %BUILD_DIR%
)

if not exist %BUILD_DIR%\GLFW.sln (
    cd %BUILD_DIR%
    if %PLATFORM% == Win32 (
        %ROOT%\cmake\bin\cmake.exe -G "Visual Studio 14 2015" ..\
    ) else (
        %ROOT%\cmake\bin\cmake.exe -G "Visual Studio 14 2015 Win64" ..\
    )
    cd %ROOT%
)

%MSBUILD% %BUILD_DIR%\GLFW.sln /t:%TARGET% /p:Configuration=%CONFIG% /p:Platform=%PLATFORM% || goto error

rem end ==============================

cd /d %CURDIR%

exit /b 0

:error
cd /d %CURDIR%
echo "Error====="
pause
exist /b 1