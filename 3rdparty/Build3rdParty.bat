@echo off

set CURDIR=%CD%

set BASEDIR=%~dp0

cd /d %BASEDIR%

set TARGET=Build

set CONFIG=%1
if not defined CONFIG (
    set CONFIG=Debug
)

set PLATFORM=x64

set VS="Visual Studio 16 2019"

rem Path to vs tools depends on which version is installed. e.g. "BuildTools", "Community".
set VS_TARGET=%2
if not defined VS_TARGET (
    set VS_TARGET=Community
)

rem Check if path to msbuild. If no, run vs tool bat file.
where msbuild
if not %errorlevel%==0 (
    call "C:\Program Files (x86)\Microsoft Visual Studio\2019\%VS_TARGET%\Common7\Tools\VsMSBuildCmd.bat"
)

rem glfw =============================

set BUILD_DIR=glfw\x64

if not exist %BUILD_DIR% (
    mkdir %BUILD_DIR%
)

if not exist %BUILD_DIR%\GLFW.sln (
    cd %BUILD_DIR%
    cmake -D GLFW_BUILD_DOCS=FALSE -D GLFW_BUILD_EXAMPLES=FALSE -D GLFW_BUILD_TESTS=FALSE -G %VS% ..\
    cd %BASEDIR%
)

MSBuild %BUILD_DIR%\GLFW.sln /t:%TARGET% /p:Configuration=%CONFIG% /p:Platform=%PLATFORM% || goto error

rem glew =============================

set BUILD_DIR=glew\build\vc16

if not exist %BUILD_DIR% (
    mkdir %BUILD_DIR%
)

if not exist %BUILD_DIR%\glew.sln (
    cd %BUILD_DIR%
    cmake -D BUILD_UTILS=FALSE -G %VS% ..\cmake
    cd %BASEDIR%
)

MSBuild %BUILD_DIR%\glew.sln /t:%TARGET% /p:Configuration=%CONFIG% /p:Platform=%PLATFORM% || goto error
xcopy /Y /D %BUILD_DIR%\lib\%CONFIG% glew\lib\%CONFIG%\%PLATFORM%\
xcopy /Y /D %BUILD_DIR%\bin\%CONFIG% glew\bin\%CONFIG%\%PLATFORM%\

rem tinyobjloader ====================

set BUILD_DIR=tinyobjloader\build

if not exist %BUILD_DIR% (
    mkdir %BUILD_DIR%
)

if not exist %BUILD_DIR%\tinyobjloader.sln (
    cd %BUILD_DIR%
    cmake -G %VS% ..\
    cd %BASEDIR%
)

MSBuild %BUILD_DIR%\tinyobjloader.sln /t:%TARGET% /p:Configuration=%CONFIG% /p:Platform=%PLATFORM% || goto error

rem assimp ==========================

set BUILD_DIR=assimp\build

if not exist %BUILD_DIR% (
    mkdir %BUILD_DIR%
)

if not exist %BUILD_DIR%\Assimp.sln (
    cd %BUILD_DIR%
    cmake -D ASSIMP_BUILD_TESTS=FALSE -D ASSIMP_INSTALL=FALSE -D ASSIMP_INSTALL_PDB=FALSE -D LIBRARY_SUFFIX= -D CMAKE_DEBUG_POSTFIX= -D ASSIMP_BUILD_ASSIMP_TOOLS=FALSE -G %VS% ..\
    cd %BASEDIR%
)

MSBuild %BUILD_DIR%\Assimp.sln /t:%TARGET% /p:Configuration=%CONFIG% /p:Platform=%PLATFORM% || goto error

rem googletest =======================

set BUILD_DIR=googletest\build

if not exist %BUILD_DIR% (
    mkdir %BUILD_DIR%
)

if not exist %BUILD_DIR%\googletest-distribution.sln (
    cd %BUILD_DIR%
    cmake -D BUILD_SHARED_LIBS=TRUE -G %VS% ..\
    cd %BASEDIR%
)

MSBuild %BUILD_DIR%\googletest-distribution.sln /t:%TARGET% /p:Configuration=%CONFIG% /p:Platform=%PLATFORM% || goto error

rem Copy files for Profile configuration ==============================
if %CONFIG% == Release (
   cd /d %BASEDIR%
   xcopy /Y /D /E glfw\%PLATFORM%\src\Release glfw\%PLATFORM%\src\Profile\
   xcopy /Y /D /E glew\lib\Release\%PLATFORM% glew\lib\Profile\%PLATFORM%\
   xcopy /Y /D /E glew\bin\Release\%PLATFORM% glew\bin\Profile\%PLATFORM%\
   xcopy /Y /D /E tinyobjloader\bin\Release tinyobjloader\bin\Profile\
)

cd /d %CURDIR%

exit /b 0

:error
cd /d %CURDIR%
echo "Error====="
exit /b 1
