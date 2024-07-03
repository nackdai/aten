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

cmake -S glfw -B glfw\x64 -D GLFW_BUILD_DOCS=FALSE -D GLFW_BUILD_EXAMPLES=FALSE -D GLFW_BUILD_TESTS=FALSE
cmake --build glfw\x64 --config=%CONFIG%

rem glew =============================

cmake -S glew\build\cmake -B glew\build\vc16 -D BUILD_UTILS=FALSE
cmake --build glew\build\vc16 --config=%CONFIG%

set BUILD_DIR=glew\build\vc16
xcopy /Y /D %BUILD_DIR%\lib\%CONFIG% glew\lib\%CONFIG%\%PLATFORM%\
xcopy /Y /D %BUILD_DIR%\bin\%CONFIG% glew\bin\%CONFIG%\%PLATFORM%\

rem tinyobjloader ====================

cmake -S tinyobjloader -B tinyobjloader\build
cmake --build tinyobjloader\build --config=%CONFIG%

rem assimp ==========================

cmake -S assimp -B assimp\build -D ASSIMP_BUILD_TESTS=FALSE -D ASSIMP_INSTALL=FALSE -D ASSIMP_INSTALL_PDB=FALSE -D LIBRARY_SUFFIX= -D CMAKE_DEBUG_POSTFIX= -D ASSIMP_BUILD_ASSIMP_TOOLS=FALSE
cmake --build assimp\build --config=%CONFIG%

rem googletest =======================

cmake -S googletest -B googletest\build -D BUILD_SHARED_LIBS=TRUE
cmake --build googletest\build --config=%CONFIG%

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
