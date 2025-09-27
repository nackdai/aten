@echo off

set CURDIR=%CD%

set BASEDIR=%~dp0

cd /d %BASEDIR%

rem if reconfiguration is necessary, enable the following line.
rem set EXTRA_CMAKE_OPTION="--fresh"

set TARGET=Build

set CONFIG=%1
if not defined CONFIG (
    set CONFIG=Debug
)

set PLATFORM=x64

set VS="Visual Studio 17 2022"

rem Path to vs tools depends on which version is installed. e.g. "BuildTools", "Community".
set VS_TARGET=%2
if not defined VS_TARGET (
    set VS_TARGET=Community
)

rem Check if path to msbuild. If no, run vs tool bat file.
where msbuild
if not %errorlevel%==0 (
	call "C:\Program Files\Microsoft Visual Studio\2022\%VS_TARGET%\Common7\Tools\VsMSBuildCmd.bat"
)

rem glfw =============================

cmake %EXTRA_CMAKE_OPTION% -S glfw -B glfw\x64 -A %PLATFORM% -DGLFW_BUILD_DOCS=FALSE -DGLFW_BUILD_EXAMPLES=FALSE -DGLFW_BUILD_TESTS=FALSE || goto error
cmake --build glfw\x64 --config=%CONFIG% -j 4 || goto error

rem glew =============================

cmake %EXTRA_CMAKE_OPTION% -S glew\build\cmake -B glew\build\vc16 -A %PLATFORM% -DBUILD_UTILS=FALSE || goto error
cmake --build glew\build\vc16 --config=%CONFIG% -j 4 || goto error

set BUILD_DIR=glew\build\vc16
xcopy /Y /D %BUILD_DIR%\lib\%CONFIG% glew\lib\%CONFIG%\%PLATFORM%\ || goto error
xcopy /Y /D %BUILD_DIR%\bin\%CONFIG% glew\bin\%CONFIG%\%PLATFORM%\ || goto error

rem tinyobjloader ====================

cmake %EXTRA_CMAKE_OPTION% -S tinyobjloader -B tinyobjloader\build -A %PLATFORM% || goto error
cmake --build tinyobjloader\build --config=%CONFIG% -j 4 || goto error

rem assimp ==========================

cmake %EXTRA_CMAKE_OPTION% -S assimp -B assimp\build -A %PLATFORM% -DCMAKE_CXX_FLAGS="/wd4834 /wd4819 /utf-8 /EHsc" -DASSIMP_BUILD_TESTS=FALSE -DASSIMP_INSTALL=FALSE -DASSIMP_INSTALL_PDB=FALSE -DLIBRARY_SUFFIX= -DCMAKE_DEBUG_POSTFIX= -DASSIMP_BUILD_ASSIMP_TOOLS=FALSE || goto error
cmake --build assimp\build --config=%CONFIG% -j 4 || goto error

rem googletest =======================

cmake %EXTRA_CMAKE_OPTION% -S googletest -B googletest\build -A %PLATFORM% -DBUILD_SHARED_LIBS=TRUE || goto error
cmake --build googletest\build --config=%CONFIG% -j 4 || goto error

rem openvdb ==========================
rem Nanovdb is header only. There is nothing to build as Debug. Tools are also unnecessary.

rem Copy files for Profile configuration ==============================
if %CONFIG% == Release (
   cd /d %BASEDIR%
   xcopy /Y /D /E glfw\%PLATFORM%\src\Release glfw\%PLATFORM%\src\Profile\ || goto error
   xcopy /Y /D /E glew\lib\Release\%PLATFORM% glew\lib\Profile\%PLATFORM%\ || goto error
   xcopy /Y /D /E tinyobjloader\build\Release tinyobjloader\build\Profile\ || goto error
)

cd /d %CURDIR%

exit /b 0

:error
cd /d %CURDIR%
echo "Error====="
exit /b 1
