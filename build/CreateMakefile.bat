@echo off

set CURDIR=%CD%

cd /d %~dp0

rem Create Makefile by MakeItSo =======================================

..\3rdparty\makeitso\output\MakeItSo.exe -file=makefile.sln -config=MakeItSo_Libs.config

rem Copy Makefie ======================================================
if %CONFIG% == Release (
   cd /d %~dp0
   xcopy /Y /D ..\vs2015\*.makefile .\
)

cd /d %CURDIR%

exit /b 0

:error
cd /d %CURDIR%
echo "Error====="
pause
exit /b 1