@echo off

set CURDIR=%CD%

cd /d %~dp0

rem Create Makefile by MakeItSo =======================================

..\3rdparty\makeitso\output\MakeItSo.exe -file=..\vs2015\aten.sln -config=MakeItSo.config -nl=crlf

rem Copy Makefie ======================================================
xcopy /Y /D ..\vs2015\*.makefile .\
xcopy /Y /D ..\vs2015\Makefile .\

cd /d %CURDIR%

exit /b 0

:error
cd /d %CURDIR%
echo "Error====="
pause
exit /b 1