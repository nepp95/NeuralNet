@ECHO OFF
SETLOCAL

SET "ROOT=%~dp0.."
IF NOT EXIST "%ROOT%\premake5.lua" GOTO InvalidRoot
IF NOT EXIST "%ROOT%\NeuralNetwork" GOTO InvalidRoot

PUSHD "%ROOT%" || EXIT /B 1

FOR %%D IN (
    "build"
    ".tools"
    ".vcpkg-cache"
) DO IF EXIST "%%~D" RMDIR /S /Q "%%~D"

FOR /D /R %%D IN (__pycache__) DO IF EXIST "%%~D" RMDIR /S /Q "%%~D"

FOR /R %%F IN (
    *.sln
    *.slnx
    *.vcxproj
    *.vcxproj.filters
    *.ninja
    .ninja_deps
    .ninja_log
    compile_commands.json
    *.pyc
    *.pyo
) DO IF EXIST "%%~F" DEL /F /Q "%%~F"

POPD
ECHO Removed all generated build artifacts.
EXIT /B 0

:InvalidRoot
ECHO Refusing to clean unexpected repository root "%ROOT%".
EXIT /B 1
