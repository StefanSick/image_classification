@echo off
setlocal enabledelayedexpansion

echo --- Validating Conda Installation ---

:: Check if conda command works
where conda >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Conda is required but was not found in PATH.
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('conda --version') do set CONDA_VER=%%i
echo Conda detected: %CONDA_VER%

echo --- Validating Tools Environment ---

:: Check if 'conda-tools' exists in env list
call conda env list | findstr /b /c:"conda-tools " >nul
if %ERRORLEVEL% neq 0 (
    echo Required environment 'conda-tools' not found.
    echo Run the following command once:
    echo.
    echo    conda create -n conda-tools -c conda-forge mamba -y
    echo.
    pause
    exit /b 1
)

echo conda-tools environment found.

if not exist environment.yml (
    echo environment.yml not found in the current directory.
    pause
    exit /b 1
)

echo --- Creating / Updating Project Environment ---

:: Extract name from environment.yml (searches for "name: " and takes second part)
for /f "tokens=2" %%a in ('findstr "name:" environment.yml') do set ENV_NAME=%%a

:: Check if project environment exists
call conda env list | findstr /b /c:"%ENV_NAME% " >nul
if %ERRORLEVEL% neq 0 (
    echo Environment '%ENV_NAME%' not found. Creating...
    call conda run -n conda-tools mamba env create -f environment.yml
) else (
    echo Environment '%ENV_NAME%' exists. Updating...
    call conda run -n conda-tools mamba env update -f environment.yml --prune
)

echo Process complete.
pause