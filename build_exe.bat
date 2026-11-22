@echo off
setlocal
cd /d "%~dp0"

echo =====================================
echo Building Student Performance System
echo =====================================

set "PY=%CD%\.venv\Scripts\python.exe"

if not exist "%PY%" (
    echo Virtual environment not found!
    echo Please create your .venv first.
    pause
    exit /b 1
)

echo Installing PyInstaller...

"%PY%" -m pip install pyinstaller

echo.
echo Building executable...

"%PY%" -m PyInstaller --noconfirm --clean --onedir ^
--name StudentPerformancePrediction ^
--collect-all streamlit ^
--collect-all sklearn ^
--collect-all matplotlib ^
--collect-all pandas ^
--collect-all numpy ^
--collect-all altair ^
--collect-all pydeck ^
--add-data "app.py;." ^
--add-data "data;data" ^
launcher.py

if errorlevel 1 (
    echo.
    echo BUILD FAILED!
    pause
    exit /b 1
)

echo.
echo =====================================
echo BUILD SUCCESSFUL!
echo =====================================
echo.
echo Find your EXE inside:
echo dist\StudentPerformancePrediction\
echo.
pause
