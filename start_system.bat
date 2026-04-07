@echo off
setlocal

cd /d "%~dp0"

if not exist ".tmp" mkdir ".tmp"
set "TMP=%~dp0.tmp"
set "TEMP=%~dp0.tmp"

set "PYTHON_EXE=%USERPROFILE%\python-sdk\python3.13.2\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"
if exist "%~dp0.venv\Scripts\python.exe" set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if exist "%~dp0venv\Scripts\python.exe" set "PYTHON_EXE=%~dp0venv\Scripts\python.exe"
if exist "%~dp0env\Scripts\python.exe" set "PYTHON_EXE=%~dp0env\Scripts\python.exe"

"%PYTHON_EXE%" -c "import sys; print(sys.executable)" >nul 2>nul
if errorlevel 1 (
  echo Python is not runnable.
  echo Current PYTHON_EXE = %PYTHON_EXE%
  pause
  exit /b 1
)

echo Using Python: %PYTHON_EXE%

if not exist "artifacts\uci_cleveland\model.joblib" goto :build
if not exist "artifacts\framingham\model.joblib" goto :build
if not exist "artifacts\cardio70k\model.joblib" goto :build
goto :start

:build
echo.
echo Artifacts not found. Building artifacts first...
"%PYTHON_EXE%" build_system_artifacts.py
if errorlevel 1 (
  echo Failed to build artifacts.
  pause
  exit /b 1
)

:start
echo.
echo Starting Streamlit app...
"%PYTHON_EXE%" -m streamlit run app.py
if errorlevel 1 (
  echo Streamlit exited with error.
  pause
  exit /b 1
)

exit /b 0
