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

if exist "results\uci_cleveland" rmdir /s /q "results\uci_cleveland"
if exist "results\framingham" rmdir /s /q "results\framingham"
if exist "results\cardio70k" rmdir /s /q "results\cardio70k"

echo.
echo Running SHAP: uci_cleveland
"%PYTHON_EXE%" run_experiments.py --dataset uci_cleveland --csv heart_disease_uci.csv --target num --test-size 0.2 --seed 42 --n-iter 25 --cv-folds 5 --shap --shap-background 200 --shap-samples 200 --shap-local-index 0
if errorlevel 1 goto :fail

echo.
echo Running SHAP: framingham
"%PYTHON_EXE%" run_experiments.py --dataset framingham --csv framingham.csv --target TenYearCHD --test-size 0.2 --seed 42 --n-iter 25 --cv-folds 5 --shap --shap-background 200 --shap-samples 200 --shap-local-index 0
if errorlevel 1 goto :fail

echo.
echo Running SHAP: cardio70k
"%PYTHON_EXE%" run_experiments.py --dataset cardio70k --csv cardio_train.csv --target cardio --test-size 0.2 --seed 42 --n-iter 25 --cv-folds 5 --shap --shap-background 200 --shap-samples 200 --shap-local-index 0
if errorlevel 1 goto :fail

echo.
echo Done.
echo SHAP outputs are in results\{dataset}\ with latest run_id filenames.
pause
exit /b 0

:fail
echo.
echo Failed. See error above.
pause
exit /b 1
