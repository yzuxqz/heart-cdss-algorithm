@echo off
setlocal

cd /d "%~dp0"

if not exist ".tmp" mkdir ".tmp"
set "TMP=%~dp0.tmp"
set "TEMP=%~dp0.tmp"

REM Set your Python path here (recommended).
set "PYTHON_EXE=%USERPROFILE%\python-sdk\python3.13.2\python.exe"

if not exist "%PYTHON_EXE%" (
  set "PYTHON_EXE=python"
)
if exist "%~dp0.venv\Scripts\python.exe" set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if exist "%~dp0venv\Scripts\python.exe" set "PYTHON_EXE=%~dp0venv\Scripts\python.exe"
if exist "%~dp0env\Scripts\python.exe" set "PYTHON_EXE=%~dp0env\Scripts\python.exe"
set "PYTHON_ARGS="

call :check_python
if errorlevel 1 goto :try_py
goto :python_ok

:try_py
where py >nul 2>nul
if not errorlevel 1 (
  set "PYTHON_EXE=py"
  set "PYTHON_ARGS=-3"
  call :check_python
  if not errorlevel 1 goto :python_ok
)

call :find_common_python
if not errorlevel 1 (
  call :check_python
  if not errorlevel 1 goto :python_ok
)

goto :python_fail

:python_fail
echo Python is not runnable.
echo Current PYTHON_EXE = %PYTHON_EXE% %PYTHON_ARGS%
echo Please set a valid python.exe path in this bat file.
pause
exit /b 1

:python_ok

echo Using Python: %PYTHON_EXE% %PYTHON_ARGS%

if exist "results\uci_cleveland" rmdir /s /q "results\uci_cleveland"
if exist "results\framingham" rmdir /s /q "results\framingham"
if exist "results\cardio70k" rmdir /s /q "results\cardio70k"

if exist "artifacts\uci_cleveland" rmdir /s /q "artifacts\uci_cleveland"
if exist "artifacts\framingham" rmdir /s /q "artifacts\framingham"
if exist "artifacts\cardio70k" rmdir /s /q "artifacts\cardio70k"

echo.
echo Running: uci_cleveland
"%PYTHON_EXE%" %PYTHON_ARGS% run_experiments.py --dataset uci_cleveland --csv heart_disease_uci.csv --target num --test-size 0.2 --seed 42 --n-iter 25 --cv-folds 5
if errorlevel 1 goto :fail

echo.
echo Running: framingham
"%PYTHON_EXE%" %PYTHON_ARGS% run_experiments.py --dataset framingham --csv framingham.csv --target TenYearCHD --test-size 0.2 --seed 42 --n-iter 25 --cv-folds 5
if errorlevel 1 goto :fail

echo.
echo Running: cardio70k
"%PYTHON_EXE%" %PYTHON_ARGS% run_experiments.py --dataset cardio70k --csv cardio_train.csv --target cardio --test-size 0.2 --seed 42 --n-iter 25 --cv-folds 5
if errorlevel 1 goto :fail

echo.
echo Building artifacts
"%PYTHON_EXE%" %PYTHON_ARGS% build_system_artifacts.py
if errorlevel 1 goto :fail

echo.
echo Done.
echo Results: results\
echo Artifacts: artifacts\
pause
exit /b 0

:fail
echo.
echo Failed. See error above.
pause
exit /b 1

:check_python
"%PYTHON_EXE%" %PYTHON_ARGS% -c "import sys; print(sys.executable)" >nul 2>nul
exit /b %errorlevel%

:find_common_python
for /d %%D in ("%LocalAppData%\Programs\Python\Python*") do (
  if exist "%%D\python.exe" (
    set "PYTHON_EXE=%%D\python.exe"
    set "PYTHON_ARGS="
    exit /b 0
  )
)
for /d %%D in ("%ProgramFiles%\Python*") do (
  if exist "%%D\python.exe" (
    set "PYTHON_EXE=%%D\python.exe"
    set "PYTHON_ARGS="
    exit /b 0
  )
)
for /d %%D in ("%ProgramFiles(x86)%\Python*") do (
  if exist "%%D\python.exe" (
    set "PYTHON_EXE=%%D\python.exe"
    set "PYTHON_ARGS="
    exit /b 0
  )
)
exit /b 1
