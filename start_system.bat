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
pause
exit /b 1

:python_ok

echo Using Python: %PYTHON_EXE% %PYTHON_ARGS%

if not exist "artifacts\uci_cleveland\model.joblib" goto :build
if not exist "artifacts\framingham\model.joblib" goto :build
if not exist "artifacts\cardio70k\model.joblib" goto :build
goto :start

:build
echo.
echo Artifacts not found. Building artifacts first...
"%PYTHON_EXE%" %PYTHON_ARGS% build_system_artifacts.py
if errorlevel 1 (
  echo Failed to build artifacts.
  pause
  exit /b 1
)

:start
echo.
echo Starting Streamlit app...
"%PYTHON_EXE%" %PYTHON_ARGS% -m streamlit run app.py
if errorlevel 1 (
  echo Streamlit exited with error.
  pause
  exit /b 1
)

exit /b 0

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
