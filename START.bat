@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"

echo ==============================================
echo  BYBIT SUPERTREND LAB v0.7.1
echo ==============================================
echo.

where py >nul 2>nul
if %errorlevel%==0 (
  set "PY=py"
) else (
  where python >nul 2>nul
  if not %errorlevel%==0 (
    echo ERROR: Python was not found.
    echo Install Python 3.11+ and enable Add Python to PATH.
    pause
    exit /b 1
  )
  set "PY=python"
)

if not exist ".venv\Scripts\python.exe" (
  echo [1/3] Creating virtual environment...
  %PY% -m venv .venv
  if errorlevel 1 goto :error
) else (
  echo [1/3] Virtual environment already exists.
)

echo [2/3] Checking libraries...
".venv\Scripts\python.exe" -c "import numpy,pandas,requests,matplotlib,numba" >nul 2>nul
if errorlevel 1 (
  echo Libraries are missing. Installing them now.
  echo You will see pip output below - this is normal on the first run.
  echo.
  ".venv\Scripts\python.exe" -m pip install --upgrade pip
  if errorlevel 1 goto :error
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 goto :error
  > ".venv\dependencies_ready.txt" echo OK
) else (
  echo Libraries are ready. Nothing to install.
)

echo [3/3] Starting program...
echo.
".venv\Scripts\python.exe" -u BYBIT_SUPERTREND_LAB.py
if errorlevel 1 goto :error
exit /b 0

:error
echo.
echo ERROR: startup failed. See the messages above.
pause
exit /b 1
