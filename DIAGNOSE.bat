@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
  set "PY=.venv\Scripts\python.exe"
) else (
  where py >nul 2>nul
  if %errorlevel%==0 (
    set "PY=py"
  ) else (
    set "PY=python"
  )
)

echo Running Bybit connection diagnostic...
"%PY%" DIAGNOSE.py
echo.
echo Result saved to logs\diagnose.txt
pause
