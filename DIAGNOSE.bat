@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PY="
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if not defined PY (
  where py >nul 2>nul
  if not errorlevel 1 set "PY=py"
)
if not defined PY set "PY=python"

echo Running Bybit connection diagnostic...
"%PY%" DIAGNOSE.py
echo.
echo Result saved to logs\diagnose.txt
pause
