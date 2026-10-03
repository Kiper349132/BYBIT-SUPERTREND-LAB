@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" SELF_TEST.py
) else (
  where py >nul 2>nul
  if %errorlevel%==0 (py SELF_TEST.py) else (python SELF_TEST.py)
)
pause
