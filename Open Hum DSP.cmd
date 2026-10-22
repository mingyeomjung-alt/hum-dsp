@echo off
setlocal
cd /d "%~dp0"
if exist "%~dp0.venv\Scripts\python.exe" (
  "%~dp0.venv\Scripts\python.exe" -X utf8 "%~dp0start_local.py"
) else (
  py -3 -X utf8 "%~dp0start_local.py"
)
if errorlevel 1 (
  echo.
  echo Hum DSP could not start. Python 3.12+ and NumPy are required.
  echo See README.md for the source-install instructions.
  pause
)
endlocal
