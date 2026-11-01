@echo off
setlocal
cd /d "%~dp0"
echo Chameleon - quality-gated face privacy experiment
echo 1. Gentle - recommended start; strictest visual-quality gate
echo 2. Balanced - more face change, still quality-gated
echo 3. Strong - largest allowed local face change
choice /c 123 /n /m "Choose strength [1-3]: "
set "MODE=gentle"
if errorlevel 2 set "MODE=balanced"
if errorlevel 3 set "MODE=strong"
"%~dp0.venv-defense\Scripts\python.exe" "%~dp0Chameleon_Defense.py" --mode %MODE% %*
if errorlevel 1 echo Processing failed. See the error above.
pause
