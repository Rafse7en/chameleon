@echo off
setlocal EnableDelayedExpansion
REM Chameleon Engine Launcher v2.1.1
REM Usage: double-click and type/paste a path, OR drag-and-drop images onto this .bat
REM Modes: subtle=cleanest/weakest, balanced=default, strong, max=most aggressive

set "SCRIPT_DIR=%~dp0"
set "ENGINE=%SCRIPT_DIR%Chameleon_Engine.py"
set "MODE=balanced"
REM Silence harmless OpenCV DNN backend notices
set "OPENCV_LOG_LEVEL=ERROR"
set "OPENCV_DNN_LOG_LEVEL=ERROR"

echo [Chameleon Engine v2.1.1]
echo =======================
echo Experimental perturbations. Anti-training protection is UNVERIFIED.
echo Face verification does not measure likeness or training protection.

REM --- 1. Verify Python ---
python --version >nul 2>&1
if ERRORLEVEL 1 (
    echo [ERROR] Python 3.8+ is required but not found.
    echo Please install Python from https://www.python.org/downloads/
    echo Make sure to tick "Add python.exe to PATH" during install.
    pause
    exit /b 1
)

REM --- 2. Verify engine script exists ---
if not exist "%ENGINE%" (
    echo [ERROR] Chameleon_Engine.py not found next to this .bat file.
    echo Expected: %ENGINE%
    pause
    exit /b 1
)

REM --- 3. Verify Python dependencies (numpy, PIL, cv2) ---
python -c "import numpy, PIL, cv2" >nul 2>&1
if NOT ERRORLEVEL 1 goto deps_ok
    echo [SETUP] Missing Python packages detected.
    echo [SETUP] Installing: numpy pillow opencv-python ...
    python -m pip install --upgrade pip
    python -m pip install numpy pillow opencv-python
    if ERRORLEVEL 1 (
        echo [ERROR] Failed to install dependencies.
        echo Try manually: python -m pip install numpy pillow opencv-python
        pause
        exit /b 1
    )
    REM Re-check after install
    python -c "import numpy, PIL, cv2" >nul 2>&1
    if ERRORLEVEL 1 (
        echo [ERROR] Dependencies still missing after install.
        pause
        exit /b 1
    )
    echo [SETUP] Dependencies installed OK.
    echo.
:deps_ok

REM --- 4. Collect files to process ---
REM If no args (plain double-click), ask for a path instead of just exiting.
if NOT "%~1"=="" goto process_args

:manual_prompt
    echo No files were passed on the command line.
    echo.
    echo Supported formats: JPG, PNG, WEBP, BMP, TIFF
    echo.
    echo Option A: drag one or more image files onto this .bat file and run again.
    echo Option B: type or paste an image path now.
    echo.
    set /p "MANUAL_INPUT=Image path - Enter to quit: "
    if "!MANUAL_INPUT!"=="" (
        echo Exiting.
        pause
        exit /b 0
    )
    call :strip_and_process
    echo.
    echo Done. Check the input folder for the *_chameleon.jpg output.
    pause
    exit /b 0

REM --- 5. Process all dropped/command-line files ---
REM Use SHIFT loop (not FOR IN (%*)) so paths with spaces work.
:process_args
set "COUNT=0"
set "FAILED=0"

:next_arg
if "%~1"=="" goto done_args
call :process_one "%~1"
shift
goto next_arg

:done_args
echo.
echo =======================
echo Finished: %COUNT% file(s) processed, %FAILED% failed.
echo Outputs are saved next to each input as *_chameleon.jpg
pause
exit /b 0

REM ================= Subroutine =================
:process_one
REM %~1 = full input path (quotes stripped by %~1, re-added below)
set "INFILE=%~1"
if "%INFILE%"=="" exit /b 0
if not exist "%INFILE%" (
    echo [SKIP] Not found: %INFILE%
    set /a FAILED+=1
    exit /b 0
)
set "BASENAME=%~n1"
set "INPUT_DIR=%~dp1"
set "OUTPUT_FILE=!INPUT_DIR!!BASENAME!_chameleon.jpg"

set /a COUNT+=1
echo.
echo [%COUNT%] Processing: "%INFILE%"
echo      -^> "!OUTPUT_FILE!"

python "%ENGINE%" "%INFILE%" -o "!OUTPUT_FILE!" --mode %MODE% --verify
if ERRORLEVEL 1 (
    echo [FAILED] !BASENAME!
    set /a FAILED+=1
) else (
    echo [COMPLETE] !BASENAME! -^> !BASENAME!_chameleon.jpg
)
exit /b 0

:strip_and_process
REM Runs outside any parenthesised IF block so quote-stripping is safe.
set "MANUAL_INPUT=!MANUAL_INPUT:"=!"
if not exist "!MANUAL_INPUT!" (
    echo [ERROR] File not found: "!MANUAL_INPUT!"
    pause
    exit /b 1
)
call :process_one "!MANUAL_INPUT!"
exit /b 0
