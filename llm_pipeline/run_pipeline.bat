@echo off
REM ==============================================
REM PubMed Statistical Extraction Pipeline Launcher
REM ==============================================

echo.
echo ============================================
echo PubMed Statistical Extraction Pipeline
echo ============================================
echo.

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Display menu
:menu
echo Select an option:
echo.
echo 1. Check Requirements
echo 2. Test Run (100 articles)
echo 3. Small Run (10,000 articles)
echo 4. Full Pipeline (4.48M articles)
echo 5. Resume from Checkpoint
echo 6. Start Ollama Instances
echo 7. Monitor Progress
echo 8. Stop All Processes
echo 9. Exit
echo.

set /p choice="Enter your choice (1-9): "

if "%choice%"=="1" goto check
if "%choice%"=="2" goto test
if "%choice%"=="3" goto small
if "%choice%"=="4" goto full
if "%choice%"=="5" goto resume
if "%choice%"=="6" goto ollama
if "%choice%"=="7" goto monitor
if "%choice%"=="8" goto stop
if "%choice%"=="9" goto end

echo Invalid choice. Please try again.
echo.
goto menu

:check
echo.
echo Checking requirements...
python start_pipeline.py check
pause
goto menu

:test
echo.
echo Starting test run (100 articles)...
python start_pipeline.py test
pause
goto menu

:small
echo.
echo Starting small run (10,000 articles)...
python start_pipeline.py small
pause
goto menu

:full
echo.
echo WARNING: This will process all 4.48M articles and may take 30-40 days!
set /p confirm="Are you sure? (y/n): "
if /i "%confirm%"=="y" (
    python start_pipeline.py full
) else (
    echo Cancelled.
)
pause
goto menu

:resume
echo.
echo Resuming from last checkpoint...
python start_pipeline.py resume
pause
goto menu

:ollama
echo.
echo Starting Ollama instances...
python start_pipeline.py start-ollama
pause
goto menu

:monitor
echo.
echo Monitoring pipeline progress...
python start_pipeline.py monitor
pause
goto menu

:stop
echo.
echo Stopping all processes...
python start_pipeline.py stop-ollama
taskkill /F /IM python.exe /T >nul 2>&1
echo All processes stopped.
pause
goto menu

:end
echo.
echo Goodbye!
exit /b 0
