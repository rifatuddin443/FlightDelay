@echo off
REM Weather Impact Comparison - Quick Start Script (Windows)
REM =========================================================
REM
REM This script runs the complete weather analysis pipeline in one command.
REM
REM Usage:
REM   run_weather_comparison.bat
REM   run_weather_comparison.bat --epochs 100 --classifier both
REM
REM Prerequisites:
REM   - Python 3.8+ in PATH
REM   - PyTorch, tsai, PyG installed
REM   - Weather data files (weather_cn.npy or similar)

setlocal enabledelayedexpansion

REM Default parameters
set EPOCHS=50
set CLASSIFIER=TSiTPlus
set BATCH_SIZE=16
set LR=1e-4
set DATA_SOURCE=cdata
set DEVICE=auto
set SKIP_ANALYSIS=0
set PLOT=0

REM Parse arguments
:parse_args
if "%1"=="" goto args_done
if "%1"=="--epochs" (
    set EPOCHS=%2
    shift
    shift
    goto parse_args
)
if "%1"=="--classifier" (
    set CLASSIFIER=%2
    shift
    shift
    goto parse_args
)
if "%1"=="--batch_size" (
    set BATCH_SIZE=%2
    shift
    shift
    goto parse_args
)
if "%1"=="--lr" (
    set LR=%2
    shift
    shift
    goto parse_args
)
if "%1"=="--data_source" (
    set DATA_SOURCE=%2
    shift
    shift
    goto parse_args
)
if "%1"=="--device" (
    set DEVICE=%2
    shift
    shift
    goto parse_args
)
if "%1"=="--no-analysis" (
    set SKIP_ANALYSIS=1
    shift
    goto parse_args
)
if "%1"=="--plot" (
    set PLOT=1
    shift
    goto parse_args
)
if "%1"=="--help" (
    goto show_help
)
echo Unknown option: %1
goto error_exit

:show_help
cls
echo.
echo Weather Impact Comparison - Quick Start
echo ========================================
echo.
echo Usage: run_weather_comparison.bat [OPTIONS]
echo.
echo Options:
echo   --epochs N              Number of training epochs (default: 50^)
echo   --classifier NAME       TSiTPlus, ConvTranPlus, or both (default: TSiTPlus^)
echo   --batch_size N          Batch size (default: 16^)
echo   --lr LR                 Learning rate (default: 1e-4^)
echo   --data_source SOURCE    cdata or udata (default: cdata^)
echo   --device DEVICE         cuda, cpu, or auto (default: auto^)
echo   --no-analysis           Skip analysis step
echo   --plot                  Generate plots (requires matplotlib^)
echo   --help                  Show this help message
echo.
echo Examples:
echo   run_weather_comparison.bat --epochs 100
echo   run_weather_comparison.bat --classifier both --epochs 50 --plot
echo   run_weather_comparison.bat --batch_size 32 --lr 5e-5
echo.
exit /b 0

:args_done
cls
echo.
echo ===============================================================
echo         WEATHER IMPACT COMPARISON - QUICK START
echo ===============================================================
echo.

echo Configuration:
echo   Epochs:       %EPOCHS%
echo   Classifier:   %CLASSIFIER%
echo   Batch size:   %BATCH_SIZE%
echo   Learning rate: %LR%
echo   Data source:  %DATA_SOURCE%
echo   Device:       %DEVICE%
echo.

echo Checking dependencies...

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please ensure Python is in your PATH.
    exit /b 1
)

REM Check required Python packages
python -c "import torch" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] PyTorch not found. Install with: pip install torch
    exit /b 1
)

python -c "import torch_geometric" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] PyTorch Geometric not found. Install with: pip install torch-geometric
    exit /b 1
)

REM Check comparison script exists
if not exist "stacked_gru_transformer_weather_comparison.py" (
    echo [ERROR] stacked_gru_transformer_weather_comparison.py not found
    echo Make sure you are in the STPN-main directory
    exit /b 1
)

if not exist "analyze_weather_comparison.py" (
    echo [WARNING] analyze_weather_comparison.py not found
    set SKIP_ANALYSIS=1
)

echo [OK] All dependencies found
echo.

REM Run main comparison
echo Step 1/2: Running comparison experiment...
echo Command: python stacked_gru_transformer_weather_comparison.py
echo          --epochs %EPOCHS% --classifier %CLASSIFIER% --batch_size %BATCH_SIZE%
echo          --lr %LR% --data_source %DATA_SOURCE% --device %DEVICE%
echo.

python stacked_gru_transformer_weather_comparison.py ^
    --epochs %EPOCHS% ^
    --classifier %CLASSIFIER% ^
    --batch_size %BATCH_SIZE% ^
    --lr %LR% ^
    --data_source %DATA_SOURCE% ^
    --device %DEVICE%

if errorlevel 1 (
    echo [ERROR] Comparison script failed
    exit /b 1
)

REM Find latest results directory
for /f "tokens=*" %%A in ('dir /b /ad /t:d /o:-d weather_comparison_* 2^>nul ^| findstr /r "weather_comparison_[0-9]" ^| head -1') do (
    set RESULT_DIR=%%A
)

if not defined RESULT_DIR (
    echo [ERROR] No results directory found
    exit /b 1
)

echo.
echo [OK] Comparison complete
echo Results saved to: %RESULT_DIR%
echo.

REM Run analysis
if "%SKIP_ANALYSIS%"=="0" (
    echo Step 2/2: Analyzing results...
    echo.
    
    if "%PLOT%"=="1" (
        python analyze_weather_comparison.py %RESULT_DIR% --plot
    ) else (
        python analyze_weather_comparison.py %RESULT_DIR%
    )
    
    echo.
)

REM Show summary
if exist "%RESULT_DIR%\WEATHER_COMPARISON_SUMMARY.csv" (
    echo.
    echo Summary Results (WEATHER_COMPARISON_SUMMARY.csv^):
    echo.
    type "%RESULT_DIR%\WEATHER_COMPARISON_SUMMARY.csv" | findstr /v "^$"
    echo.
)

REM Final summary
echo ===============================================================
echo                    WORKFLOW COMPLETE
echo ===============================================================
echo.
echo Output directory:  %RESULT_DIR%\
echo.
echo Key outputs:
echo   * WEATHER_COMPARISON_SUMMARY.csv   - Main results
echo   * {classifier}_WITH_weather_*.csv  - Detailed metrics
echo   * {classifier}_NO_weather_*.csv    - Baseline metrics
echo   * *.pth                            - Model checkpoints
if exist "%RESULT_DIR%\weather_comparison_plots.png" (
    echo   * weather_comparison_plots.png     - Visualizations
)
echo.
echo Next steps:
echo   1. Review results in: %RESULT_DIR%\
echo   2. Check WEATHER_COMPARISON_SUMMARY.csv for impact metrics
echo   3. Run analysis again for detailed report:
echo      python analyze_weather_comparison.py %RESULT_DIR% --plot
echo   4. Update production config based on results
echo.
pause
exit /b 0

:error_exit
echo [ERROR] Script failed
exit /b 1
