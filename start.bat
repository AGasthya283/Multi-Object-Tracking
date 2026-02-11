@echo off
REM Multi-Object Tracking System - Quick Start Script (Windows)

echo ==========================================
echo Multi-Object Tracking System - Quick Start
echo ==========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed
    echo Please install Python 3.8 or higher from python.org
    pause
    exit /b 1
)

python --version
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
    echo Virtual environment created
) else (
    echo Virtual environment already exists
)

REM Activate virtual environment
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo Virtual environment activated

REM Install requirements
echo.
echo Installing dependencies...
echo This may take a few minutes...
python -m pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt

if errorlevel 1 (
    echo Error installing dependencies
    pause
    exit /b 1
)
echo Dependencies installed successfully

REM Check for YOLOv8 models
echo.
echo Checking for YOLOv8 models...
if not exist "models\" (
    mkdir models
    echo Created models directory
)

dir /b models\*.pt >nul 2>&1
if errorlevel 1 (
    echo Warning: No models found in 'models' directory
    echo.
    set /p download_choice="Would you like to download models now? (y/n): "
    if /i "%download_choice%"=="y" (
        python download_models.py
    ) else (
        echo.
        echo Warning: You need to add models to 'models' directory before using the app
        echo    Run 'python download_models.py' later to download models
    )
) else (
    echo Models found in 'models' directory:
    dir /b models\*.pt
)

REM Start Streamlit app
echo.
echo ==========================================
echo Starting Multi-Object Tracking System...
echo ==========================================
echo.
echo The application will open in your browser.
echo If it doesn't, navigate to: http://localhost:8501
echo.
echo Press Ctrl+C to stop the application
echo.

streamlit run app.py