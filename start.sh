#!/bin/bash

# Multi-Object Tracking System - Quick Start Script
# This script sets up the environment and starts the application

echo "=========================================="
echo "Multi-Object Tracking System - Quick Start"
echo "=========================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

echo "✓ Python found: $(python3 --version)"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi
echo "✓ Virtual environment activated"

# Install requirements
echo ""
echo "Installing dependencies..."
echo "This may take a few minutes..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed successfully"
else
    echo "✗ Error installing dependencies"
    exit 1
fi

# Download YOLOv8 model
echo ""
echo "Checking for YOLOv8 models..."
if [ ! -d "models" ] || [ -z "$(ls -A models/*.pt 2>/dev/null)" ]; then
    echo "⚠️  No models found in 'models' directory"
    echo ""
    read -p "Would you like to download models now? (y/n): " download_choice
    if [[ $download_choice == "y" || $download_choice == "Y" ]]; then
        python3 download_models.py
    else
        echo ""
        echo "⚠️  Warning: You need to add models to 'models' directory before using the app"
        echo "   Run 'python download_models.py' later to download models"
    fi
else
    echo "✓ Models found in 'models' directory:"
    ls -lh models/*.pt | awk '{print "  -", $9, "(" $5 ")"}'
fi

# Start Streamlit app
echo ""
echo "=========================================="
echo "Starting Multi-Object Tracking System..."
echo "=========================================="
echo ""
echo "The application will open in your browser."
echo "If it doesn't, navigate to: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the application"
echo ""

streamlit run app.py