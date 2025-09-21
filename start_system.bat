@echo off
echo 🚀 Starting OMR Processing System...
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

echo ✅ Virtual environment activated
echo.

REM Start Flask backend in a new window
echo 🔧 Starting Flask Backend API Server...
start "OMR Backend API" cmd /k "python flask_backend.py"

REM Wait a moment for backend to start
timeout /t 3 /nobreak > nul

REM Start Streamlit frontend
echo 🌐 Starting Streamlit Frontend...
python -m streamlit run streamlit_frontend.py

pause
