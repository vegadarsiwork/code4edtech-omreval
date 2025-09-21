#!/bin/bash

# OMR Processing System - Startup Script (Linux/macOS)

echo "🚀 Starting OMR Processing System..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "📋 Installing dependencies..."
pip install -r requirements.txt

# Start Flask backend in background
echo "🖥️  Starting Flask backend..."
python flask_backend.py &
FLASK_PID=$!

# Wait a moment for Flask to start
sleep 3

# Start Streamlit frontend
echo "🌐 Starting Streamlit frontend..."
echo "📱 Frontend will be available at: http://localhost:8502"
echo "🔗 Backend API available at: http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

# Start Streamlit
streamlit run streamlit_frontend.py --server.port 8502

# Cleanup: Kill Flask process when Streamlit exits
echo "🛑 Shutting down..."
kill $FLASK_PID 2>/dev/null
echo "✅ System stopped"
