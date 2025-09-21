#!/usr/bin/env python3
"""
OMR Processing System Setup Script
Automated setup and verification script for the OMR Processing System
"""

import os
import sys
import subprocess
import platform
import urllib.request
import time
from pathlib import Path

def print_banner():
    print("=" * 60)
    print("🎯 OMR Processing System Setup")
    print("=" * 60)
    print()

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def create_virtual_environment():
    """Create and activate virtual environment"""
    print("\n📦 Setting up virtual environment...")
    
    if os.path.exists("venv"):
        print("✅ Virtual environment already exists")
        return True
    
    try:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("✅ Virtual environment created successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to create virtual environment")
        return False

def get_activation_command():
    """Get the appropriate activation command for the OS"""
    if platform.system() == "Windows":
        return os.path.join("venv", "Scripts", "activate.bat")
    else:
        return "source venv/bin/activate"

def install_requirements():
    """Install required packages"""
    print("\n📋 Installing dependencies...")
    
    # Get the appropriate pip executable
    if platform.system() == "Windows":
        pip_executable = os.path.join("venv", "Scripts", "pip.exe")
    else:
        pip_executable = os.path.join("venv", "bin", "pip")
    
    try:
        # Upgrade pip first
        subprocess.run([pip_executable, "install", "--upgrade", "pip"], check=True)
        
        # Install requirements
        subprocess.run([pip_executable, "install", "-r", "requirements.txt"], check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = [
        "temp_uploads",
        "temp_results", 
        "results",
        "omr_eval/data/uploads"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True

def download_sample_images():
    """Download sample OMR images if not present"""
    print("\n🖼️  Checking sample images...")
    
    uploads_dir = Path("omr_eval/data/uploads")
    
    # Check if there are any image files
    image_files = list(uploads_dir.glob("*.jpg")) + list(uploads_dir.glob("*.png")) + list(uploads_dir.glob("*.jpeg"))
    
    if image_files:
        print(f"✅ Found {len(image_files)} sample images")
        return True
    else:
        print("ℹ️  No sample images found. You can add OMR images to omr_eval/data/uploads/ for testing")
        return True

def create_sample_answer_key():
    """Create a sample answer key if it doesn't exist"""
    print("\n🔑 Checking answer key...")
    
    answer_key_path = Path("omr_eval/data/sample.xlsx")
    
    if answer_key_path.exists():
        print("✅ Answer key already exists")
        return True
    
    try:
        # Get the appropriate python executable
        if platform.system() == "Windows":
            python_executable = os.path.join("venv", "Scripts", "python.exe")
        else:
            python_executable = os.path.join("venv", "bin", "python")
        
        # Create answer key using pandas
        create_key_script = '''
import pandas as pd
import os

# Create answer key data
questions = list(range(1, 101))
answers = [["A", "B", "C", "D"][i % 4] for i in range(100)]

df = pd.DataFrame({
    "Question": questions,
    "Answer": answers
})

# Ensure directory exists
os.makedirs("omr_eval/data", exist_ok=True)

# Save to Excel
df.to_excel("omr_eval/data/sample.xlsx", index=False)
print("Sample answer key created successfully")
'''
        
        subprocess.run([python_executable, "-c", create_key_script], check=True)
        print("✅ Sample answer key created")
        return True
        
    except subprocess.CalledProcessError:
        print("⚠️  Could not create sample answer key automatically")
        print("   You can create it manually later")
        return True

def test_imports():
    """Test if all required modules can be imported"""
    print("\n🧪 Testing imports...")
    
    # Get the appropriate python executable
    if platform.system() == "Windows":
        python_executable = os.path.join("venv", "Scripts", "python.exe")
    else:
        python_executable = os.path.join("venv", "bin", "python")
    
    test_script = '''
try:
    import cv2
    import numpy
    import pandas
    import flask
    import streamlit
    import plotly
    import requests
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)
'''
    
    try:
        subprocess.run([python_executable, "-c", test_script], check=True)
        return True
    except subprocess.CalledProcessError:
        print("❌ Import test failed")
        return False

def test_flask_startup():
    """Test if Flask backend can start"""
    print("\n🖥️  Testing Flask backend startup...")
    
    # Get the appropriate python executable
    if platform.system() == "Windows":
        python_executable = os.path.join("venv", "Scripts", "python.exe")
    else:
        python_executable = os.path.join("venv", "bin", "python")
    
    try:
        # Test import of flask_backend
        test_script = '''
try:
    from flask_backend import app
    print("✅ Flask backend imports successfully")
except ImportError as e:
    print(f"❌ Flask backend import error: {e}")
    exit(1)
'''
        subprocess.run([python_executable, "-c", test_script], check=True)
        return True
    except subprocess.CalledProcessError:
        print("❌ Flask backend test failed")
        return False

def print_usage_instructions():
    """Print usage instructions"""
    print("\n" + "=" * 60)
    print("🎉 Setup completed successfully!")
    print("=" * 60)
    print()
    print("📋 How to start the system:")
    print()
    
    if platform.system() == "Windows":
        print("   Windows:")
        print("   > start_system.bat")
        print()
        print("   Or manually:")
        print("   > venv\\Scripts\\activate")
        print("   > python flask_backend.py")
        print("   > (in another terminal) streamlit run streamlit_frontend.py --server.port 8502")
    else:
        print("   Linux/macOS:")
        print("   $ chmod +x start_system.sh")
        print("   $ ./start_system.sh")
        print()
        print("   Or manually:")
        print("   $ source venv/bin/activate")
        print("   $ python flask_backend.py")
        print("   $ (in another terminal) streamlit run streamlit_frontend.py --server.port 8502")
    
    print()
    print("🌐 Access URLs:")
    print("   Frontend: http://localhost:8502")
    print("   Backend API: http://localhost:5000")
    print()
    print("📖 For more information, see README.md")
    print()

def main():
    """Main setup function"""
    print_banner()
    
    # Check requirements
    if not check_python_version():
        return False
    
    # Setup steps
    steps = [
        ("Create virtual environment", create_virtual_environment),
        ("Install dependencies", install_requirements),
        ("Create directories", create_directories),
        ("Check sample images", download_sample_images),
        ("Create sample answer key", create_sample_answer_key),
        ("Test imports", test_imports),
        ("Test Flask startup", test_flask_startup),
    ]
    
    for step_name, step_function in steps:
        if not step_function():
            print(f"\n❌ Setup failed at step: {step_name}")
            return False
    
    print_usage_instructions()
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
