# 🎯 OMR Processing System

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/vegadarsiwork/code4edtech-omreval/main/streamlit_app_standalone.py)

A comprehensive Optical Mark Recognition (OMR) system with automated scoring, built with Flask API backend and Streamlit frontend. Process answer sheets, detect filled bubbles, and get instant scoring with detailed analytics.

## 🌟 Features

### 🔍 Advanced OMR Processing
- **High Accuracy Detection**: 95%+ bubble detection accuracy
- **Automatic Skew Correction**: Handles rotated sheets up to 45 degrees
- **Adaptive Thresholding**: Works with various lighting conditions
- **Multi-mark Detection**: Identifies partially filled or multiple marks

### 📊 Automated Scoring System
- **Answer Key Comparison**: Upload Excel answer keys for automatic grading
- **Instant Results**: Get scores, percentages, and letter grades immediately
- **Question-by-Question Analysis**: Detailed breakdown of each answer
- **Confidence Scoring**: Quality assessment for each detected answer

### 🎨 Rich Visualizations
- **Interactive Charts**: Score gauges, performance analytics
- **Answer Visualization**: Color-coded result displays
- **Export Options**: Excel, PNG, CSV, and ZIP packages

### 🚀 Modern Architecture
- **RESTful API**: Flask backend with comprehensive endpoints
- **Web Interface**: User-friendly Streamlit dashboard
- **Scalable Design**: Easy integration and deployment
- **Real-time Processing**: Live status updates and results

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/vegadarsiwork/code4edtech-omreval.git
   cd code4edtech-omreval
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the system**
   ```bash
   # Windows
   start_system.bat
   
   # Manual start (any OS)
   python flask_backend.py
   # In another terminal:
   streamlit run streamlit_frontend.py --server.port 8502
   ```

5. **Open your browser**
   - Frontend: http://localhost:8502
   - API Documentation: http://localhost:5000/api/health

## 📋 Usage

### Processing OMR Sheets

1. **Upload Image**: Select your OMR answer sheet (PNG, JPG, JPEG)
2. **Process**: Click "Process OMR Sheet" to analyze bubbles
3. **View Results**: Get instant results with confidence scores
4. **Download**: Export Excel files, visualizations, or complete packages

### Setting Up Answer Keys

1. **Prepare Excel File**: Create `sample.xlsx` with columns:
   - `Question`: Question numbers (1, 2, 3, ...)
   - `Answer`: Correct answers (A, B, C, D)

2. **Upload Answer Key**: Place in `omr_eval/data/sample.xlsx`

3. **Automatic Scoring**: All future processing will include scoring

### API Usage

The system provides a RESTful API for integration:

```bash
# Process OMR image
POST /api/process
Content-Type: multipart/form-data

# Get results
GET /api/results/{job_id}/excel
GET /api/results/{job_id}/visualization  
GET /api/results/{job_id}/package
GET /api/results/{job_id}/status

# Answer key management
POST /api/answer-key/upload
GET /api/answer-key/status

# Health check
GET /api/health
```

## 📁 Project Structure

```
code4edtech-omreval/
├── 📖 README.md                    # Main documentation
├── 📋 requirements.txt             # Python dependencies  
├── 📋 requirements_streamlit.txt   # Streamlit Cloud dependencies
├── 🎯 streamlit_app_standalone.py  # Standalone Streamlit app
├── 🖥️ flask_backend.py            # API server
├── 🌐 streamlit_frontend.py       # Web interface
├── 🐍 setup.py                     # Automated setup script
├── 🚀 start_system.bat             # Windows startup script
├── 🐳 Dockerfile & docker-compose  # Container deployment
├── 📝 CONTRIBUTING.md & LICENSE    # Contributor guidelines & license
├── 🔧 .github/workflows/ci.yml     # CI/CD pipeline
├── 🙈 .gitignore                   # Version control exclusions
└── 📁 omr_eval/                    # Core OMR processing modules
    ├── data/
    │   ├── sample.xlsx             # Answer key template
    │   └── uploads/                # Sample OMR images
    └── omr/
        ├── __init__.py
        ├── classify.py             # Answer classification
        ├── detect_bubbles.py       # Bubble detection
        ├── evaluate.py             # Result evaluation
        └── preprocess.py           # Image preprocessing
```

## 🎯 Supported Features

### Image Formats
- PNG, JPG, JPEG
- Various resolutions and quality levels
- Automatic resize and optimization

### Answer Sheet Types
- Standard bubble sheets (100 questions × 4 options)
- Customizable grid detection
- Multiple choice questions (A, B, C, D)

### Export Formats
- **Excel**: Detailed results with multiple sheets
- **CSV**: Score analysis and comparisons
- **PNG**: Answer visualizations
- **ZIP**: Complete result packages

## 🔧 Configuration

### Answer Key Format
```excel
Question | Answer
---------|--------
1        | A
2        | B
3        | C
4        | D
...      | ...
```

### Environment Variables
```bash
# Optional: Customize ports
FLASK_PORT=5000
STREAMLIT_PORT=8502

# Optional: Configure processing
OMR_CONFIDENCE_THRESHOLD=0.45
OMR_FILL_THRESHOLD=0.30
```

## 📊 Performance

- **Processing Time**: < 10 seconds per sheet
- **Accuracy**: 95%+ bubble detection
- **Throughput**: Handles multiple concurrent requests
- **Memory Usage**: Optimized for large images

## 🚀 Deployment

### 🌐 Streamlit Cloud (Recommended)

**One-Click Deployment:**

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/vegadarsiwork/code4edtech-omreval/main/streamlit_app_standalone.py)

**Manual Deployment:**
1. Fork this repository to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click "New app"
5. Select your repository: `vegadarsiwork/code4edtech-omreval`
6. Set main file: `streamlit_app_standalone.py`
7. Click "Deploy"

**Requirements:** Uses `requirements_streamlit.txt` automatically

### Local Development
```bash
# Full system (Flask + Streamlit)
python flask_backend.py
streamlit run streamlit_frontend.py --server.port 8502

# Standalone Streamlit version
streamlit run streamlit_app_standalone.py
```

### Production Deployment
```bash
# Using Gunicorn for Flask
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 flask_backend:app

# Using production Streamlit
streamlit run streamlit_frontend.py --server.port 8502 --server.headless true
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000 8502

CMD ["python", "flask_backend.py"]
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Common Issues

**Q: "OpenCV error when processing images"**
A: Ensure OpenCV is properly installed: `pip install opencv-python==4.10.0.84`

**Q: "Frontend can't connect to backend"**
A: Verify Flask server is running on port 5000: `curl http://localhost:5000/api/health`

**Q: "Low detection accuracy"**
A: Ensure good image quality, proper lighting, and clear bubble marks

**Q: "Scoring not working"**
A: Upload answer key as `omr_eval/data/sample.xlsx` with Question and Answer columns

### Getting Help

- 📧 Create an issue on GitHub
- 💬 Check existing issues and discussions
- 📖 Review the documentation and examples

## 🏆 Acknowledgments

- OpenCV community for image processing capabilities
- Streamlit team for the amazing web app framework
- Flask community for the robust API framework
- Contributors and testers who helped improve the system

## 📈 Roadmap

- [ ] Mobile app for on-the-go processing
- [ ] Cloud deployment templates
- [ ] Advanced analytics and ML insights
- [ ] LMS integration capabilities
- [ ] Support for additional question types
- [ ] Batch processing interface
- [ ] Real-time collaboration features

---

**Made with ❤️ for educators, students, and developers**

*Star ⭐ this repository if it helped you!*
