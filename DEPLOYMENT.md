# 🚀 Streamlit Cloud Deployment Guide

## Quick Deploy to Streamlit Cloud

### Option 1: One-Click Deploy
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/vegadarsiwork/code4edtech-omreval/main/streamlit_app_standalone.py)

### Option 2: Manual Deploy

1. **Go to Streamlit Cloud**
   - Visit: https://share.streamlit.io
   - Sign in with your GitHub account

2. **Deploy New App**
   - Click "New app"
   - Repository: `vegadarsiwork/code4edtech-omreval`
   - Branch: `main`
   - Main file path: `streamlit_app_standalone.py`
   - Click "Deploy!"

3. **Automatic Configuration**
   - Streamlit Cloud will automatically use `requirements_streamlit.txt`
   - All dependencies will be installed automatically
   - The app will be available at a custom URL

## 🎯 What You Get

- **Live Demo**: https://your-app-name.streamlit.app
- **Free Hosting**: No cost for public repositories
- **Auto-Updates**: Deploys automatically when you push to GitHub
- **HTTPS**: Secure connection by default
- **Easy Sharing**: Send the link to anyone

## 🔧 Features Available in Cloud Version

✅ **Upload & Process OMR Sheets**
✅ **Automatic Bubble Detection** 
✅ **Answer Key Comparison**
✅ **Instant Scoring & Grading**
✅ **Interactive Visualizations**
✅ **Download Results** (Excel, PNG, ZIP)
✅ **Score Analytics Dashboard**

## 📱 Usage Instructions

1. **Open the Streamlit Cloud link**
2. **Upload an OMR image** (PNG, JPG, JPEG)
3. **Click "Process OMR Sheet"**
4. **View instant results** with scores and analytics
5. **Download reports** in various formats

## 🆘 Troubleshooting

**If the app doesn't load:**
- Check if the repository is public
- Ensure `streamlit_app_standalone.py` is in the root directory
- Verify `requirements_streamlit.txt` exists

**If processing fails:**
- Make sure images are clear and well-lit
- Try with different image formats (PNG recommended)
- Check that bubbles are clearly marked

## 🔄 Local Development

For local development with full Flask backend:

```bash
git clone https://github.com/vegadarsiwork/code4edtech-omreval.git
cd code4edtech-omreval
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
python flask_backend.py
# In another terminal:
streamlit run streamlit_frontend.py --server.port 8502
```

## 📞 Support

- GitHub Issues: https://github.com/vegadarsiwork/code4edtech-omreval/issues
- Repository: https://github.com/vegadarsiwork/code4edtech-omreval
- Documentation: See README.md for full details
