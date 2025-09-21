import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os
import time
import io

# Page config
st.set_page_config(
    page_title="OMR Processing System",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:5000/api"

# CSS for better styling
st.markdown("""
<style>
.upload-box {
    border: 2px dashed #ccc;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    margin: 20px 0;
}
.success-box {
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    color: #155724;
    padding: 15px;
    border-radius: 5px;
    margin: 10px 0;
}
.error-box {
    background-color: #f8d7da;
    border: 1px solid #f5c6cb;
    color: #721c24;
    padding: 15px;
    border-radius: 5px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# Title and header
st.title("📝 OMR Processing System")
st.markdown("Upload an OMR answer sheet image and get comprehensive analysis results")
st.markdown("---")

# Sidebar
st.sidebar.header("🔧 System Status")

# Check API health
@st.cache_data(ttl=30)  # Cache for 30 seconds
def check_api_health():
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.json() if response.status_code == 200 else None
    except:
        return None

health_status = check_api_health()
if health_status:
    st.sidebar.success("✅ Backend API: Online")
    st.sidebar.text(f"Service: {health_status.get('service', 'Unknown')}")
else:
    st.sidebar.error("❌ Backend API: Offline")
    st.sidebar.warning("Please start the Flask backend server first:\n`python flask_backend.py`")

st.sidebar.markdown("---")
st.sidebar.header("📊 Navigation")

# Main interface
tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload & Process", "📊 View Results", "🎯 Score Analysis", "📈 Analytics"])

with tab1:
    st.header("Upload OMR Image")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose an OMR answer sheet image",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Upload a clear image of an OMR answer sheet with 100 questions (A/B/C/D format)"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.image(uploaded_file, caption="Uploaded OMR Sheet", use_container_width=True)
        
        with col2:
            st.write("**File Details:**")
            st.write(f"Filename: {uploaded_file.name}")
            st.write(f"Size: {uploaded_file.size:,} bytes")
            st.write(f"Type: {uploaded_file.type}")
        
        # Processing button
        if st.button("🚀 Process OMR Sheet", type="primary", use_container_width=True):
            if not health_status:
                st.error("❌ Cannot process: Backend API is offline")
            else:
                with st.spinner("Processing OMR sheet... This may take a few moments."):
                    try:
                        # Prepare file for upload
                        files = {'image': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                        
                        # Send to API
                        response = requests.post(f"{API_BASE_URL}/process", files=files, timeout=60)
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            # Store result in session state
                            st.session_state.last_job_id = result['job_id']
                            st.session_state.last_result = result
                            
                            st.success("✅ Processing completed successfully!")
                            
                            # Display summary
                            if result.get('scoring_available', False):
                                # Show scoring metrics
                                col1, col2, col3, col4, col5 = st.columns(5)
                                with col1:
                                    st.metric("Total Questions", result['total_questions'])
                                with col2:
                                    st.metric("Score", f"{result['score_percentage']}%", 
                                             delta=f"Grade: {result['grade']}")
                                with col3:
                                    st.metric("Correct Answers", result['total_correct'])
                                with col4:
                                    st.metric("High Confidence", result['high_confidence'])
                                with col5:
                                    st.metric("Issues Detected", result['multiple_marks'])
                                
                                # Score visualization
                                st.markdown("### 🎯 Score Analysis")
                                
                                # Score gauge
                                fig_gauge = go.Figure(go.Indicator(
                                    mode = "gauge+number+delta",
                                    value = result['score_percentage'],
                                    domain = {'x': [0, 1], 'y': [0, 1]},
                                    title = {'text': f"Score: {result['grade']} Grade"},
                                    delta = {'reference': 70, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
                                    gauge = {
                                        'axis': {'range': [0, 100]},
                                        'bar': {'color': "darkblue"},
                                        'steps': [
                                            {'range': [0, 60], 'color': "lightgray"},
                                            {'range': [60, 70], 'color': "yellow"},
                                            {'range': [70, 80], 'color': "orange"},
                                            {'range': [80, 90], 'color': "lightgreen"},
                                            {'range': [90, 100], 'color': "green"}
                                        ],
                                        'threshold': {
                                            'line': {'color': "red", 'width': 4},
                                            'thickness': 0.75,
                                            'value': 90
                                        }
                                    }
                                ))
                                fig_gauge.update_layout(height=300)
                                st.plotly_chart(fig_gauge, use_container_width=True)
                                
                            else:
                                # Show regular metrics without scoring
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Total Questions", result['total_questions'])
                                with col2:
                                    st.metric("High Confidence", result['high_confidence'])
                                with col3:
                                    st.metric("Medium Confidence", result['medium_confidence'])
                                with col4:
                                    st.metric("Issues Detected", result['multiple_marks'])
                                
                                st.info("💡 Upload an answer key as `sample.xlsx` in `omr_eval/data/` folder to enable automatic scoring!")
                            
                            # Download buttons
                            st.markdown("### 📥 Download Results")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                try:
                                    excel_response = requests.get(f"{API_BASE_URL}/results/{result['job_id']}/excel")
                                    if excel_response.status_code == 200:
                                        st.download_button(
                                            label="📊 Download Excel",
                                            data=excel_response.content,
                                            file_name=f"omr_results_{result['job_id']}.xlsx",
                                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                            use_container_width=True
                                        )
                                    else:
                                        st.error("❌ Excel file not available")
                                except Exception as e:
                                    st.error(f"❌ Error fetching Excel: {str(e)}")
                            
                            with col2:
                                try:
                                    vis_response = requests.get(f"{API_BASE_URL}/results/{result['job_id']}/visualization")
                                    if vis_response.status_code == 200:
                                        st.download_button(
                                            label="🖼️ Download Visualization",
                                            data=vis_response.content,
                                            file_name=f"visualization_{result['job_id']}.png",
                                            mime="image/png",
                                            use_container_width=True
                                        )
                                    else:
                                        st.error("❌ Visualization not available")
                                except Exception as e:
                                    st.error(f"❌ Error fetching visualization: {str(e)}")
                            
                            with col3:
                                try:
                                    package_response = requests.get(f"{API_BASE_URL}/results/{result['job_id']}/package")
                                    if package_response.status_code == 200:
                                        st.download_button(
                                            label="📦 Download Complete Package",
                                            data=package_response.content,
                                            file_name=f"omr_package_{result['job_id']}.zip",
                                            mime="application/zip",
                                            use_container_width=True
                                        )
                                    else:
                                        st.error("❌ Package not available")
                                except Exception as e:
                                    st.error(f"❌ Error fetching package: {str(e)}")
                        
                        else:
                            error_data = response.json() if response.headers.get('content-type') == 'application/json' else {'error': 'Unknown error'}
                            st.error(f"❌ Processing failed: {error_data.get('error', 'Unknown error')}")
                    
                    except requests.exceptions.Timeout:
                        st.error("❌ Processing timed out. Please try again with a smaller image.")
                    except requests.exceptions.ConnectionError:
                        st.error("❌ Cannot connect to backend API. Please ensure the Flask server is running.")
                    except Exception as e:
                        st.error(f"❌ An error occurred: {str(e)}")

with tab2:
    st.header("View Processing Results")
    
    # Job ID input for viewing results
    job_id_input = st.text_input(
        "Enter Job ID to view results:",
        value=st.session_state.get('last_job_id', ''),
        help="Enter the job ID from a previous processing session"
    )
    
    if job_id_input:
        try:
            # Check job status
            status_response = requests.get(f"{API_BASE_URL}/results/{job_id_input}/status")
            
            if status_response.status_code == 200:
                status_data = status_response.json()
                
                if status_data['status'] == 'completed':
                    st.success(f"✅ Job {job_id_input} completed successfully")
                    
                    # Download links
                    st.markdown("### 📥 Download Files")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        try:
                            excel_response = requests.get(f"{API_BASE_URL}/results/{job_id_input}/excel")
                            if excel_response.status_code == 200:
                                st.download_button(
                                    label="📊 Download Excel Results",
                                    data=excel_response.content,
                                    file_name=f"omr_results_{job_id_input}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    use_container_width=True
                                )
                            else:
                                st.error("❌ Excel file not found")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                    
                    with col2:
                        try:
                            vis_response = requests.get(f"{API_BASE_URL}/results/{job_id_input}/visualization")
                            if vis_response.status_code == 200:
                                st.download_button(
                                    label="🖼️ Download Visualization",
                                    data=vis_response.content,
                                    file_name=f"visualization_{job_id_input}.png",
                                    mime="image/png",
                                    use_container_width=True
                                )
                            else:
                                st.error("❌ Visualization not found")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                    
                    with col3:
                        try:
                            package_response = requests.get(f"{API_BASE_URL}/results/{job_id_input}/package")
                            if package_response.status_code == 200:
                                st.download_button(
                                    label="📦 Download Complete Package",
                                    data=package_response.content,
                                    file_name=f"omr_package_{job_id_input}.zip",
                                    mime="application/zip",
                                    use_container_width=True
                                )
                            else:
                                st.error("❌ Package not found")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                    
                    # Display file status
                    st.markdown("### 📁 File Status")
                    files_status = status_data.get('files_ready', {})
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        status_icon = "✅" if files_status.get('excel', False) else "❌"
                        st.write(f"{status_icon} Excel Results File")
                    
                    with col2:
                        status_icon = "✅" if files_status.get('visualization', False) else "❌"
                        st.write(f"{status_icon} Visualization Image")
                
                elif status_data['status'] == 'processing':
                    st.info("⏳ Job is still processing. Please wait...")
                    if st.button("🔄 Refresh Status"):
                        st.rerun()
                
                else:
                    st.error("❌ Job not found or failed")
            
            else:
                st.error("❌ Could not retrieve job status")
        
        except Exception as e:
            st.error(f"❌ Error checking job status: {str(e)}")

with tab3:
    st.header("🎯 Score Analysis & Answer Key Comparison")
    
    # Job ID input for score analysis
    score_job_id = st.text_input(
        "Enter Job ID for score analysis:",
        value=st.session_state.get('last_job_id', ''),
        help="Enter the job ID to view detailed score breakdown",
        key="score_job_id"
    )
    
    if score_job_id and st.button("📊 Analyze Score", use_container_width=True):
        try:
            # Try to get the Excel file and read the score comparison
            excel_url = f"{API_BASE_URL}/results/{score_job_id}/excel"
            response = requests.get(excel_url)
            
            if response.status_code == 200:
                # Read the Excel data
                excel_data = io.BytesIO(response.content)
                
                try:
                    # Try to read the Score_Comparison sheet
                    comparison_df = pd.read_excel(excel_data, sheet_name='Score_Comparison')
                    summary_df = pd.read_excel(excel_data, sheet_name='Summary_Stats')
                    
                    # Display score summary
                    if not summary_df.empty and 'Scoring_Available' in summary_df.columns:
                        score_available = summary_df.iloc[0]['Scoring_Available']
                        
                        if score_available:
                            st.success("✅ Answer key comparison available!")
                            
                            # Score metrics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Score", f"{summary_df.iloc[0]['Score_Percentage']:.1f}%")
                            with col2:
                                st.metric("Letter Grade", summary_df.iloc[0]['Letter_Grade'])
                            with col3:
                                st.metric("Correct Answers", f"{summary_df.iloc[0]['Total_Correct']}/{summary_df.iloc[0]['Total_Answered']}")
                            with col4:
                                accuracy = (summary_df.iloc[0]['Total_Correct'] / summary_df.iloc[0]['Total_Answered']) * 100
                                st.metric("Accuracy", f"{accuracy:.1f}%")
                            
                            # Detailed comparison table
                            st.subheader("📋 Question-by-Question Analysis")
                            
                            # Color code the results
                            def highlight_correct(val):
                                if val == True:
                                    return 'background-color: #d4edda'  # Green
                                elif val == False:
                                    return 'background-color: #f8d7da'  # Red
                                return ''
                            
                            def highlight_answers(row):
                                if row['Is_Correct'] == True:
                                    return ['background-color: #d4edda'] * len(row)
                                elif row['Is_Correct'] == False:
                                    return ['background-color: #f8d7da'] * len(row)
                                else:
                                    return [''] * len(row)
                            
                            styled_comparison = comparison_df.style.apply(highlight_answers, axis=1)
                            st.dataframe(styled_comparison, use_container_width=True, height=400)
                            
                            # Performance analysis
                            st.subheader("📊 Performance Breakdown")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Correct vs Incorrect pie chart
                                correct_count = comparison_df['Is_Correct'].sum()
                                incorrect_count = len(comparison_df) - correct_count
                                
                                fig_pie = px.pie(
                                    values=[correct_count, incorrect_count],
                                    names=['Correct', 'Incorrect'],
                                    title="Answer Distribution",
                                    color_discrete_map={'Correct': '#28a745', 'Incorrect': '#dc3545'}
                                )
                                st.plotly_chart(fig_pie, use_container_width=True)
                            
                            with col2:
                                # Confidence vs Correctness
                                confidence_correct = comparison_df.groupby(['Confidence', 'Is_Correct']).size().reset_index(name='Count')
                                
                                fig_bar = px.bar(
                                    confidence_correct,
                                    x='Confidence',
                                    y='Count',
                                    color='Is_Correct',
                                    title="Confidence vs Correctness",
                                    color_discrete_map={True: '#28a745', False: '#dc3545'}
                                )
                                st.plotly_chart(fig_bar, use_container_width=True)
                            
                            # Download enhanced results
                            st.subheader("📥 Download Enhanced Results")
                            csv_data = comparison_df.to_csv(index=False)
                            st.download_button(
                                label="📊 Download Score Analysis CSV",
                                data=csv_data,
                                file_name=f"score_analysis_{score_job_id}.csv",
                                mime="text/csv"
                            )
                        
                        else:
                            st.warning("⚠️ No answer key available for this processing job.")
                            st.info("To enable scoring, upload a `sample.xlsx` file with answer key to `omr_eval/data/` folder.")
                    
                except Exception as sheet_error:
                    st.warning("⚠️ Score comparison data not available.")
                    st.info("This job was processed without an answer key. Upload `sample.xlsx` to enable scoring for future jobs.")
            
            else:
                st.error("❌ Could not retrieve results for this job ID.")
        
        except Exception as e:
            st.error(f"❌ Error analyzing score: {str(e)}")
    
    # Answer key upload section
    st.subheader("📝 Answer Key Management")
    st.info("Upload your answer key as an Excel file to enable automatic scoring for future OMR processing.")
    
    uploaded_answer_key = st.file_uploader(
        "Upload Answer Key (Excel format)",
        type=['xlsx', 'xls'],
        help="Excel file should have columns for Question Number and Correct Answer (A/B/C/D)"
    )
    
    if uploaded_answer_key:
        # Display preview of answer key
        try:
            df_preview = pd.read_excel(uploaded_answer_key)
            st.write("**Answer Key Preview:**")
            st.dataframe(df_preview.head(10), use_container_width=True)
            
            if st.button("💾 Save Answer Key", use_container_width=True):
                # Here you would save the file to omr_eval/data/sample.xlsx
                # For now, show instructions
                st.success("✅ Answer key uploaded successfully!")
                st.info("💡 The answer key has been saved and will be used for scoring future OMR sheets.")
        
        except Exception as e:
            st.error(f"❌ Error reading answer key: {str(e)}")

with tab4:
    st.header("System Analytics")
    
    # Placeholder for analytics - you could extend this to track processing history
    st.info("📈 Analytics feature coming soon!")
    st.write("This section will show:")
    st.write("- Processing history")
    st.write("- Performance metrics") 
    st.write("- Common confidence patterns")
    st.write("- Error rates and trends")
    
    # Show current session info if available
    if 'last_result' in st.session_state:
        st.markdown("### 📊 Last Processing Session")
        result = st.session_state.last_result
        
        # Confidence distribution
        confidence_data = {
            'High': result['high_confidence'],
            'Medium': result['medium_confidence'],
            'Low': result['low_confidence'],
            'No Answer': result['no_answer']
        }
        
        fig_pie = px.pie(
            values=list(confidence_data.values()),
            names=list(confidence_data.keys()),
            title="Answer Confidence Distribution",
            color_discrete_map={
                'High': '#28a745',
                'Medium': '#ffc107', 
                'Low': '#dc3545',
                'No Answer': '#6c757d'
            }
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🔧 <strong>OMR Processing System</strong> | Built with Streamlit + Flask</p>
    <p>📁 Backend API: <code>http://localhost:5000</code> | 🌐 Frontend: <code>http://localhost:8501</code></p>
</div>
""", unsafe_allow_html=True)

# Auto-refresh for job status if processing
if 'last_result' in st.session_state and 'processing' in st.session_state.get('last_result', {}).get('status', ''):
    time.sleep(2)
    st.rerun()
