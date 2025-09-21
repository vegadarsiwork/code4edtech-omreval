import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os

# Page config
st.set_page_config(
    page_title="OMR Results Viewer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and header
st.title("📝 OMR Answer Sheet Results")
st.markdown("---")

# File path to Excel results
excel_path = Path("results/omr_answers.xlsx")

if not excel_path.exists():
    st.error(f"❌ Excel file not found at: {excel_path.absolute()}")
    st.info("Please run the OMR processing first by executing: `python omr_eval/app.py`")
    st.stop()

# Load data
@st.cache_data
def load_data():
    try:
        # Load all sheets
        sheets = pd.read_excel(excel_path, sheet_name=None)
        return sheets
    except Exception as e:
        st.error(f"Error loading Excel file: {e}")
        return None

data_sheets = load_data()

if data_sheets is None:
    st.stop()

# Sidebar for navigation
st.sidebar.header("📊 Navigation")
view_mode = st.sidebar.selectbox(
    "Select View:",
    ["📋 Summary Dashboard", "📝 Detailed Results", "🔍 Question Analysis", "📈 Statistics"]
)

# Load main data
df_results = data_sheets['OMR_Results']
df_summary = data_sheets['Summary_Stats']
df_clean = data_sheets['Clean_Answers']

# Summary Dashboard
if view_mode == "📋 Summary Dashboard":
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Questions", df_summary.iloc[0]['Total_Questions'])
    with col2:
        st.metric("High Confidence", df_summary.iloc[0]['High_Confidence_Answers'], 
                 delta=f"{df_summary.iloc[0]['High_Confidence_Answers']/df_summary.iloc[0]['Total_Questions']*100:.1f}%")
    with col3:
        st.metric("Medium Confidence", df_summary.iloc[0]['Medium_Confidence_Answers'])
    with col4:
        st.metric("Low/No Answer", df_summary.iloc[0]['Low_Confidence_Answers'] + df_summary.iloc[0]['No_Answer_Detected'])
    
    # Confidence distribution chart
    st.subheader("📈 Confidence Distribution")
    confidence_data = {
        'High': df_summary.iloc[0]['High_Confidence_Answers'],
        'Medium': df_summary.iloc[0]['Medium_Confidence_Answers'], 
        'Low': df_summary.iloc[0]['Low_Confidence_Answers'],
        'No Answer': df_summary.iloc[0]['No_Answer_Detected']
    }
    
    fig_pie = px.pie(
        values=list(confidence_data.values()),
        names=list(confidence_data.keys()),
        title="Answer Confidence Levels",
        color_discrete_map={
            'High': '#00CC88',
            'Medium': '#FFB84D', 
            'Low': '#FF6B6B',
            'No Answer': '#666666'
        }
    )
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # Answer distribution
    st.subheader("📊 Answer Distribution")
    answer_counts = df_clean['Answer'].value_counts()
    if '' in answer_counts:
        answer_counts = answer_counts.drop('')  # Remove empty answers
    
    fig_bar = px.bar(
        x=answer_counts.index,
        y=answer_counts.values,
        title="Distribution of Selected Answers (A, B, C, D)",
        labels={'x': 'Answer Option', 'y': 'Count'},
        color=answer_counts.values,
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# Detailed Results
elif view_mode == "📝 Detailed Results":
    st.subheader("📝 Complete OMR Results")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        confidence_filter = st.multiselect(
            "Filter by Confidence:",
            ['High', 'Medium', 'Low', 'None'],
            default=['High', 'Medium', 'Low', 'None']
        )
    with col2:
        status_filter = st.multiselect(
            "Filter by Status:",
            df_results['Status'].unique(),
            default=df_results['Status'].unique()
        )
    with col3:
        show_multiple = st.checkbox("Show only multiple marks", False)
    
    # Apply filters
    filtered_df = df_results[
        (df_results['Confidence_Level'].isin(confidence_filter)) &
        (df_results['Status'].isin(status_filter))
    ]
    
    if show_multiple:
        filtered_df = filtered_df[filtered_df['Multiple_Marks_Detected'] == 'Yes']
    
    # Color code the dataframe
    def color_confidence(val):
        if val == 'High':
            return 'background-color: #d4edda'
        elif val == 'Medium':
            return 'background-color: #fff3cd'
        elif val == 'Low':
            return 'background-color: #f8d7da'
        else:
            return 'background-color: #f6f6f6'
    
    def color_answer(val):
        if val == 'NO_ANSWER':
            return 'background-color: #f8d7da; color: #721c24'
        return ''
    
    # Display table
    styled_df = filtered_df.style.applymap(color_confidence, subset=['Confidence_Level']) \
                                 .applymap(color_answer, subset=['Final_Answer'])
    
    st.dataframe(styled_df, use_container_width=True, height=600)
    
    # Download button
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Results as CSV",
        data=csv,
        file_name="omr_filtered_results.csv",
        mime="text/csv"
    )

# Question Analysis
elif view_mode == "🔍 Question Analysis":
    st.subheader("🔍 Individual Question Analysis")
    
    question_num = st.selectbox(
        "Select Question Number:",
        range(1, 101),
        index=0
    )
    
    # Get question data
    question_data = df_results[df_results['Question_Number'] == question_num].iloc[0]
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Question Details")
        st.write(f"**Question:** {question_data['Question_Number']}")
        st.write(f"**Final Answer:** {question_data['Final_Answer']}")
        st.write(f"**Confidence:** {question_data['Confidence_Level']}")
        st.write(f"**Status:** {question_data['Status']}")
        st.write(f"**Multiple Marks:** {question_data['Multiple_Marks_Detected']}")
    
    with col2:
        st.markdown("### Fill Percentages")
        
        # Extract fill percentages
        fill_data = {
            'A': float(question_data['A_Fill_Percent'].rstrip('%')),
            'B': float(question_data['B_Fill_Percent'].rstrip('%')),
            'C': float(question_data['C_Fill_Percent'].rstrip('%')),
            'D': float(question_data['D_Fill_Percent'].rstrip('%'))
        }
        
        # Create bar chart
        fig_question = go.Figure(data=[
            go.Bar(
                x=list(fill_data.keys()),
                y=list(fill_data.values()),
                marker_color=['#00CC88' if opt == question_data['Final_Answer'] else '#FFB84D' for opt in fill_data.keys()],
                text=[f"{v:.1f}%" for v in fill_data.values()],
                textposition='auto'
            )
        ])
        
        fig_question.update_layout(
            title=f"Fill Percentages for Question {question_num}",
            xaxis_title="Answer Options",
            yaxis_title="Fill Percentage (%)",
            yaxis=dict(range=[0, 100])
        )
        
        # Add threshold line
        fig_question.add_hline(y=50, line_dash="dash", line_color="red", 
                              annotation_text="50% Threshold")
        
        st.plotly_chart(fig_question, use_container_width=True)
    
    # Raw detection data
    st.markdown("### Raw Detection Data")
    st.code(question_data['Raw_Detection_Data'])

# Statistics
elif view_mode == "📈 Statistics":
    st.subheader("📈 Detailed Statistics")
    
    # Overall statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Confidence Breakdown")
        confidence_counts = df_results['Confidence_Level'].value_counts()
        fig_conf = px.pie(
            values=confidence_counts.values,
            names=confidence_counts.index,
            title="Confidence Distribution"
        )
        st.plotly_chart(fig_conf, use_container_width=True)
    
    with col2:
        st.markdown("### Status Breakdown")
        status_counts = df_results['Status'].value_counts()
        fig_status = px.bar(
            x=status_counts.values,
            y=status_counts.index,
            orientation='h',
            title="Detection Status Distribution"
        )
        st.plotly_chart(fig_status, use_container_width=True)
    
    # Question-wise confidence heatmap
    st.markdown("### Question Confidence Heatmap")
    
    # Create confidence score (High=3, Medium=2, Low=1, None=0)
    confidence_map = {'High': 3, 'Medium': 2, 'Low': 1, 'None': 0}
    df_results['Confidence_Score'] = df_results['Confidence_Level'].map(confidence_map)
    
    # Reshape for heatmap (10x10 grid)
    confidence_matrix = df_results['Confidence_Score'].values.reshape(10, 10)
    
    fig_heatmap = px.imshow(
        confidence_matrix,
        title="Question Confidence Heatmap (10x10 grid)",
        labels=dict(x="Question Column", y="Question Row", color="Confidence"),
        color_continuous_scale="RdYlGn",
        aspect="auto"
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Summary table
    st.markdown("### Summary Statistics")
    summary_table = pd.DataFrame([{
        'Metric': 'Total Questions',
        'Value': len(df_results)
    }, {
        'Metric': 'Average Confidence Score',
        'Value': f"{df_results['Confidence_Score'].mean():.2f}"
    }, {
        'Metric': 'Questions with Multiple Marks',
        'Value': len(df_results[df_results['Multiple_Marks_Detected'] == 'Yes'])
    }, {
        'Metric': 'Questions with No Answer',
        'Value': len(df_results[df_results['Final_Answer'] == 'NO_ANSWER'])
    }])
    
    st.table(summary_table)

# Footer
st.markdown("---")
st.markdown("🔧 **Generated by OMR Processing System** | 📁 Data source: `results/omr_answers.xlsx`")
