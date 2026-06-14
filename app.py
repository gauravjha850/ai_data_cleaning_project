import streamlit as st
import pandas as pd
import requests
import os
import plotly.graph_objects as go
import json
from dotenv import load_dotenv

load_dotenv()

API_URL = "http://localhost:8000"

st.set_page_config(page_title="AI Data Cleaning Agent", layout="wide")

st.title("AI Data Cleaning Agent")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("System Status")
    
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code == 200:
            st.success("Backend Connected")
        else:
            st.error("Backend Error")
    except:
        st.error("Backend Not Running")
        st.info("Run: python backend.py")
    
    st.markdown("---")
    st.header("Settings")
    max_rows = st.slider("Max rows to process", 100, 50000, 1000, 1000)

# File upload
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx'])

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    st.success(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    st.subheader("Data Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", len(df))
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    with col4:
        st.metric("Duplicate Rows", df.duplicated().sum())
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["EDA & Visualizations", "Feature Engineering", "Time Series", "Clean Data"])
    
    # ============ TAB 1: EDA & VISUALIZATIONS ============
    with tab1:
        st.header("Exploratory Data Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Generate EDA Report", use_container_width=True):
                with st.spinner("Generating EDA report..."):
                    uploaded_file.seek(0)
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    response = requests.post(f"{API_URL}/eda/report", files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.json(result["eda_report"])
                    else:
                        st.error(f"Error: {response.text}")
        
        with col2:
            if st.button("Generate Visualizations", use_container_width=True):
                with st.spinner("Creating visualizations..."):
                    uploaded_file.seek(0)
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    response = requests.post(f"{API_URL}/eda/visualizations", files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        vis_data = result["visualizations"]
                        
                        if vis_data.get("distributions"):
                            for name, fig_json in vis_data["distributions"].items():
                                if "histogram" in name:
                                    fig = go.Figure(json.loads(fig_json))
                                    st.plotly_chart(fig, use_container_width=True)
                        
                        if vis_data.get("correlation_heatmap"):
                            fig = go.Figure(json.loads(vis_data["correlation_heatmap"]))
                            st.plotly_chart(fig, use_container_width=True)
                        
                        if vis_data.get("boxplots"):
                            fig = go.Figure(json.loads(vis_data["boxplots"]))
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(f"Error: {response.text}")
    
    # ============ TAB 2: FEATURE ENGINEERING ============
    with tab2:
        if st.button("Extract Features", use_container_width=True):
            with st.spinner("Extracting features..."):
                uploaded_file.seek(0)
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                response = requests.post(f"{API_URL}/features/extract", files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"Extracted {result['total_new_features']} new features!")
                    st.metric("Original Shape", f"{result['original_shape'][0]} x {result['original_shape'][1]}")
                    st.metric("New Shape", f"{result['new_shape'][0]} x {result['new_shape'][1]}")
                    
                    engineered_df = pd.DataFrame(result["data"])
                    st.dataframe(engineered_df.head(100))
                    
                    csv = engineered_df.to_csv(index=False)
                    st.download_button("Download Engineered Data", csv, "engineered_data.csv", "text/csv")
                else:
                    st.error(f"Error: {response.text}")
    
    # ============ TAB 3: TIME SERIES ============
    with tab3:
        date_cols = []
        for col in df.columns:
            try:
                pd.to_datetime(df[col])
                date_cols.append(col)
            except:
                pass
        
        if date_cols:
            date_column = st.selectbox("Select Date Column", date_cols)
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            target_column = st.selectbox("Select Target Column", ["None"] + numeric_cols)
            
            if st.button("Analyze Time Series", use_container_width=True):
                with st.spinner("Analyzing..."):
                    uploaded_file.seek(0)
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    params = {"date_column": date_column}
                    if target_column != "None":
                        params["target_column"] = target_column
                    
                    response = requests.post(f"{API_URL}/timeseries/analyze", files=files, params=params)
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.json(result.get("basic_info", {}))
                        
                        stationarity = result.get("stationarity", {})
                        if stationarity:
                            if stationarity.get("is_stationary"):
                                st.success("Data is STATIONARY")
                            else:
                                st.warning("Data is NON-STATIONARY")
                        
                        st.subheader("Model Recommendations")
                        for rec in result.get("model_recommendations", []):
                            st.info(f"{rec['model']} - {rec['reason']}")
                    else:
                        st.error(f"Error: {response.text}")
        else:
            st.warning("No date column found for time series analysis")
    
    # ============ TAB 4: CLEAN DATA ============
    with tab4:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Rule-Based Cleaning", use_container_width=True):
                with st.spinner("Cleaning..."):
                    uploaded_file.seek(0)
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    response = requests.post(f"{API_URL}/clean/rule", files=files, params={"max_rows": max_rows})
                    
                    if response.status_code == 200:
                        result = response.json()
                        cleaned_df = pd.DataFrame(result["data"])
                        st.success(f"Cleaned {result['original_rows']} rows")
                        st.dataframe(cleaned_df.head(100))
                        
                        csv = cleaned_df.to_csv(index=False)
                        st.download_button("Download CSV", csv, "cleaned_data.csv", "text/csv")
                    else:
                        st.error(f"Error: {response.text}")
        
        with col2:
            if st.button("AI-Powered Cleaning", use_container_width=True):
                with st.spinner("AI cleaning..."):
                    uploaded_file.seek(0)
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    response = requests.post(f"{API_URL}/clean/ai", files=files, params={"max_rows": min(max_rows, 500)})
                    
                    if response.status_code == 200:
                        result = response.json()
                        cleaned_df = pd.DataFrame(result["data"])
                        st.success(f"AI cleaned {result['original_rows']} rows")
                        st.dataframe(cleaned_df.head(100))
                        
                        csv = cleaned_df.to_csv(index=False)
                        st.download_button("Download CSV", csv, "cleaned_data_ai.csv", "text/csv")
                    else:
                        st.error(f"Error: {response.text}")

else:
    st.info("Please upload a CSV or Excel file to start cleaning")

st.markdown("---")
st.caption("AI Data Cleaning Agent | Powered by Groq AI, FastAPI, and Streamlit")