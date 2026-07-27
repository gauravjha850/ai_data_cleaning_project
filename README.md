# 🧹 AI Data Cleaning Agent

An end-to-end AI-powered data cleaning and analysis platform with FastAPI backend, Streamlit frontend, and Groq AI integration.

---

## 📌 Overview

**AI Data Cleaning Agent** is a comprehensive data analysis platform that combines traditional data cleaning, AI-powered preprocessing, exploratory data analysis (EDA), feature engineering, and time series forecasting. Built for data analysts and data scientists, it automates the most time-consuming aspects of data preparation and analysis.

### 🎯 Purpose

- **Automate Data Cleaning** — Reduce manual data preprocessing time from hours to minutes
- **Enable Deep EDA** — Generate comprehensive statistical reports and visualizations
- **Accelerate Feature Engineering** — Automatically create 15+ new features for ML models
- **Simplify Time Series Analysis** — Perform stationarity tests and get model recommendations

### 📊 Key Metrics

| Metric | Value |
|--------|-------|
| **Processing Capacity** | 50,000+ rows |
| **File Size Support** | 10MB+ |
| **Visualizations** | 12+ interactive plots |
| **Automated Features** | 15+ features |
| **Data Sources** | 4 (CSV, Excel, API, PostgreSQL) |
| **API Endpoints** | 8 REST endpoints |
| **Model Recommendations** | 3 (ARIMA, Prophet, LSTM) |

---

## 🔑 Key Features

| Feature | Description |
|---------|-------------|
| **📁 Multi-Source Data Loading** | CSV, Excel, REST API, PostgreSQL |
| **🧹 AI-Powered Cleaning** | Groq Llama 3.1 LLM for intelligent data cleaning (800+ tokens/sec) |
| **🔧 Rule-Based Cleaning** | Fast, traditional cleaning for missing values, duplicates, and outliers |
| **📊 Comprehensive EDA** | 12+ interactive visualizations + statistical summaries |
| **⚙️ Automated Feature Engineering** | 15+ new features (polynomial, ratio, log, date) |
| **📈 Time Series Analysis** | ADF stationarity test, ACF/PACF (40 lags), forecasting recommendations |
| **🗄️ Database Integration** | Save cleaned data directly to PostgreSQL |
| **📥 Export Capabilities** | Download cleaned data as CSV |

---

## 💡 Why This Project Matters for Data Analysts

### 1. Demonstrates Full EDA Lifecycle

- **Statistical Summary**: Mean, median, std, skewness, kurtosis for 5+ numeric columns
- **Missing Value Analysis**: Comprehensive missing data detection and imputation strategies
- **Correlation Analysis**: Strong correlation detection (>0.5) with direction identification
- **Outlier Detection**: IQR-based outlier identification with percentage analysis
- **Data Quality Score**: Automatic quality scoring (0-100) with actionable recommendations

### 2. Shows Time Series Analytical Skills

- **ADF Stationarity Test**: Statistical test for time series stationarity (p-value < 0.05)
- **ACF/PACF Analysis**: 40-lag autocorrelation and partial autocorrelation analysis
- **Seasonality Detection**: Auto-detection of seasonal patterns (7, 12, 24, 52, 365 days)
- **Model Recommendations**: Based on data characteristics — ARIMA, Prophet, LSTM

### 3. Proves Feature Engineering Proficiency

- **Polynomial Features**: Squared terms and interaction terms
- **Ratio Features**: Automated ratio creation between numeric columns
- **Log Transformations**: Handles skewed distributions
- **Date Feature Extraction**: Year, month, day, dayofweek, quarter

### 4. Demonstrates Technical Breadth

- **Full-Stack Development**: FastAPI backend + Streamlit frontend
- **LLM Integration**: Groq AI for intelligent data processing
- **Database Management**: PostgreSQL with SQLAlchemy ORM
- **API Development**: 8 REST endpoints with Swagger documentation

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| **Backend** | FastAPI, Uvicorn |
| **Frontend** | Streamlit, Plotly |
| **AI/LLM** | Groq (Llama 3.1), LangGraph |
| **ML/Stats** | Scikit-learn, StatsModels, SciPy |
| **Data Processing** | Pandas, NumPy |
| **Database** | SQLAlchemy, PostgreSQL |
| **Visualization** | Plotly, Matplotlib |
| **Dev Tools** | Git, GitHub, VS Code, Docker |

### 🧠 ML/DL Techniques Used

| Technique | Application |
|-----------|-------------|
| **LSTM Networks** | Time series forecasting |
| **QLoRA / LoRA** | Parameter-efficient LLM fine-tuning |
| **PEFT** | Efficient model adaptation |
| **RAG** | Retrieval-augmented generation |
| **Transformers** | LLM-based data processing |

---

## 📦 Installation

### Prerequisites

- Python 3.11+
- Groq API Key ([Get free key](https://console.groq.com/keys))

### Step 1: Clone the Repository

```bash
git clone https://github.com/gauravjha850/ai_data_cleaning_project.git
cd ai_data_cleaning_project
