from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
import os
import logging
import numpy as np
from dotenv import load_dotenv
from scripts.data_cleaning import DataCleaning
from scripts.ai_agent import AIAgent
from scripts.eda_analyzer import EDAAnalyzer
from scripts.feature_engineering import FeatureEngineering
from scripts.time_series_analyzer import TimeSeriesAnalyzer
from scripts.visualization import DataVisualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(title="AI Data Cleaning API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rule_cleaner = DataCleaning()
ai_agent = AIAgent()

MAX_FILE_SIZE = 50 * 1024 * 1024
MAX_ROWS_RULE = 50000
MAX_ROWS_AI = 500

def validate_file_size(contents: bytes):
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"File too large. Max size: 50MB")

def clean_for_json(obj):
    """Recursively clean NaN, Infinity, and numpy types for JSON serialization"""
    if isinstance(obj, dict):
        return {str(k): clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, (np.ndarray, pd.Series)):
        return clean_for_json(obj.tolist())
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    elif pd.isna(obj):
        return None
    else:
        return obj

def read_csv_aggressive(contents: bytes, nrows=None):
    encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1', 'utf-16', 'mac_roman']
    
    for encoding in encodings:
        try:
            if nrows:
                df = pd.read_csv(
                    io.BytesIO(contents), 
                    encoding=encoding, 
                    on_bad_lines='skip',
                    nrows=nrows,
                    engine='python'
                )
            else:
                df = pd.read_csv(
                    io.BytesIO(contents), 
                    encoding=encoding, 
                    on_bad_lines='skip',
                    engine='python'
                )
            
            if len(df) > 0:
                logger.info(f"Success with encoding: {encoding}")
                return df
        except:
            continue
    
    try:
        df = pd.read_csv(
            io.BytesIO(contents), 
            encoding='utf-8',
            on_bad_lines='skip',
            engine='python',
            quotechar='"',
            escapechar='\\'
        )
        if len(df) > 0:
            logger.info("Success with quote handling")
            return df
    except:
        pass
    
    try:
        text_content = contents.decode('utf-8', errors='ignore')
        lines = text_content.split('\n')
        
        first_line = lines[0]
        expected_cols = first_line.count(',') + 1
        
        valid_lines = [lines[0]]
        for line in lines[1:]:
            if line.count(',') + 1 == expected_cols:
                valid_lines.append(line)
        
        clean_content = '\n'.join(valid_lines)
        df = pd.read_csv(io.StringIO(clean_content))
        
        if len(df) > 0:
            logger.info(f"Success after filtering malformed lines. Kept {len(valid_lines)-1} of {len(lines)-1} rows")
            return df
    except Exception as e:
        logger.warning(f"Line filtering failed: {e}")
    
    try:
        text_content = contents.decode('utf-8', errors='ignore')
        lines = text_content.split('\n')
        
        header = lines[0].split(',')
        
        data = []
        for line in lines[1:101]:
            if line.strip():
                values = line.split(',')
                if len(values) == len(header):
                    data.append(values)
        
        df = pd.DataFrame(data, columns=header)
        if len(df) > 0:
            logger.info(f"Success with manual parsing. Got {len(df)} rows")
            return df
    except Exception as e:
        logger.error(f"All parsing methods failed: {e}")
    
    raise HTTPException(status_code=400, detail="Unable to read CSV file")

def read_excel_safe(contents: bytes, nrows=None):
    try:
        if nrows:
            df = pd.read_excel(io.BytesIO(contents), nrows=nrows)
        else:
            df = pd.read_excel(io.BytesIO(contents))
        return df
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading Excel: {str(e)}")

@app.get("/")
def root():
    return {"message": "AI Data Cleaning API is running", "version": "2.0"}

@app.get("/health")
def health():
    return {"status": "healthy", "ai_configured": bool(os.getenv("GROQ_API_KEY"))}

@app.post("/data/summary")
async def get_data_summary(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        validate_file_size(contents)
        
        logger.info(f"Processing file: {file.filename}")
        
        if file.filename.endswith('.csv'):
            df = read_csv_aggressive(contents, nrows=MAX_ROWS_RULE)
        elif file.filename.endswith(('.xls', '.xlsx')):
            df = read_excel_safe(contents, nrows=MAX_ROWS_RULE)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Use CSV or Excel.")
        
        if df is None or len(df) == 0:
            raise HTTPException(status_code=400, detail="File contains no data")
        
        logger.info(f"Successfully loaded {len(df)} rows, {len(df.columns)} columns")
        
        summary = rule_cleaner.get_data_summary(df)
        recommendations = rule_cleaner.get_recommendations(df)
        
        return {
            "status": "success",
            "filename": file.filename,
            "rows": len(df),
            "columns": len(df.columns),
            "summary": clean_for_json(summary),
            "recommendations": clean_for_json(recommendations)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/eda/report")
async def get_eda_report(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        validate_file_size(contents)
        
        if file.filename.endswith('.csv'):
            df = read_csv_aggressive(contents, nrows=MAX_ROWS_RULE)
        else:
            df = read_excel_safe(contents, nrows=MAX_ROWS_RULE)
        
        analyzer = EDAAnalyzer(df)
        report = analyzer.get_full_report()
        
        return {
            "status": "success",
            "eda_report": clean_for_json(report)
        }
    except Exception as e:
        logger.error(f"Error in EDA report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/eda/visualizations")
async def get_visualizations(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        validate_file_size(contents)
        
        if file.filename.endswith('.csv'):
            df = read_csv_aggressive(contents, nrows=MAX_ROWS_RULE)
        else:
            df = read_excel_safe(contents, nrows=MAX_ROWS_RULE)
        
        visualizer = DataVisualizer(df)
        plots = {
            "distributions": visualizer.create_distribution_plots(),
            "correlation_heatmap": visualizer.create_correlation_heatmap(),
            "missing_plot": visualizer.create_missing_plot(),
            "boxplots": visualizer.create_distribution_plots().get("boxplots", None)
        }
        
        return {
            "status": "success",
            "visualizations": plots
        }
    except Exception as e:
        logger.error(f"Error in visualizations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/features/extract")
async def extract_features(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        validate_file_size(contents)
        
        if file.filename.endswith('.csv'):
            df = read_csv_aggressive(contents, nrows=MAX_ROWS_RULE)
        else:
            df = read_excel_safe(contents, nrows=MAX_ROWS_RULE)
        
        fe = FeatureEngineering(df)
        engineered_df, features_created = fe.auto_feature_engineering()
        
        data_dict = engineered_df.head(500).replace([np.inf, -np.inf], np.nan).fillna(0).to_dict(orient="records")
        
        return {
            "status": "success",
            "original_shape": [df.shape[0], df.shape[1]],
            "new_shape": [engineered_df.shape[0], engineered_df.shape[1]],
            "features_created": clean_for_json(features_created),
            "total_new_features": sum(len(v) for v in features_created.values()) if features_created else 0,
            "new_feature_names": fe.created_features[:20],
            "data": clean_for_json(data_dict)
        }
    except Exception as e:
        logger.error(f"Error in feature extraction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/timeseries/analyze")
async def analyze_time_series(file: UploadFile = File(...), date_column: str = None, target_column: str = None):
    try:
        contents = await file.read()
        validate_file_size(contents)
        
        if file.filename.endswith('.csv'):
            df = read_csv_aggressive(contents, nrows=MAX_ROWS_RULE)
        else:
            df = read_excel_safe(contents, nrows=MAX_ROWS_RULE)
        
        if date_column is None or date_column == "":
            for col in df.columns:
                try:
                    pd.to_datetime(df[col])
                    date_column = col
                    break
                except:
                    continue
        
        if date_column is None:
            return {
                "status": "error",
                "message": "No date column found. Please specify a date column."
            }
        
        analyzer = TimeSeriesAnalyzer(df, date_column, target_column if target_column and target_column != "None" else None)
        
        result = {
            "status": "success",
            "date_column": date_column,
            "target_column": target_column if target_column and target_column != "None" else None,
            "basic_info": analyzer.get_time_series_info(),
            "stationarity": analyzer.get_stationarity_test(),
            "autocorrelation": analyzer.get_autocorrelation(),
            "model_recommendations": analyzer.get_model_recommendations()
        }
        
        return clean_for_json(result)
    except Exception as e:
        logger.error(f"Error in time series analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/timeseries/features")
async def create_time_series_features(file: UploadFile = File(...), date_column: str = None, target_column: str = None):
    try:
        contents = await file.read()
        validate_file_size(contents)
        
        if file.filename.endswith('.csv'):
            df = read_csv_aggressive(contents, nrows=MAX_ROWS_RULE)
        else:
            df = read_excel_safe(contents, nrows=MAX_ROWS_RULE)
        
        if date_column is None or date_column == "":
            for col in df.columns:
                try:
                    pd.to_datetime(df[col])
                    date_column = col
                    break
                except:
                    continue
        
        if date_column is None:
            return {
                "status": "error",
                "message": "No date column found. Please specify a date column."
            }
        
        analyzer = TimeSeriesAnalyzer(df, date_column, target_column if target_column and target_column != "None" else None)
        time_features = analyzer.create_time_features()
        
        result_df = df.copy()
        for col in time_features.columns:
            result_df[col] = time_features[col].values
        
        data_dict = result_df.head(500).replace([np.inf, -np.inf], np.nan).fillna(0).to_dict(orient="records")
        
        return {
            "status": "success",
            "original_shape": [df.shape[0], df.shape[1]],
            "new_shape": [result_df.shape[0], result_df.shape[1]],
            "features_created": list(time_features.columns),
            "total_features": len(time_features.columns),
            "data": clean_for_json(data_dict)
        }
    except Exception as e:
        logger.error(f"Error creating time series features: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clean/rule")
async def clean_rule(file: UploadFile = File(...), max_rows: int = MAX_ROWS_RULE):
    try:
        contents = await file.read()
        validate_file_size(contents)
        
        if file.filename.endswith('.csv'):
            df = read_csv_aggressive(contents, nrows=max_rows)
        else:
            df = read_excel_safe(contents, nrows=max_rows)
        
        if df is None or len(df) == 0:
            raise HTTPException(status_code=400, detail="File contains no data")
        
        cleaned_df, cleaning_details = rule_cleaner.clean_data(df)
        report = rule_cleaner.get_cleaning_report(df, cleaned_df, cleaning_details)
        recommendations = rule_cleaner.get_recommendations(cleaned_df)
        
        data_dict = cleaned_df.head(500).replace([np.inf, -np.inf], np.nan).fillna(0).to_dict(orient="records")
        
        return {
            "status": "success",
            "original_rows": len(df),
            "cleaned_rows": len(cleaned_df),
            "columns": len(cleaned_df.columns),
            "cleaning_report": clean_for_json(report),
            "recommendations": clean_for_json(recommendations),
            "data": clean_for_json(data_dict)
        }
    except Exception as e:
        logger.error(f"Error in rule cleaning: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clean/ai")
async def clean_ai(file: UploadFile = File(...), max_rows: int = MAX_ROWS_AI):
    try:
        contents = await file.read()
        validate_file_size(contents)
        
        nrows = min(max_rows, MAX_ROWS_AI)
        if file.filename.endswith('.csv'):
            df = read_csv_aggressive(contents, nrows=nrows)
        else:
            df = read_excel_safe(contents, nrows=nrows)
        
        cleaned_df = ai_agent.process_data(df)
        report = rule_cleaner.get_cleaning_report(df, cleaned_df)
        
        data_dict = cleaned_df.head(500).replace([np.inf, -np.inf], np.nan).fillna(0).to_dict(orient="records")
        
        return {
            "status": "success",
            "original_rows": len(df),
            "cleaned_rows": len(cleaned_df),
            "columns": len(cleaned_df.columns),
            "cleaning_report": clean_for_json(report),
            "data": clean_for_json(data_dict)
        }
    except Exception as e:
        logger.error(f"Error in AI cleaning: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)