import os
import io
import pandas as pd
import json
from dotenv import load_dotenv
from groq import Groq
from langgraph.graph import StateGraph, END
from pydantic import BaseModel

load_dotenv()

class CleaningState(BaseModel):
    input_text: str = ""
    structured_response: str = ""
    task_type: str = "general"

class AIAgent:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in .env file")
        self.client = Groq(api_key=self.api_key)
        self.workflow = self._build_workflow()
        print("AI Agent initialized with Groq")
    
    def check_api_key(self):
        """Verify Groq API key is working"""
        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": "Say OK"}],
                max_tokens=5
            )
            return True
        except Exception as e:
            print(f"API key check failed: {e}")
            return False
    
    def _build_workflow(self):
        def cleaning_node(state: CleaningState):
            try:
                system_prompts = {
                    "general": "You are a data cleaning expert. Return ONLY cleaned CSV data. No explanations, no markdown.",
                    "eda": "You are a data analysis expert. Analyze the data and return insights. Return ONLY valid JSON.",
                    "feature_engineering": "You are a feature engineering expert. Suggest new features. Return ONLY valid JSON.",
                    "time_series": "You are a time series expert. Analyze and return insights. Return ONLY valid JSON."
                }
                
                system_prompt = system_prompts.get(state.task_type, system_prompts["general"])
                
                response = self.client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": state.input_text}
                    ],
                    temperature=0.1,
                    max_tokens=4000
                )
                
                result = response.choices[0].message.content
                
                # Clean up response
                if '```json' in result:
                    result = result.split('```json')[1].split('```')[0]
                elif '```csv' in result:
                    result = result.split('```csv')[1].split('```')[0]
                elif '```' in result:
                    result = result.split('```')[1].split('```')[0]
                
                return {"structured_response": result.strip()}
            except Exception as e:
                print(f"Groq API error: {e}")
                return {"structured_response": state.input_text}
        
        graph = StateGraph(CleaningState)
        graph.add_node("cleaning_agent", cleaning_node)
        graph.set_entry_point("cleaning_agent")
        graph.add_edge("cleaning_agent", END)
        return graph.compile()
    
    def process_data(self, df, batch_size=20):
        """Process data in small batches for AI cleaning"""
        if len(df) == 0:
            return df
        
        print(f"Starting AI cleaning for {len(df)} rows...")
        cleaned_chunks = []
        
        for i in range(0, min(len(df), 100), batch_size):
            batch = df.iloc[i:i+batch_size]
            prompt = f"""Clean this CSV data:
1. Fix missing values (use appropriate defaults)
2. Remove duplicate rows
3. Fix data types (numbers, dates)
4. Standardize text (trim spaces, consistent case)

Return ONLY clean CSV, no explanations:

{batch.to_csv(index=False)}"""
            
            initial_state = CleaningState(input_text=prompt, task_type="general")
            final_state = self.workflow.invoke(initial_state)
            
            # Get response text properly (handles both dict and object)
            if isinstance(final_state, dict):
                response_text = final_state.get("structured_response", "")
            else:
                response_text = final_state.structured_response
            
            try:
                text = response_text.strip()
                if '```csv' in text:
                    text = text.split('```csv')[1].split('```')[0]
                elif '```' in text:
                    text = text.split('```')[1].split('```')[0]
                
                chunk_df = pd.read_csv(io.StringIO(text))
                cleaned_chunks.append(chunk_df)
                print(f"Processed rows {i} to {i+batch_size}")
                
            except Exception as e:
                print(f"Error on batch {i}: {e}")
                cleaned_chunks.append(batch)
        
        result = pd.concat(cleaned_chunks, ignore_index=True) if cleaned_chunks else df
        print(f"AI cleaning completed. Result: {result.shape}")
        return result

    def process_large_dataset(self, df, chunk_size=1000):
        """Process large datasets in chunks to avoid memory issues"""
        print(f"Processing large dataset with {len(df)} rows in chunks of {chunk_size}")
        chunks = []
        
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size]
            print(f"Processing chunk {i//chunk_size + 1}/{(len(df)-1)//chunk_size + 1}")
            cleaned_chunk = self.process_data(chunk)
            chunks.append(cleaned_chunk)
        
        result = pd.concat(chunks, ignore_index=True)
        print(f"Large dataset processing completed. Final shape: {result.shape}")
        return result

    def get_data_insights(self, df):
        """Get AI-powered insights about the data"""
        prompt = f"""
        Analyze this dataset and provide key insights in JSON format:
        {{
            "data_quality": ["issue1", "issue2"],
            "patterns": ["pattern1", "pattern2"],
            "recommendations": ["rec1", "rec2"],
            "potential_insights": ["insight1", "insight2"]
        }}
        
        Data sample (first 20 rows):
        {df.head(20).to_csv(index=False)}
        
        Return ONLY valid JSON, no other text.
        """
        
        initial_state = CleaningState(input_text=prompt, task_type="eda")
        final_state = self.workflow.invoke(initial_state)
        
        if isinstance(final_state, dict):
            result = final_state.get("structured_response", "")
        else:
            result = final_state.structured_response
        
        try:
            return json.loads(result)
        except:
            return {"insights": result}

    def suggest_features(self, df):
        """Get AI-powered feature engineering suggestions"""
        prompt = f"""
        Based on this data, suggest 5-10 new features for machine learning.
        Return in JSON format:
        {{
            "features": [
                {{
                    "name": "feature_name",
                    "description": "what this feature represents",
                    "creation_logic": "how to create it (formula/calculation)"
                }}
            ]
        }}
        
        Data sample:
        {df.head(20).to_csv(index=False)}
        
        Return ONLY valid JSON, no other text.
        """
        
        initial_state = CleaningState(input_text=prompt, task_type="feature_engineering")
        final_state = self.workflow.invoke(initial_state)
        
        if isinstance(final_state, dict):
            result = final_state.get("structured_response", "")
        else:
            result = final_state.structured_response
        
        try:
            return json.loads(result)
        except:
            return {"features": [{"suggestion": result}]}

    def analyze_time_series(self, df, date_col=None, target_col=None):
        """Get AI-powered time series analysis"""
        prompt = f"""
        Analyze this time series data and return JSON:
        {{
            "trend": "increasing/decreasing/stable",
            "seasonality": "daily/weekly/monthly/yearly/none",
            "stationarity": "stationary/non-stationary",
            "recommended_models": ["model1", "model2"],
            "key_observations": ["obs1", "obs2"]
        }}
        
        Date column: {date_col if date_col else 'first column'}
        Target column: {target_col if target_col else 'numeric column'}
        
        Data sample:
        {df.head(30).to_csv(index=False)}
        
        Return ONLY valid JSON, no other text.
        """
        
        initial_state = CleaningState(input_text=prompt, task_type="time_series")
        final_state = self.workflow.invoke(initial_state)
        
        if isinstance(final_state, dict):
            result = final_state.get("structured_response", "")
        else:
            result = final_state.structured_response
        
        try:
            return json.loads(result)
        except:
            return {"analysis": result}

    def explain_cleaning(self, df_original, df_cleaned):
        """Get AI explanation of what was cleaned"""
        prompt = f"""
        Compare the original and cleaned data and explain what was fixed.
        Return in plain text format.
        
        Original data (first 10 rows):
        {df_original.head(10).to_csv(index=False)}
        
        Cleaned data (first 10 rows):
        {df_cleaned.head(10).to_csv(index=False)}
        
        Explain:
        1. What issues were fixed
        2. How missing values were handled
        3. What duplicates were removed
        """
        
        initial_state = CleaningState(input_text=prompt, task_type="general")
        final_state = self.workflow.invoke(initial_state)
        
        if isinstance(final_state, dict):
            return final_state.get("structured_response", "")
        else:
            return final_state.structured_response