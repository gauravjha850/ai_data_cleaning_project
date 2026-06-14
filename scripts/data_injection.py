import pandas as pd
import requests
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv()

class DataInjection:
    def __init__(self, db_url=None):
        if db_url:
            self.engine = create_engine(db_url)
        else:
            # Try to get from environment variable
            db_url = os.getenv("DATABASE_URL")
            self.engine = create_engine(db_url) if db_url else None
        
        if self.engine:
            print("Database engine initialized successfully")
        else:
            print("No database URL provided")

    def load_csv(self, file_path):
        try:
            df = pd.read_csv(file_path)
            print(f"CSV loaded successfully: {len(df)} rows")
            return df
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return None

    def load_excel(self, file_path):
        try:
            df = pd.read_excel(file_path)
            print(f"Excel loaded successfully: {len(df)} rows")
            return df
        except Exception as e:
            print(f"Error loading Excel: {e}")
            return None

    def load_from_database(self, query):
        if not self.engine:
            raise ValueError("Database engine not configured. Please provide DATABASE_URL")
        try:
            df = pd.read_sql(query, self.engine)
            print(f"Database query executed successfully: {len(df)} rows")
            return df
        except Exception as e:
            print(f"Error reading from database: {e}")
            raise e

    def save_to_database(self, df, table_name, if_exists='replace'):
        if not self.engine:
            raise ValueError("Database engine not configured")
        try:
            df.to_sql(table_name, self.engine, if_exists=if_exists, index=False)
            print(f"Data saved to table '{table_name}' successfully: {len(df)} rows")
            return True
        except Exception as e:
            print(f"Error saving to database: {e}")
            return False

    def get_table_names(self):
        if not self.engine:
            return []
        try:
            from sqlalchemy import inspect
            inspector = inspect(self.engine)
            tables = inspector.get_table_names()
            print(f"Found tables: {tables}")
            return tables
        except Exception as e:
            print(f"Error getting tables: {e}")
            return []

    def fetch_from_api(self, api_url, params=None):
        try:
            response = requests.get(api_url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                elif isinstance(data, dict) and 'data' in data:
                    df = pd.DataFrame(data['data'])
                else:
                    df = pd.DataFrame([data])
                print(f"API data fetched successfully: {len(df)} rows")
                return df
            else:
                print(f"API error: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error fetching API: {e}")
            return None