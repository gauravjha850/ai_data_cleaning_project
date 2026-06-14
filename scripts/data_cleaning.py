import pandas as pd
import numpy as np

class DataCleaning:
    def handle_missing_values(self, df, strategy="mean"):
        df_cleaned = df.copy()
        missing_before = df_cleaned.isnull().sum().sum()
        
        for col in df_cleaned.columns:
            if df_cleaned[col].isnull().any():
                if strategy == "mean" and pd.api.types.is_numeric_dtype(df_cleaned[col]):
                    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mean())
                elif strategy == "median" and pd.api.types.is_numeric_dtype(df_cleaned[col]):
                    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
                elif strategy == "mode":
                    mode_val = df_cleaned[col].mode()
                    if not mode_val.empty:
                        df_cleaned[col] = df_cleaned[col].fillna(mode_val[0])
                elif strategy == "drop":
                    df_cleaned = df_cleaned.dropna(subset=[col])
        
        missing_after = df_cleaned.isnull().sum().sum()
        return df_cleaned, missing_before - missing_after

    def remove_duplicates(self, df):
        before = len(df)
        df_cleaned = df.drop_duplicates()
        after = len(df_cleaned)
        return df_cleaned, before - after

    def fix_data_types(self, df):
        df_cleaned = df.copy()
        type_changes = []
        
        for col in df_cleaned.columns:
            original_type = df_cleaned[col].dtype
            if df_cleaned[col].dtype == 'object':
                try:
                    df_cleaned[col] = pd.to_numeric(df_cleaned[col])
                    type_changes.append(f"{col}: {original_type} -> numeric")
                except:
                    pass
                
                try:
                    df_cleaned[col] = pd.to_datetime(df_cleaned[col])
                    type_changes.append(f"{col}: {original_type} -> datetime")
                except:
                    pass
        
        return df_cleaned, type_changes

    def remove_outliers(self, df, column=None, method='iqr', threshold=1.5):
        """Remove outliers from numeric columns"""
        df_cleaned = df.copy()
        
        if column is None:
            # Apply to all numeric columns
            numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
            cols_to_process = numeric_cols
        else:
            cols_to_process = [column] if column in df_cleaned.columns else []
        
        outliers_removed = {}
        
        for col in cols_to_process:
            data = df_cleaned[col].dropna()
            if len(data) > 0:
                if method == 'iqr':
                    Q1 = data.quantile(0.25)
                    Q3 = data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    
                    before = len(df_cleaned)
                    df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
                    after = len(df_cleaned)
                    
                    outliers_removed[col] = before - after
                    print(f"Removed {outliers_removed[col]} outliers from {col}")
        
        return df_cleaned, outliers_removed

    def clean_data(self, df, remove_outliers_cols=None):
        print("Starting rule-based cleaning")
        
        # Step 1: Handle missing values
        df, missing_fixed = self.handle_missing_values(df)
        
        # Step 2: Remove duplicates
        df, duplicates_removed = self.remove_duplicates(df)
        
        # Step 3: Fix data types
        df, type_changes = self.fix_data_types(df)
        
        # Step 4: Remove outliers (if columns specified)
        outliers_removed = {}
        if remove_outliers_cols:
            df, outliers_removed = self.remove_outliers(df, column=remove_outliers_cols)
        
        return df, {
            "missing_fixed": missing_fixed,
            "duplicates_removed": duplicates_removed,
            "type_changes": type_changes,
            "outliers_removed": outliers_removed
        }

    def get_data_summary(self, df):
        """Generate comprehensive data summary"""
        summary = {
            "basic_info": {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "total_cells": len(df) * len(df.columns),
                "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
            },
            "column_info": {},
            "missing_values": {
                "total_missing": int(df.isnull().sum().sum()),
                "missing_percentage": round((df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100, 2),
                "by_column": df.isnull().sum().to_dict()
            },
            "duplicates": {
                "total_duplicates": int(df.duplicated().sum()),
                "duplicate_percentage": round((df.duplicated().sum() / len(df)) * 100, 2)
            },
            "data_types": df.dtypes.astype(str).to_dict(),
            "numeric_columns": [],
            "categorical_columns": [],
            "datetime_columns": []
        }
        
        for col in df.columns:
            col_info = {
                "dtype": str(df[col].dtype),
                "unique_values": df[col].nunique(),
                "null_count": int(df[col].isnull().sum()),
                "null_percentage": round((df[col].isnull().sum() / len(df)) * 100, 2)
            }
            
            if pd.api.types.is_numeric_dtype(df[col]):
                summary["numeric_columns"].append(col)
                col_info["min"] = float(df[col].min()) if not df[col].isnull().all() else None
                col_info["max"] = float(df[col].max()) if not df[col].isnull().all() else None
                col_info["mean"] = float(df[col].mean()) if not df[col].isnull().all() else None
                col_info["median"] = float(df[col].median()) if not df[col].isnull().all() else None
                col_info["std"] = float(df[col].std()) if not df[col].isnull().all() else None
            
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                summary["datetime_columns"].append(col)
                col_info["min"] = str(df[col].min()) if not df[col].isnull().all() else None
                col_info["max"] = str(df[col].max()) if not df[col].isnull().all() else None
            
            else:
                summary["categorical_columns"].append(col)
                top_values = df[col].value_counts().head(5).to_dict()
                col_info["top_values"] = {str(k): int(v) for k, v in top_values.items()}
            
            summary["column_info"][col] = col_info
        
        return summary

    def get_cleaning_report(self, original_df, cleaned_df, cleaning_details=None):
        """Generate detailed cleaning report"""
        original_summary = self.get_data_summary(original_df)
        cleaned_summary = self.get_data_summary(cleaned_df)
        
        report = {
            "summary": {
                "original": original_summary["basic_info"],
                "cleaned": cleaned_summary["basic_info"]
            },
            "changes": {
                "rows_removed": len(original_df) - len(cleaned_df),
                "rows_removed_percentage": round(((len(original_df) - len(cleaned_df)) / len(original_df)) * 100, 2) if len(original_df) > 0 else 0,
                "columns_changed": len(original_df.columns) - len(cleaned_df.columns),
                "missing_values_fixed": original_summary["missing_values"]["total_missing"] - cleaned_summary["missing_values"]["total_missing"],
                "duplicates_removed": original_summary["duplicates"]["total_duplicates"]
            },
            "before": {
                "missing_values": original_summary["missing_values"],
                "duplicates": original_summary["duplicates"],
                "data_types": original_summary["data_types"]
            },
            "after": {
                "missing_values": cleaned_summary["missing_values"],
                "duplicates": cleaned_summary["duplicates"],
                "data_types": cleaned_summary["data_types"]
            },
            "cleaning_actions": cleaning_details or {}
        }
        
        return report

    def get_recommendations(self, df):
        """Generate data cleaning recommendations"""
        recommendations = []
        
        # Check missing values
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            recommendations.append({
                "issue": "Missing Values Detected",
                "severity": "High" if df.isnull().sum().sum() > len(df) * 0.1 else "Medium",
                "affected_columns": missing_cols,
                "suggestion": "Fill with mean/median for numeric columns or mode for categorical columns",
                "action": "Use handle_missing_values() method"
            })
        
        # Check duplicates
        if df.duplicated().sum() > 0:
            recommendations.append({
                "issue": "Duplicate Rows Found",
                "severity": "Medium",
                "affected_rows": int(df.duplicated().sum()),
                "suggestion": "Remove duplicate rows to avoid bias",
                "action": "Use remove_duplicates() method"
            })
        
        # Check data types
        object_cols = df.select_dtypes(include=['object']).columns.tolist()
        if object_cols:
            recommendations.append({
                "issue": "Object/String Columns",
                "severity": "Low",
                "affected_columns": object_cols[:5],
                "suggestion": "Consider converting to categorical or numeric if applicable",
                "action": "Use fix_data_types() method"
            })
        
        # Check outliers
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if len(df[col].dropna()) > 0:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
                if len(outliers) > 0:
                    recommendations.append({
                        "issue": f"Outliers detected in column '{col}'",
                        "severity": "Medium",
                        "affected_rows": len(outliers),
                        "suggestion": "Consider capping or removing outliers for better model performance",
                        "action": "Use remove_outliers() method"
                    })
                    break
        
        # Check for constant columns
        for col in df.columns:
            if df[col].nunique() == 1:
                recommendations.append({
                    "issue": f"Constant column '{col}' (single value)",
                    "severity": "Low",
                    "suggestion": "Consider dropping this column as it adds no information",
                    "action": "Drop column using df.drop()"
                })
        
        if not recommendations:
            recommendations.append({
                "issue": "No Issues Found",
                "severity": "Low",
                "suggestion": "Your data looks clean!",
                "action": "Ready for analysis"
            })
        
        return recommendations