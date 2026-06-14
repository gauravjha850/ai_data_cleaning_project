import pandas as pd
import numpy as np

class EDAAnalyzer:
    def __init__(self, df):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

    def _convert(self, obj):
        """Convert numpy types to Python native types"""
        if isinstance(obj, dict):
            return {k: self._convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif pd.isna(obj):
            return None
        else:
            return obj

    def get_basic_info(self):
        return self._convert({
            "rows": len(self.df),
            "columns": len(self.df.columns),
            "numeric_columns": len(self.numeric_cols),
            "categorical_columns": len(self.categorical_cols),
            "datetime_columns": len(self.datetime_cols),
            "memory_mb": round(self.df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
        })

    def get_missing_analysis(self):
        missing = self.df.isnull().sum()
        return self._convert({
            "total_missing": int(missing.sum()),
            "missing_percentage": round((missing.sum() / (len(self.df) * len(self.df.columns))) * 100, 2),
            "columns_with_missing": missing[missing > 0].index.tolist(),
            "missing_by_column": {str(k): int(v) for k, v in missing[missing > 0].to_dict().items()}
        })

    def get_numeric_summary(self):
        summary = {}
        for col in self.numeric_cols[:5]:
            data = self.df[col].dropna()
            if len(data) > 0:
                summary[col] = {
                    "mean": round(data.mean(), 2),
                    "median": round(data.median(), 2),
                    "std": round(data.std(), 2),
                    "min": round(data.min(), 2),
                    "max": round(data.max(), 2),
                    "range": round(data.max() - data.min(), 2),
                    "skewness": round(data.skew(), 3),
                    "kurtosis": round(data.kurtosis(), 3),
                    "missing_count": int(self.df[col].isnull().sum()),
                    "missing_percentage": round((self.df[col].isnull().sum() / len(self.df)) * 100, 2)
                }
        return self._convert(summary)

    def get_correlation_analysis(self):
        if len(self.numeric_cols) < 2:
            return {}
        
        corr_matrix = self.df[self.numeric_cols].corr()
        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.5:
                    strong_corr.append({
                        "columns": f"{corr_matrix.columns[i]} & {corr_matrix.columns[j]}",
                        "correlation": round(corr_value, 3),
                        "direction": "positive" if corr_value > 0 else "negative"
                    })
        return self._convert({
            "strong_correlations": strong_corr[:5],
            "highest_correlation": strong_corr[0] if strong_corr else None
        })

    def get_outlier_analysis(self):
        outliers = {}
        for col in self.numeric_cols[:5]:
            data = self.df[col].dropna()
            if len(data) > 0:
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_count = len(data[(data < lower_bound) | (data > upper_bound)])
                outliers[col] = {
                    "outlier_count": int(outlier_count),
                    "outlier_percentage": round((outlier_count / len(data)) * 100, 2),
                    "has_outliers": outlier_count > 0,
                    "lower_bound": round(lower_bound, 2),
                    "upper_bound": round(upper_bound, 2)
                }
        return self._convert(outliers)

    def get_unique_analysis(self):
        unique_analysis = {}
        for col in self.df.columns[:10]:
            unique_count = self.df[col].nunique()
            unique_analysis[col] = {
                "unique_values": int(unique_count),
                "unique_percentage": round((unique_count / len(self.df)) * 100, 2),
                "is_constant": unique_count == 1,
                "is_high_cardinality": unique_count > 100
            }
        return self._convert(unique_analysis)

    def get_quality_score(self):
        score = 100
        
        missing_pct = (self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))) * 100
        score -= missing_pct * 0.5
        
        duplicate_pct = (self.df.duplicated().sum() / len(self.df)) * 100
        score -= duplicate_pct * 0.3
        
        constant_cols = len([col for col in self.df.columns if self.df[col].nunique() == 1])
        score -= constant_cols * 2
        
        return int(max(0, min(100, round(score, 2))))

    def get_recommendations(self):
        recommendations = []
        
        if self.df.isnull().sum().sum() > 0:
            recommendations.append({
                "issue": "Missing values detected",
                "severity": "High",
                "action": "Fill with mean/median/mode or drop rows/columns"
            })
        
        if self.df.duplicated().sum() > 0:
            recommendations.append({
                "issue": f"Duplicate rows found ({self.df.duplicated().sum()} rows)",
                "severity": "Medium",
                "action": "Remove duplicate rows using drop_duplicates()"
            })
        
        outliers = self.get_outlier_analysis()
        outlier_cols = [col for col, info in outliers.items() if info["has_outliers"]]
        if outlier_cols:
            recommendations.append({
                "issue": f"Outliers detected in: {', '.join(outlier_cols[:3])}",
                "severity": "Medium",
                "action": "Cap, transform, or remove outliers for better model performance"
            })
        
        if not recommendations:
            recommendations.append({
                "issue": "No major issues found",
                "severity": "Low",
                "action": "Data is ready for analysis and modeling"
            })
        
        return self._convert(recommendations)

    def get_full_report(self):
        return {
            "basic_info": self.get_basic_info(),
            "missing_data": self.get_missing_analysis(),
            "numeric_stats": self.get_numeric_summary(),
            "correlations": self.get_correlation_analysis(),
            "outliers": self.get_outlier_analysis(),
            "unique_analysis": self.get_unique_analysis(),
            "quality_score": self.get_quality_score(),
            "recommendations": self.get_recommendations()
        }