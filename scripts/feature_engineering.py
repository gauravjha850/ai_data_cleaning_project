import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineering:
    def __init__(self, df, target_column=None):
        self.df = df.copy()
        self.target_column = target_column
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        self.created_features = []

    def create_polynomial_features(self, columns=None, degree=2):
        if columns is None:
            columns = self.numeric_cols[:3]
        
        poly_features = []
        for i in range(len(columns)):
            for j in range(i, len(columns)):
                if i == j and degree >= 2:
                    new_col = f"{columns[i]}_squared"
                    self.df[new_col] = self.df[columns[i]] ** 2
                    poly_features.append(new_col)
                elif i != j:
                    new_col = f"{columns[i]}_{columns[j]}_interaction"
                    self.df[new_col] = self.df[columns[i]] * self.df[columns[j]]
                    poly_features.append(new_col)
        
        self.created_features.extend(poly_features)
        return self.df, poly_features

    def create_date_features(self, date_column=None):
        if date_column is None and self.datetime_cols:
            date_column = self.datetime_cols[0]
        
        if date_column and date_column in self.df.columns:
            self.df[date_column] = pd.to_datetime(self.df[date_column])
            
            date_features = []
            self.df[f"{date_column}_year"] = self.df[date_column].dt.year
            self.df[f"{date_column}_month"] = self.df[date_column].dt.month
            self.df[f"{date_column}_day"] = self.df[date_column].dt.day
            self.df[f"{date_column}_dayofweek"] = self.df[date_column].dt.dayofweek
            self.df[f"{date_column}_quarter"] = self.df[date_column].dt.quarter
            
            date_features = [f"{date_column}_year", f"{date_column}_month", f"{date_column}_day", 
                           f"{date_column}_dayofweek", f"{date_column}_quarter"]
            
            self.created_features.extend(date_features)
            return self.df, date_features
        
        return self.df, []

    def create_ratio_features(self):
        ratio_features = []
        
        if len(self.numeric_cols) >= 2:
            for i in range(min(3, len(self.numeric_cols))):
                for j in range(i+1, min(4, len(self.numeric_cols))):
                    col1, col2 = self.numeric_cols[i], self.numeric_cols[j]
                    ratio_col = f"{col1}_to_{col2}_ratio"
                    self.df[ratio_col] = self.df[col1] / (self.df[col2] + 1e-6)
                    ratio_features.append(ratio_col)
        
        self.created_features.extend(ratio_features)
        return self.df, ratio_features

    # FIXED: No interval objects - convert to string labels
    def create_binning_features(self, columns=None, bins=5):
        if columns is None:
            columns = self.numeric_cols[:3]
        
        bin_features = []
        for col in columns:
            if col in self.df.columns:
                new_col = f"{col}_binned"
                # Convert to string labels to avoid JSON serialization issues
                labels = [f"{col}_bin_{i+1}" for i in range(bins)]
                self.df[new_col] = pd.cut(self.df[col], bins=bins, labels=labels)
                bin_features.append(new_col)
        
        self.created_features.extend(bin_features)
        return self.df, bin_features

    def create_log_features(self, columns=None):
        if columns is None:
            columns = []
            for col in self.numeric_cols:
                if len(self.df[col].dropna()) > 0 and abs(self.df[col].skew()) > 1:
                    columns.append(col)
            columns = columns[:3]
        
        log_features = []
        for col in columns:
            if col in self.df.columns:
                new_col = f"{col}_log"
                self.df[new_col] = np.log1p(self.df[col].clip(lower=0))
                log_features.append(new_col)
        
        self.created_features.extend(log_features)
        return self.df, log_features

    def create_one_hot_encoding(self, columns=None, max_categories=10):
        if columns is None:
            columns = self.categorical_cols
        
        ohe_features = []
        for col in columns[:3]:
            if self.df[col].nunique() <= max_categories:
                dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=True)
                self.df = pd.concat([self.df, dummies], axis=1)
                ohe_features.extend(dummies.columns.tolist())
        
        self.created_features.extend(ohe_features)
        return self.df, ohe_features

    def create_aggregate_features(self, group_column=None, agg_columns=None):
        if group_column is None and self.categorical_cols:
            group_column = self.categorical_cols[0]
        
        if agg_columns is None:
            agg_columns = self.numeric_cols[:2]
        
        agg_features = []
        if group_column and group_column in self.df.columns:
            for col in agg_columns:
                if col in self.df.columns:
                    mean_col = f"{col}_mean_by_{group_column}"
                    self.df[mean_col] = self.df.groupby(group_column)[col].transform('mean')
                    agg_features.append(mean_col)
        
        self.created_features.extend(agg_features)
        return self.df, agg_features

    def scale_features(self, columns=None, method='standard'):
        if columns is None:
            columns = self.numeric_cols[:5]
        
        if method == 'standard':
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        
        self.df[columns] = scaler.fit_transform(self.df[columns])
        return self.df, scaler

    def drop_highly_correlated(self, threshold=0.95):
        if len(self.numeric_cols) < 2:
            return self.df, []
        
        corr_matrix = self.df[self.numeric_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        dropped = to_drop.copy()
        self.df = self.df.drop(columns=to_drop)
        
        return self.df, dropped

    def get_feature_importance_indicators(self):
        importance = []
        for col in self.numeric_cols:
            importance.append({
                "feature": col,
                "variance": round(self.df[col].var(), 2),
                "unique_ratio": round(self.df[col].nunique() / len(self.df), 3),
                "missing_ratio": round(self.df[col].isnull().mean(), 3)
            })
        return sorted(importance, key=lambda x: x['variance'], reverse=True)[:10]

    # FIXED: Simplified auto_feature_engineering (removed binning to avoid errors)
    def auto_feature_engineering(self, include_advanced=True):
        features_created = {}
        
        if len(self.numeric_cols) >= 2:
            self.df, poly_feats = self.create_polynomial_features()
            features_created['polynomial'] = poly_feats
        
        if self.datetime_cols:
            self.df, date_feats = self.create_date_features()
            features_created['date'] = date_feats
        
        if len(self.numeric_cols) >= 2:
            self.df, ratio_feats = self.create_ratio_features()
            features_created['ratio'] = ratio_feats
        
        if include_advanced:
            # Binning disabled to avoid JSON issues
            # self.df, bin_feats = self.create_binning_features()
            # features_created['binning'] = bin_feats
            
            self.df, log_feats = self.create_log_features()
            features_created['log'] = log_feats
            
            if self.categorical_cols:
                self.df, ohe_feats = self.create_one_hot_encoding()
                features_created['one_hot'] = ohe_feats
            
            if self.categorical_cols and self.numeric_cols:
                self.df, agg_feats = self.create_aggregate_features()
                features_created['aggregate'] = agg_feats
        
        return self.df, features_created

    def get_feature_summary(self):
        return {
            "total_features_created": len(self.created_features),
            "created_features_list": self.created_features[:20],
            "original_columns": len(self.df.columns) - len(self.created_features),
            "final_columns": len(self.df.columns),
            "numeric_columns": len(self.numeric_cols),
            "categorical_columns": len(self.categorical_cols),
            "datetime_columns": len(self.datetime_cols)
        }