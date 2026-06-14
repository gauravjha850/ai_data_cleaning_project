import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesAnalyzer:
    def __init__(self, df, date_column, target_column=None):
        self.df = df.copy()
        self.date_column = date_column
        self.target_column = target_column
        
        self.df[date_column] = pd.to_datetime(self.df[date_column])
        self.df = self.df.sort_values(date_column)
        self.df.set_index(date_column, inplace=True)
        
        # Detect frequency
        self.frequency = self._detect_frequency()
        self.seasonal_period = self._detect_seasonality()

    def _detect_frequency(self):
        """Auto-detect time series frequency"""
        try:
            freq = pd.infer_freq(self.df.index)
            if freq is None:
                diffs = self.df.index.to_series().diff().mode()
                if len(diffs) > 0:
                    diff = diffs.iloc[0]
                    if diff == pd.Timedelta(days=1):
                        return 'D'
                    elif diff == pd.Timedelta(days=7):
                        return 'W'
                    elif diff == pd.Timedelta(days=30):
                        return 'M'
            return freq
        except:
            return None

    def _detect_seasonality(self):
        """Detect seasonal period"""
        if self.target_column and self.target_column in self.df.columns:
            data = self.df[self.target_column].dropna()
            if len(data) > 50:
                periods = [7, 12, 24, 52, 365]
                best_period = None
                best_autocorr = 0
                for period in periods:
                    if len(data) > period * 2:
                        autocorr = data.autocorr(lag=period)
                        if abs(autocorr) > abs(best_autocorr):
                            best_autocorr = autocorr
                            best_period = period
                return best_period
        return None

    # ============ EXISTING METHODS ============

    def get_time_series_info(self):
        return {
            "date_range": {
                "start": str(self.df.index.min()),
                "end": str(self.df.index.max()),
                "duration_days": (self.df.index.max() - self.df.index.min()).days
            },
            "total_points": len(self.df),
            "frequency": self.frequency,
            "seasonal_period": self.seasonal_period
        }

    def get_stationarity_test(self):
        if not self.target_column or self.target_column not in self.df.columns:
            return {}
        
        data = self.df[self.target_column].dropna()
        
        # ADF Test
        adf_result = adfuller(data, autolag='AIC')
        
        # KPSS Test (NEW)
        try:
            kpss_result = kpss(data, regression='c', nlags='auto')
            kpss_dict = {
                "statistic": round(kpss_result[0], 4),
                "p_value": round(kpss_result[1], 4),
                "is_stationary": kpss_result[1] > 0.05
            }
        except:
            kpss_dict = {"error": "KPSS test failed"}
        
        return {
            "adf_test": {
                "statistic": round(adf_result[0], 4),
                "p_value": round(adf_result[1], 4),
                "critical_values": {k: round(v, 4) for k, v in adf_result[4].items()},
                "is_stationary": adf_result[1] < 0.05
            },
            "kpss_test": kpss_dict,
            "is_stationary": adf_result[1] < 0.05 and (kpss_dict.get("is_stationary", True) if "error" not in kpss_dict else True)
        }

    def get_autocorrelation(self, lags=40):
        if not self.target_column or self.target_column not in self.df.columns:
            return {}
        
        data = self.df[self.target_column].dropna()
        acf_values = acf(data, nlags=lags, fft=False)
        pacf_values = pacf(data, nlags=lags, method='ywm')
        
        significant_lags = [i for i in range(1, len(acf_values)) if abs(acf_values[i]) > 1.96/np.sqrt(len(data))]
        
        return {
            "significant_lags": significant_lags[:5],
            "suggested_p": significant_lags[0] if significant_lags else 1,
            "suggested_q": len([l for l in significant_lags if l < 10]) or 1,
            "acf_values": acf_values.tolist(),
            "pacf_values": pacf_values.tolist()
        }

    # ============ NEW METHODS ============

    def get_seasonal_decomposition(self, model='additive', period=None):
        """Decompose time series into trend, seasonal, residual"""
        if not self.target_column or self.target_column not in self.df.columns:
            return None
        
        data = self.df[self.target_column].dropna()
        period = period or self.seasonal_period or 7
        
        if len(data) > period * 2:
            try:
                decomposition = seasonal_decompose(data, model=model, period=period)
                return {
                    "trend": decomposition.trend.dropna().tolist(),
                    "seasonal": decomposition.seasonal[:100].tolist(),
                    "residual": decomposition.resid.dropna().tolist(),
                    "trend_strength": 1 - (decomposition.resid.var() / (decomposition.trend.var() + decomposition.resid.var())) if not np.isnan(decomposition.resid.var()) else 0,
                    "seasonal_strength": 1 - (decomposition.resid.var() / (decomposition.seasonal.var() + decomposition.resid.var())) if not np.isnan(decomposition.resid.var()) else 0
                }
            except:
                return None
        return None

    def get_rolling_statistics(self, windows=[7, 14, 30]):
        """Calculate rolling statistics"""
        if not self.target_column or self.target_column not in self.df.columns:
            return {}
        
        rolling_stats = {}
        for window in windows:
            if len(self.df) > window:
                rolling_stats[f'rolling_mean_{window}'] = self.df[self.target_column].rolling(window).mean().tolist()
                rolling_stats[f'rolling_std_{window}'] = self.df[self.target_column].rolling(window).std().tolist()
        
        return rolling_stats

    def get_forecast(self, steps=10):
        """Simple forecast using last value or moving average"""
        if not self.target_column or self.target_column not in self.df.columns:
            return {}
        
        data = self.df[self.target_column].dropna()
        
        # Simple methods
        last_value = data.iloc[-1]
        mean_value = data.mean()
        ma_7 = data.tail(7).mean()
        
        # Generate forecast dates
        last_date = self.df.index[-1]
        forecast_dates = pd.date_range(start=last_date, periods=steps+1, freq=self.frequency or 'D')[1:]
        
        return {
            "last_value_forecast": [float(last_value)] * steps,
            "mean_forecast": [float(mean_value)] * steps,
            "moving_average_forecast": [float(ma_7)] * steps,
            "forecast_dates": [str(d.date()) for d in forecast_dates]
        }

    def detect_anomalies(self, method='iqr', threshold=1.5):
        """Detect anomalies in time series"""
        if not self.target_column or self.target_column not in self.df.columns:
            return {}
        
        data = self.df[self.target_column].dropna()
        
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            anomalies = data[(data < lower_bound) | (data > upper_bound)]
            
            return {
                "anomaly_count": len(anomalies),
                "anomaly_percentage": round((len(anomalies) / len(data)) * 100, 2),
                "anomaly_indices": [str(idx.date()) for idx in anomalies.index[:10]],
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound)
            }
        return {}

    def fill_missing_values(self, method='ffill'):
        """Fill missing values in time series"""
        if method == 'ffill':
            self.df[self.target_column] = self.df[self.target_column].fillna(method='ffill')
        elif method == 'bfill':
            self.df[self.target_column] = self.df[self.target_column].fillna(method='bfill')
        elif method == 'linear':
            self.df[self.target_column] = self.df[self.target_column].interpolate(method='linear')
        elif method == 'mean':
            self.df[self.target_column] = self.df[self.target_column].fillna(self.df[self.target_column].mean())
        
        return self.df

    def create_time_features(self):
        features = pd.DataFrame(index=self.df.index)
        
        # Basic time components
        features['year'] = self.df.index.year
        features['month'] = self.df.index.month
        features['quarter'] = self.df.index.quarter
        features['day'] = self.df.index.day
        features['dayofweek'] = self.df.index.dayofweek
        features['dayofyear'] = self.df.index.dayofyear
        features['weekofyear'] = self.df.index.isocalendar().week
        
        # Cyclical features
        features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
        features['dayofweek_sin'] = np.sin(2 * np.pi * features['dayofweek'] / 7)
        features['dayofweek_cos'] = np.cos(2 * np.pi * features['dayofweek'] / 7)
        
        # Boolean flags
        features['is_weekend'] = (features['dayofweek'] >= 5).astype(int)
        features['is_month_start'] = self.df.index.is_month_start.astype(int)
        features['is_month_end'] = self.df.index.is_month_end.astype(int)
        
        # Time elapsed
        features['days_since_start'] = (self.df.index - self.df.index.min()).days
        
        # Lag features for target
        if self.target_column and self.target_column in self.df.columns:
            for lag in [1, 2, 3, 7, 14, 30]:
                if len(self.df) > lag:
                    features[f'target_lag_{lag}'] = self.df[self.target_column].shift(lag)
            
            # Rolling statistics
            for window in [7, 14, 30]:
                if len(self.df) > window:
                    features[f'target_rolling_mean_{window}'] = self.df[self.target_column].rolling(window).mean()
                    features[f'target_rolling_std_{window}'] = self.df[self.target_column].rolling(window).std()
        
        return features

    def get_model_recommendations(self):
        recommendations = []
        data_length = len(self.df)
        stationarity = self.get_stationarity_test()
        
        # Based on data length
        if data_length < 50:
            recommendations.append({"model": "Exponential Smoothing", "reason": "Limited data (<50 points)"})
        elif data_length < 200:
            recommendations.append({"model": "ARIMA/SARIMA", "reason": "Medium length data"})
        else:
            recommendations.append({"model": "Prophet (Facebook)", "reason": "Large dataset with potential seasonality"})
            recommendations.append({"model": "LSTM/GRU (Deep Learning)", "reason": "Large dataset with complex patterns"})
        
        # Based on stationarity
        if stationarity and not stationarity.get("is_stationary", True):
            recommendations.append({"model": "ARIMA with differencing", "reason": "Data is non-stationary, needs differencing"})
        
        # Based on seasonality
        if self.seasonal_period and self.seasonal_period > 0:
            recommendations.append({"model": "SARIMA/SARIMAX", "reason": f"Seasonal pattern detected (period={self.seasonal_period})"})
        
        # ML approach
        recommendations.append({"model": "XGBoost / LightGBM with time features", "reason": "When you have multiple variables"})
        
        return recommendations

    def get_full_report(self):
        """Generate complete time series analysis report"""
        return {
            "basic_info": self.get_time_series_info(),
            "stationarity": self.get_stationarity_test(),
            "autocorrelation": self.get_autocorrelation(),
            "seasonality": {
                "detected": self.seasonal_period is not None,
                "period": self.seasonal_period
            },
            "decomposition": self.get_seasonal_decomposition(),
            "anomalies": self.detect_anomalies(),
            "model_recommendations": self.get_model_recommendations(),
            "forecast": self.get_forecast(steps=10),
            "feature_summary": {
                "features_available": list(self.create_time_features().columns),
                "total_features": len(self.create_time_features().columns)
            }
        }