import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

class DataVisualizer:
    def __init__(self, df):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    def create_distribution_plots(self):
        plots = {}
        for col in self.numeric_cols[:3]:
            fig = px.histogram(self.df, x=col, title=f"Distribution of {col}", nbins=30)
            plots[f"histogram_{col}"] = fig.to_json()
        
        if self.numeric_cols:
            fig = go.Figure()
            for col in self.numeric_cols[:4]:
                fig.add_trace(go.Box(y=self.df[col].dropna(), name=col))
            fig.update_layout(title="Box Plots", height=400)
            plots["boxplots"] = fig.to_json()
        
        return plots

    def create_correlation_heatmap(self):
        if len(self.numeric_cols) >= 2:
            corr_matrix = self.df[self.numeric_cols].corr()
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmin=-1, zmax=1
            ))
            fig.update_layout(title="Correlation Heatmap", height=500)
            return fig.to_json()
        return None

    def create_missing_plot(self):
        missing_df = self.df.isnull().sum().reset_index()
        missing_df.columns = ['Column', 'Missing Count']
        missing_df = missing_df[missing_df['Missing Count'] > 0]
        
        if len(missing_df) > 0:
            fig = px.bar(missing_df, x='Column', y='Missing Count', title="Missing Values")
            return fig.to_json()
        return None