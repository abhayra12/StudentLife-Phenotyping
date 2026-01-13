"""
Modeling Analysis: Feature Importance with SHAP (Task 6.2)

Goal: Interpret the best XGBoost Regressor model using SHAP.
Model: XGBoost (Regression). Metric: MAE 1.66
Params: Loaded from Task 6.1 results.

Analysis:
1. Global Importance (Beeswarm Plot)
2. Feature Dependence (Scatter Plots)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import xgboost as xgb
import shap
import ast

# Add project root to path
sys.path.append('.')
from src.features.temporal_features import process_temporal_features
from src.features.activity_sleep import process_activity_sleep
from src.features.location_features import process_location_features

def load_data():
    """Load train data for training and interpretation."""
    train_path = Path('data/processed/train.parquet')
    df = pd.read_parquet(train_path)
    
    # Ensure index
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.sort_values(['participant_id', 'timestamp'], inplace=True)
        df.set_index('timestamp', drop=False, inplace=True)

    # Features
    process_temporal_features(df)
    process_activity_sleep(df)
    try: process_location_features(df)
    except: pass
    
    # Target (Shift -24h)
    df['target'] = df.groupby('participant_id')['activity_active_minutes'].shift(-24)
    df.dropna(subset=['target'], inplace=True)
    
    # Select Features
    exclude = ['target', 'timestamp', 'participant_id', 'date']
    features = [c for c in df.columns if df[c].dtype in ['float64', 'int64'] and c not in exclude]
    
    return df[features], df['target']

def load_best_params():
    """Load best parameters from Task 6.1."""
    res_path = Path('reports/results/xgboost_optimization.csv')
    if not res_path.exists():
        raise FileNotFoundError("Optimization results not found.")
        
    df = pd.read_csv(res_path)
    # Get Regression params
    row = df[df['Task'] == 'Regression'].iloc[0]
    params = ast.literal_eval(row['Params'])
    print(f"Loaded Best Regression Params: {params}")
    return params

def run_shap_analysis():
    print("--- Task 6.2: SHAP Analysis ---")
    
    # 1. Load Data
    X, y = load_data()
    X = X.fillna(0) # XGB handles NaNs but SHAP likes clean data often
    
    # 2. Train Model
    params = load_best_params()
    # Ensure objective is set (sometimes not in best_params if fixed)
    params['objective'] = 'reg:squarederror'
    
    print("Training XGBoost...")
    model = xgb.XGBRegressor(**params)
    model.fit(X, y)
    
    # 3. Compute SHAP
    print("Computing SHAP values...")
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    
    # 4. Plots
    out_dir = Path('reports/figures/modeling')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # A. Summary Plot (Beeswarm)
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.title("SHAP Summary (Global Importance)")
    plt.tight_layout()
    plt.savefig(out_dir / 'shap_summary.png')
    plt.close()
    print(f"Summary plot saved to {out_dir / 'shap_summary.png'}")
    
    # B. Bar Plot (Mean Abs SHAP)
    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title("Feature Importance (Mean |SHAP|)")
    plt.tight_layout()
    plt.savefig(out_dir / 'shap_importance_bar.png')
    plt.close()
    
    # C. Top 3 Dependence Plots
    # Find top features
    mean_shap = np.abs(shap_values.values).mean(axis=0)
    top_indices = np.argsort(mean_shap)[::-1][:3]
    top_features = X.columns[top_indices]
    
    print(f"Top 3 Features: {list(top_features)}")
    
    for feat in top_features:
        plt.figure()
        shap.dependence_plot(feat, shap_values.values, X, show=False)
        plt.title(f"SHAP Dependence: {feat}")
        plt.tight_layout()
        plt.savefig(out_dir / f'shap_dependence_{feat}.png')
        plt.close()
        print(f"Dependence plot for {feat} saved.")

if __name__ == "__main__":
    try:
        run_shap_analysis()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
