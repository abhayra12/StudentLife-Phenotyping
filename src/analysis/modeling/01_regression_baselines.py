"""
Modeling Script: Regression Baselines (Task 5.1)

Goal: Predict next-day activity level (total active minutes) using current day's features.
Models: Mean Baseline, Persistence Baseline, Linear Regression, Ridge Regression.

Methodology:
1. Load aligned data (train.parquet).
2. Apply feature engineering (Temporal, Activity/Sleep, Location).
3. Create Target: 'activity_next_day' (shift -24 hours).
4. Split Train/Test (Chronological split to prevent data leakage).
5. Train and Evaluate Models.
6. Save results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import json
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# Add project root to path
sys.path.append('.')

from src.features.temporal_features import process_temporal_features
from src.features.activity_sleep import process_activity_sleep
from src.features.location_features import process_location_features

def load_and_prep_data():
    """Load data and apply feature engineering."""
    DATA_PATH = Path('data/processed/train.parquet')
    if not DATA_PATH.exists():
        print(f"Error: {DATA_PATH} not found.")
        return None

    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    
    # Ensure timestamp index
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        df = df.set_index('timestamp', drop=False)
        
    print(f"Initial shape: {df.shape}")

    # --- Feature Engineering ---
    print("Applying Feature Engineering...")
    df = process_temporal_features(df)
    df = process_activity_sleep(df)
    
    # Location features might fail if no GPS cols, handle gracefully
    try:
        df = process_location_features(df)
    except Exception as e:
        print(f"Warning: Location features skipped ({e})")

    # --- Create Target ---
    # Target: Predict *Total Active Minutes* for the *Next 24 Hours*
    # However, for hourly data, a simpler object might be "Active minutes in the SAME hour tomorrow"
    # Or "Next Day Total". Let's do "Next Hour Activity" for simplicity first, 
    # OR "Activity 24h later". Let's stick to "Activity 24h later" as a standardized regression task.
    # Actually, predicting "Total Daily Activity" is a daily task. 
    # Predicting "Hourly Activity" is an hourly task.
    # Let's predict 'activity_active_minutes' shifted by 24 rows (assuming hourly data).
    
    target_col = 'activity_active_minutes'
    if target_col not in df.columns:
        print(f"Error: Target column {target_col} not found.")
        return None
        
    df['target'] = df.groupby('participant_id')[target_col].shift(-24)
    
    # Drop rows with NaN target (last 24h of each participant)
    df = df.dropna(subset=['target'])
    
    print(f"Shape after prep: {df.shape}")
    return df

def train_and_evaluate(df):
    """Train baseline models and evaluate."""
    # Features (Numeric only for baselines)
    features = [c for c in df.columns if df[c].dtype in ['float64', 'int64'] and c not in ['target', 'timestamp', 'participant_id']]
    
    # Handle NaNs in features (Simple imputation)
    df_model = df[features + ['target', 'participant_id', 'timestamp']].fillna(0)
    
    X = df_model[features]
    y = df_model['target']
    
    # Train/Test Split (Chronological)
    # Use last 20% of time for testing
    split_idx = int(len(df_model) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Train/Test sizes: {len(X_train)}/{len(X_test)}")
    
    results = []
    predictions = {}
    
    # --- 1. Mean Baseline ---
    y_pred_mean = np.full(len(y_test), y_train.mean())
    results.append(evaluate_model("Mean Baseline", y_test, y_pred_mean))
    predictions['Mean'] = y_pred_mean
    
    # --- 2. Persistence Baseline (Yesterday's value) ---
    # We constructed target as "Tomorrow's value", so "Today's value" is 'activity_active_minutes'
    # which is in X_test['activity_active_minutes'] if it exists, or we need to find it.
    if 'activity_active_minutes' in X_test.columns:
        y_pred_persist = X_test['activity_active_minutes']
        results.append(evaluate_model("Persistence", y_test, y_pred_persist))
        predictions['Persistence'] = y_pred_persist
    
    # --- 3. Linear Regression ---
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    results.append(evaluate_model("Linear Regression", y_test, y_pred_lr))
    predictions['Linear'] = y_pred_lr
    
    # --- 4. Ridge Regression ---
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    results.append(evaluate_model("Ridge Regression", y_test, y_pred_ridge))
    predictions['Ridge'] = y_pred_ridge
    
    # Save Results
    results_df = pd.DataFrame(results)
    OUTPUT_DIR = Path('reports/results')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_DIR / 'regression_baselines.csv', index=False)
    
    print("\n--- Results ---")
    print(results_df.to_string(index=False))
    
    return results_df, predictions, y_test

def evaluate_model(name, y_true, y_pred):
    return {
        'Model': name,
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred)
    }

def main():
    print("--- Task 5.1: Regression Baselines ---")
    df = load_and_prep_data()
    if df is not None:
        train_and_evaluate(df)

if __name__ == "__main__":
    main()
