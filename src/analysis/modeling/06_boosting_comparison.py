"""
Modeling Analysis: Boosting Comparison (Task 6.3)

Goal: Benchmark LightGBM and CatBoost against XGBoost.
Metrics: 
- Regression: MAE
- Classification: AUC
- Training Time

Models:
- XGBoost (Baseline from Task 6.1)
- LightGBM (Gradient Boosting with leaf-wise growth)
- CatBoost (Ordered Boosting, good for defaults)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import time
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.metrics import mean_absolute_error, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler

# Add project root to path
sys.path.append('.')
from src.features.temporal_features import process_temporal_features
from src.features.activity_sleep import process_activity_sleep
from src.features.location_features import process_location_features

def load_data():
    """Load train/val data."""
    train_path = Path('data/processed/train.parquet')
    val_path = Path('data/processed/val.parquet')
    
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    
    datasets = {}
    for name, df in [('train', train_df), ('val', val_df)]:
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.sort_values(['participant_id', 'timestamp'], inplace=True)
            df.set_index('timestamp', drop=False, inplace=True)
            
        process_temporal_features(df)
        process_activity_sleep(df)
        try: process_location_features(df)
        except: pass
        
        # Target (Reg)
        df['target_reg'] = df.groupby('participant_id')['activity_active_minutes'].shift(-24)
        df.dropna(subset=['target_reg'], inplace=True)
        datasets[name] = df

    train_df = datasets['train']
    val_df = datasets['val']
    
    # Target (Clf)
    threshold = train_df['target_reg'].mean()
    train_df['target_clf'] = (train_df['target_reg'] > threshold).astype(int)
    val_df['target_clf'] = (val_df['target_reg'] > threshold).astype(int)
    
    # Features
    exclude = ['target_reg', 'target_clf', 'timestamp', 'participant_id', 'date']
    features = [c for c in train_df.columns if train_df[c].dtype in ['float64', 'int64'] and c not in exclude]
    
    return train_df, val_df, features

def train_evaluate_reg(model_name, model, X_train, y_train, X_val, y_val):
    print(f"Training {model_name} (Regression)...")
    start = time.time()
    try:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    except TypeError:
         model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
         
    train_time = time.time() - start
    
    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    return {'Model': model_name, 'Task': 'Regression', 'Metric': 'MAE', 'Score': mae, 'Time': train_time}

def train_evaluate_clf(model_name, model, X_train, y_train, X_val, y_val):
    print(f"Training {model_name} (Classification)...")
    start = time.time()
    try:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    except TypeError:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        
    train_time = time.time() - start
    
    probs = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, probs)
    return {'Model': model_name, 'Task': 'Classification', 'Metric': 'AUC', 'Score': auc, 'Time': train_time}

def main():
    print("--- Task 6.3: Boosting Comparison ---")
    
    train_df, val_df, feats = load_data()
    X_train = train_df[feats].fillna(0)
    X_val = val_df[feats].fillna(0)
    
    results = []
    
    # --- Regression ---
    y_train_r, y_val_r = train_df['target_reg'], val_df['target_reg']
    
    # 1. XGBoost
    xgb_reg = xgb.XGBRegressor(n_estimators=149, learning_rate=0.06, max_depth=8, 
                               random_state=42, n_jobs=-1, objective='reg:squarederror',
                               early_stopping_rounds=20)
    results.append(train_evaluate_reg('XGBoost', xgb_reg, X_train, y_train_r, X_val, y_val_r))
    
    # 2. LightGBM
    lgb_reg = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31, 
                                random_state=42, n_jobs=-1)
    results.append(train_evaluate_reg('LightGBM', lgb_reg, X_train, y_train_r, X_val, y_val_r))
    
    # 3. CatBoost (Verbose=0 works for fit kwarg, but fit arg varies slightly)
    # CatBoost generic wrapper
    cb_reg = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, 
                               loss_function='MAE', verbose=0, random_seed=42)
    # CatBoost handles eval_set internally differently in fit standard call? 
    # Yes, it works.
    print(f"Training CatBoost (Regression)...")
    start = time.time()
    cb_reg.fit(X_train, y_train_r, eval_set=(X_val, y_val_r), early_stopping_rounds=20, verbose=False)
    results.append({'Model': 'CatBoost', 'Task': 'Regression', 'Metric': 'MAE', 
                    'Score': mean_absolute_error(y_val_r, cb_reg.predict(X_val)), 'Time': time.time() - start})
    
    # --- Classification ---
    y_train_c, y_val_c = train_df['target_clf'], val_df['target_clf']
    
    # 1. XGBoost
    xgb_clf = xgb.XGBClassifier(n_estimators=272, learning_rate=0.02, max_depth=4, 
                                scale_pos_weight=2.6, random_state=42, n_jobs=-1, eval_metric='auc',
                                early_stopping_rounds=20)
    results.append(train_evaluate_clf('XGBoost', xgb_clf, X_train, y_train_c, X_val, y_val_c))
    
    # 2. LightGBM
    lgb_clf = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.02, num_leaves=31, 
                                 random_state=42, n_jobs=-1)
    results.append(train_evaluate_clf('LightGBM', lgb_clf, X_train, y_train_c, X_val, y_val_c))
    
    # 3. CatBoost
    cb_clf = CatBoostClassifier(iterations=500, learning_rate=0.02, depth=6, 
                                eval_metric='AUC', verbose=0, random_seed=42)
    print(f"Training CatBoost (Classification)...")
    start = time.time()
    cb_clf.fit(X_train, y_train_c, eval_set=(X_val, y_val_c), early_stopping_rounds=20, verbose=False)
    results.append({'Model': 'CatBoost', 'Task': 'Classification', 'Metric': 'AUC', 
                    'Score': roc_auc_score(y_val_c, cb_clf.predict_proba(X_val)[:, 1]), 'Time': time.time() - start})
    
    # Save Results
    res_df = pd.DataFrame(results)
    
    out_dir = Path('reports/results')
    res_df.to_csv(out_dir / 'boosting_comparison.csv', index=False)
    print("\n--- Results ---")
    print(res_df.to_string(index=False))
    
    # Plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.barplot(data=res_df[res_df['Task']=='Regression'], x='Model', y='Score', hue='Model')
    plt.title('Regression MAE (Lower is Better)')
    plt.ylabel('MAE')
    
    plt.subplot(1, 2, 2)
    sns.barplot(data=res_df[res_df['Task']=='Classification'], x='Model', y='Score', hue='Model')
    plt.title('Classification AUC (Higher is Better)')
    plt.ylabel('AUC')
    
    plt.tight_layout()
    plt.savefig('reports/figures/modeling/model_comparison_bar.png')
    print("Comparison plot saved.")

if __name__ == "__main__":
    main()
