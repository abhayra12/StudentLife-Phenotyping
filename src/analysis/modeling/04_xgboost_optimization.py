"""
Modeling Script: XGBoost Optimization with Optuna (Task 6.1)

Goal: Find optimal hyperparameters for XGBoost models.
Tasks:
1. Regression (Predict next-day activity minutes). Metric: MAE.
2. Classification (Predict high/low activity). Metric: ROC-AUC.

Output: Best parameters and metrics saved to reports/results.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import xgboost as xgb
import optuna
from sklearn.metrics import mean_absolute_error, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler

# Add project root to path
sys.path.append('.')
from src.features.temporal_features import process_temporal_features
from src.features.activity_sleep import process_activity_sleep
from src.features.location_features import process_location_features

# Suppress Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

def load_and_prep_data():
    """Load train/val and prepare targets."""
    train_path = Path('data/processed/train.parquet')
    val_path = Path('data/processed/val.parquet')
    
    if not train_path.exists():
        raise FileNotFoundError("Train data not found.")
        
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    
    datasets = {}
    
    # Process both
    for name, df in [('train', train_df), ('val', val_df)]:
        # Ensure timestamp index FIRST
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.sort_values(['participant_id', 'timestamp'], inplace=True)
            df.set_index('timestamp', drop=False, inplace=True)

        # Features
        process_temporal_features(df)
        process_activity_sleep(df)
        try: process_location_features(df)
        except: pass
        
        # Target Creation (Shift -24h)
        df['target_reg'] = df.groupby('participant_id')['activity_active_minutes'].shift(-24)
        
        # Drop NaNs
        df.dropna(subset=['target_reg'], inplace=True)
        
        datasets[name] = df
        
    train_df = datasets['train']
    val_df = datasets['val']
    
    # Binary Target (Threshold from Train Mean)
    threshold = train_df['target_reg'].mean()
    print(f"Binary Threshold (Mean Activity): {threshold:.2f}")
    
    train_df['target_clf'] = (train_df['target_reg'] > threshold).astype(int)
    val_df['target_clf'] = (val_df['target_reg'] > threshold).astype(int)
    
    # Select Features (Numeric)
    exclude = ['target_reg', 'target_clf', 'timestamp', 'participant_id', 'date']
    features = [c for c in train_df.columns if train_df[c].dtype in ['float64', 'int64'] and c not in exclude]
    
    X_train = train_df[features].fillna(0)
    y_train_reg = train_df['target_reg']
    y_train_clf = train_df['target_clf']
    
    X_val = val_df[features].fillna(0)
    y_val_reg = val_df['target_reg']
    y_val_clf = val_df['target_clf']
    
    print(f"Data Loaded. Train: {X_train.shape}, Val: {X_val.shape}")
    print(f"Features: {len(features)}")
    
    return X_train, y_train_reg, y_train_clf, X_val, y_val_reg, y_val_clf

def optimize_regression(X_train, y_train, X_val, y_val, n_trials=20):
    print("\n--- Optimizing XGBoost Regressor (Metric: MAE) ---")
    
    def objective(trial):
        params = {
            'objective': 'reg:squarederror',
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'random_state': 42,
            'n_jobs': -1
        }
        
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False) # Removed early_stopping_rounds for compatibility/simplicity check
        
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        return mae

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    print(f"Best MAE: {study.best_value:.4f}")
    print(f"Best Params: {study.best_params}")
    return study.best_params, study.best_value

def optimize_classification(X_train, y_train, X_val, y_val, n_trials=20):
    print("\n--- Optimizing XGBoost Classifier (Metric: AUC) ---")
    
    def objective(trial):
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 5), # Handle imbalance
            'random_state': 42,
            'n_jobs': -1
        }
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        preds_proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, preds_proba)
        return auc

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    print(f"Best AUC: {study.best_value:.4f}")
    print(f"Best Params: {study.best_params}")
    return study.best_params, study.best_value

def main():
    try:
        data = load_and_prep_data()
        X_train, y_train_reg, y_train_clf, X_val, y_val_reg, y_val_clf = data
        
        # 1. Regression
        best_params_reg, best_mae = optimize_regression(X_train, y_train_reg, X_val, y_val_reg)
        
        # 2. Classification
        best_params_clf, best_auc = optimize_classification(X_train, y_train_clf, X_val, y_val_clf)
        
        # Save Results
        res = [
            {'Task': 'Regression', 'Best_Metric': 'MAE', 'Score': best_mae, 'Params': str(best_params_reg)},
            {'Task': 'Classification', 'Best_Metric': 'AUC', 'Score': best_auc, 'Params': str(best_params_clf)}
        ]
        
        out_dir = Path('reports/results')
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(res).to_csv(out_dir / 'xgboost_optimization.csv', index=False)
        print(f"\nOptimization results saved to reports/results/xgboost_optimization.csv")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
