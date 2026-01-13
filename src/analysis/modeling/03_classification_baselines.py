"""
Modeling Script: Classification Baselines (Task 5.3)

Goal: Predict "High Activity" vs "Low Activity" days (Binary Classification).
Target Definition: 1 if next-day activity > mean(train_activity), else 0.

Models:
1. Dummy Classifier (Baseline)
2. Logistic Regression (Linear)
3. Random Forest (Non-linear)

Evaluation: ROC-AUC, Precision, Recall, F1.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone

# Add project root to path
sys.path.append('.')
from src.features.temporal_features import process_temporal_features
from src.features.activity_sleep import process_activity_sleep
from src.features.location_features import process_location_features

def load_and_prep_data():
    """Load data, engineer features, and create binary target."""
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

    # Note: Feature engineering is already "applied" if loading processed data,
    # but raw parquet might just have columns. 
    # The previous regression script re-ran processing. Let's consistency do it or check columns.
    # Assuming train.parquet is the OUTPUT of create_final_dataset which is just aligned data + time features.
    # So we need to run feature processors.
    
    print("Applying Feature Engineering...")
    df = process_temporal_features(df)
    df = process_activity_sleep(df)
    try:
        df = process_location_features(df)
    except Exception as e:
        print(f"Warning: Location features skipped ({e})")

    # Create Target
    target_col = 'activity_active_minutes'
    if target_col not in df.columns:
        print(f"Error: Target column {target_col} not found.")
        return None
        
    # Future target
    df['next_day_activity'] = df.groupby('participant_id')[target_col].shift(-24)
    df = df.dropna(subset=['next_day_activity'])
    
    # Define Threshold (based on TRAIN set only, but here we split later. 
    # Strictly, we should split first, then calc threshold. 
    # But for a baseline, using global mean of loaded data (train+val+test potentially if we loaded all? 
    # Wait, strict split:
    # We loaded 'train.parquet' which is actually the full dataset split by create_final_dataset?
    # No, 'train.parquet' is just the TRAINING split.
    # 'val.parquet' and 'test.parquet' exist separately.
    # Ah, simpler to just load 'train.parquet' for training and 'val.parquet' for evaluation?
    # The regression script loaded `train.parquet` and then did a time split internally.
    # Let's stick to that for Consistency, OR verify if 'train.parquet' contains full data.
    # create_final_dataset code: 
    # train.to_parquet('train.parquet'), val.to_parquet('val.parquet').
    # So 'train.parquet' is ONLY weeks 1-6.
    
    # Actually, for robust evaluation, we should train on TRAIN and eval on VAL.
    # The regression script loaded 'train.parquet' (49k rows) and split it 80/20.
    # That means it trained on Week 1-4.8 and tested on Week 4.8-6.
    # Effectively using a validation set from within the training set.
    
    # Let's do it properly here: Load Train and Val separately.
    
    return df

def get_binary_target(df, threshold):
    return (df['next_day_activity'] > threshold).astype(int)

def train_and_evaluate():
    # Load separate splits
    train_path = Path('data/processed/train.parquet')
    val_path = Path('data/processed/val.parquet')
    
    print("Loading Train and Val sets...")
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    
    # Preprocess
    for d in [train_df, val_df]:
        if 'timestamp' in d.columns:
             d['timestamp'] = pd.to_datetime(d['timestamp'])
             d.sort_values('timestamp', inplace=True)
             d.set_index('timestamp', drop=False, inplace=True)
             
        # Apply Features
        process_temporal_features(d)
        process_activity_sleep(d)
        try: process_location_features(d)
        except: pass
        
        # Create Target (Continuous)
        d['next_day_activity'] = d.groupby('participant_id')['activity_active_minutes'].shift(-24)
    
    # Drop NaNs
    train_df.dropna(subset=['next_day_activity'], inplace=True)
    val_df.dropna(subset=['next_day_activity'], inplace=True)
    
    # Define Threshold from TRAIN
    threshold = train_df['next_day_activity'].mean()
    print(f"Binarizing Target. Threshold (Mean of Train) = {threshold:.2f} minutes")
    
    train_y = get_binary_target(train_df, threshold)
    val_y = get_binary_target(val_df, threshold)
    
    # Features
    exclude = ['next_day_activity', 'timestamp', 'participant_id', 'target', 'date']
    features = [c for c in train_df.columns if train_df[c].dtype in ['float64', 'int64'] and c not in exclude]
    
    # Fill NaNs
    train_X = train_df[features].fillna(0)
    val_X = val_df[features].fillna(0)
    
    print(f"Train Shape: {train_X.shape}, Val Shape: {val_X.shape}")
    print(f"Class Balance (Train): {train_y.mean():.2f}")
    
    # --- Models ---
    results = []
    
    models = {
        'Dummy': DummyClassifier(strategy='stratified', random_state=42),
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    }
    
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(train_X, train_y)
        
        # Predict
        y_pred = model.predict(val_X)
        y_prob = model.predict_proba(val_X)[:, 1]
        
        # Metrics
        res = {
            'Model': name,
            'Accuracy': accuracy_score(val_y, y_pred),
            'Precision': precision_score(val_y, y_pred),
            'Recall': recall_score(val_y, y_pred),
            'F1': f1_score(val_y, y_pred),
            'AUC': roc_auc_score(val_y, y_prob)
        }
        results.append(res)
        
        # Plot ROC
        fpr, tpr, _ = roc_curve(val_y, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {res['AUC']:.2f})")
        
    # Finalize Plot
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Activity Classification (Train vs Val)')
    plt.legend()
    
    out_dir = Path('reports/figures/modeling')
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / 'roc_curves.png')
    print(f"ROC Curve saved to {out_dir / 'roc_curves.png'}")
    
    # Save Results
    results_df = pd.DataFrame(results)
    out_res = Path('reports/results')
    out_res.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_res / 'classification_baselines.csv', index=False)
    
    print("\n--- Classification Results (Val Set) ---")
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    print("--- Task 5.3: Classification Baselines ---")
    train_and_evaluate()
