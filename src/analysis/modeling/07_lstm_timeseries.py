"""
Modeling Analysis: LSTM Timeseries (Task 6.4)

Goal: Use Deep Learning (LSTM) to capture temporal dependencies.
Input: Sequence of past 24 hours features.
Target: Activity level 24 hours into the future (matching XGBoost target).

Baseline (XGBoost): MAE ~1.66
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import sys
import copy
from sklearn.preprocessing import StandardScaler

# Add project root to path
sys.path.append('.')
from src.features.temporal_features import process_temporal_features
from src.features.activity_sleep import process_activity_sleep
from src.features.location_features import process_location_features

# Configuration
SEQ_LEN = 24  # Use past 24 hours history
PRED_HORIZON = 24 # Predict 24 hours ahead
BATCH_SIZE = 64
EPOCHS = 30
LR = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ActivityLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(ActivityLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1) # Regression output
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        out, _ = self.lstm(x)
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out.squeeze()

def create_sequences(df, features, target_col='target_reg', seq_len=24):
    """
    Create (samples, seq_len, features) tensor.
    Assumes df is already sorted by participant and time.
    We must group by participant to avoid bleeding across users.
    """
    X_all = []
    y_all = []
    
    # Group by participant
    for pid, group in df.groupby('participant_id'):
        data = group[features].values
        target = group[target_col].values
        
        # Determine number of valid sequences
        # We need seq_len rows for input.
        # The target at index i corresponds to the row i.
        # But wait, our target_col is ALREADY shifted by -24 in the dataframe prep step?
        # If df['target_reg'] contains the value at t+24...
        # Then for a sequence ending at index i (using rows i-seq_len+1 to i),
        # we want to predict target_reg at index i.
        
        if len(data) <= seq_len:
            continue
            
        # Sliding window
        # Input: rows j to j+seq_len
        # Target: target_reg at j+seq_len-1 (the last row of the input sequence contains the target info for that row)
        # Wait, if target is aligned to the row, then yes.
        
        # Vectorized approach or list comprehension
        # (This can be slow for valid loops, but safe)
        for i in range(len(data) - seq_len):
            X_all.append(data[i : i+seq_len])
            y_all.append(target[i + seq_len - 1])
            
    return np.array(X_all), np.array(y_all)

def load_and_prep_data():
    """Load and preprocess data for LSTM."""
    train_path = Path('data/processed/train.parquet')
    val_path = Path('data/processed/val.parquet')
    
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    
    # 1. Feature Engineering & Sorting
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
        
        # Create Target (Same as XGBoost)
        # Target is value 24 hours later
        df['target_reg'] = df.groupby('participant_id')['activity_active_minutes'].shift(-PRED_HORIZON)
        df.dropna(subset=['target_reg'], inplace=True) # sequence gen handles end padding implicitly
        
        datasets[name] = df
        
    train_df = datasets['train']
    val_df = datasets['val']
    
    # 2. scaling
    # Neural Networks require scaling!
    exclude = ['target_reg', 'timestamp', 'participant_id', 'date', 'target_clf']
    features = [c for c in train_df.columns if train_df[c].dtype in ['float64', 'int64'] and c not in exclude]
    
    scaler = StandardScaler()
    train_df[features] = scaler.fit_transform(train_df[features].fillna(0))
    val_df[features] = scaler.transform(val_df[features].fillna(0))
    
    print(f"Generating sequences (Len={SEQ_LEN})... this may take a moment.")
    X_train, y_train = create_sequences(train_df, features)
    X_val, y_val = create_sequences(val_df, features)
    
    print(f"Train Seq: {X_train.shape}, Val Seq: {X_val.shape}")
    
    # Convert to Tensor
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    
    return train_dataset, val_dataset, len(features)

def train_model():
    print(f"--- Task 6.4: LSTM Training (Device: {DEVICE}) ---")
    
    train_ds, val_ds, input_dim = load_and_prep_data()
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    model = ActivityLSTM(input_size=input_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.L1Loss() # MAE Loss
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        model.train()
        train_losses = []
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                val_losses.append(loss.item())
                
        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train MAE: {avg_train:.4f} | Val MAE: {avg_val:.4f}")
        
        # Checkpoint
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), 'models/lstm_best.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break
            
    print(f"\nTraining Complete. Best Validation MAE: {best_val_loss:.4f}")
    
    # Save text result
    Path('reports/results').mkdir(parents=True, exist_ok=True)
    with open('reports/results/lstm_results.txt', 'w') as f:
        f.write(f"LSTM Best Val MAE: {best_val_loss:.4f}\n")
        f.write(f"Baseline XGBoost MAE: 1.66\n")

if __name__ == "__main__":
    try:
        Path('models').mkdir(exist_ok=True)
        train_model()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
