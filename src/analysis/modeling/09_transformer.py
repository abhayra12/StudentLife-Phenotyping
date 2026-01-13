"""
Modeling Analysis: Behavioral Transformer (Task 7.2)

Goal: Use Transformers (Self-Attention) to capture long-range dependencies in behavior.
Compared to LSTM: Parallelizable, better at "remembering" distant past events in the sequence.

Model: PyTorch Transformer Encoder.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import math
from pathlib import Path
import sys
from sklearn.preprocessing import StandardScaler

# Add project root to path
sys.path.append('.')
from src.features.temporal_features import process_temporal_features
from src.features.activity_sleep import process_activity_sleep
from src.features.location_features import process_location_features

# Config
SEQ_LEN = 24
PRED_HORIZON = 24
BATCH_SIZE = 64
EPOCHS = 30
LR = 0.0005 # Transformers need lower LR often
D_MODEL = 32 # Feature dimension for transformer (embedding size)
NHEAD = 4
NUM_LAYERS = 2
DROPOUT = 0.1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class BehaviorTransformer(nn.Module):
    def __init__(self, input_dim, d_model=32, nhead=4, num_layers=2, dropout=0.1):
        super(BehaviorTransformer, self).__init__()
        
        # Project input features to d_model size
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=64, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.decoder = nn.Linear(d_model, 1) # Regression output

    def forward(self, src):
        # src: [seq_len, batch_size, features] for default Transformer
        # But we usually have [batch, seq, feat]. We need to permute.
        src = self.embedding(src) # [batch, seq, d_model]
        src = src.permute(1, 0, 2) # [seq, batch, d_model]
        
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src) # [seq, batch, d_model]
        
        # Take the last time step
        output = output[-1, :, :] # [batch, d_model]
        output = self.decoder(output)
        return output.squeeze()

def create_sequences(df, features, target_col='target_reg', seq_len=24):
    """Same as LSTM logic."""
    X_all = []
    y_all = []
    for pid, group in df.groupby('participant_id'):
        data = group[features].values
        target = group[target_col].values
        if len(data) <= seq_len: continue
        for i in range(len(data) - seq_len):
            X_all.append(data[i : i+seq_len])
            y_all.append(target[i + seq_len - 1])
    return np.array(X_all), np.array(y_all)

def load_data():
    """Load, sort, scale, and sequence."""
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
        
        df['target_reg'] = df.groupby('participant_id')['activity_active_minutes'].shift(-PRED_HORIZON)
        df.dropna(subset=['target_reg'], inplace=True)
        datasets[name] = df

    train_df = datasets['train']
    val_df = datasets['val']
    
    exclude = ['target_reg', 'timestamp', 'participant_id', 'date', 'target_clf']
    features = [c for c in train_df.columns if train_df[c].dtype in ['float64', 'int64'] and c not in exclude]
    
    scaler = StandardScaler()
    train_df[features] = scaler.fit_transform(train_df[features].fillna(0))
    val_df[features] = scaler.transform(val_df[features].fillna(0))
    
    X_train, y_train = create_sequences(train_df, features)
    X_val, y_val = create_sequences(val_df, features)
    
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    
    return train_ds, val_ds, len(features)

def train_transformer():
    print(f"--- Task 7.2: Transformer Training (Device: {DEVICE}) ---")
    
    train_ds, val_ds, input_dim = load_data()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    model = BehaviorTransformer(input_dim=input_dim, d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.L1Loss() # MAE
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    print("Training Transformer...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item()
                
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train MAE: {avg_train:.4f} | Val MAE: {avg_val:.4f}")
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), 'models/transformer_best.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping.")
                break
                
    print(f"\nTransformer Best Val MAE: {best_val_loss:.4f}")
    
    # Compare with LSTM (Read from file if exists)
    try:
        with open('reports/results/lstm_results.txt', 'r') as f:
            print("\nvs " + f.read().strip())
    except: pass
    
    # Save results
    with open('reports/results/transformer_results.txt', 'w') as f:
        f.write(f"Transformer Best Val MAE: {best_val_loss:.4f}\n")

if __name__ == "__main__":
    try:
        train_transformer()
    except Exception as e:
        print(f"Error: {e}")
