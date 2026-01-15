"""
Modeling Analysis: Dual Autoencoder Approach (Comparison Experiment)

Goal: Train SEPARATE autoencoders for weekday and weekend data.
This is an experimental comparison against the single-model approach.

Model: Two PyTorch Autoencoders (one for weekdays, one for weekends).
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from sklearn.preprocessing import StandardScaler

# Add project root to path
sys.path.append('.')
from src.features.temporal_features import process_temporal_features
from src.features.activity_sleep import process_activity_sleep
from src.features.location_features import process_location_features

# Config
LATENT_DIM = 3
EPOCHS = 50
BATCH_SIZE = 64
LR = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=3):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )
        
    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent

def load_data():
    """Load and split data by weekday/weekend."""
    train_path = Path('data/processed/train.parquet')
    val_path = Path('data/processed/val.parquet')
    
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    
    # Process features
    for df in [train_df, val_df]:
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.sort_values(['participant_id', 'timestamp'], inplace=True)
            df.set_index('timestamp', drop=False, inplace=True)
            
        process_temporal_features(df)
        process_activity_sleep(df)
        try: process_location_features(df)
        except: pass
        
        # Add day of week
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # Features only
    exclude = ['timestamp', 'participant_id', 'date', 'target_reg', 'target_clf', 'day_of_week', 'is_weekend']
    features = [c for c in train_df.columns if train_df[c].dtype in ['float64', 'int64'] and c not in exclude]
    
    # Split by day type
    train_weekday = train_df[train_df['is_weekend'] == 0]
    train_weekend = train_df[train_df['is_weekend'] == 1]
    val_weekday = val_df[val_df['is_weekend'] == 0]
    val_weekend = val_df[val_df['is_weekend'] == 1]
    
    # Scale separately
    scaler_weekday = StandardScaler()
    scaler_weekend = StandardScaler()
    
    X_train_weekday = scaler_weekday.fit_transform(train_weekday[features].fillna(0))
    X_val_weekday = scaler_weekday.transform(val_weekday[features].fillna(0))
    
    X_train_weekend = scaler_weekend.fit_transform(train_weekend[features].fillna(0))
    X_val_weekend = scaler_weekend.transform(val_weekend[features].fillna(0))
    
    return {
        'weekday': (X_train_weekday, X_val_weekday, val_weekday),
        'weekend': (X_train_weekend, X_val_weekend, val_weekend),
        'features': features,
        'val_df': val_df
    }

def train_model(X_train, X_val, name):
    """Train a single autoencoder."""
    input_dim = X_train.shape[1]
    
    train_ds = TensorDataset(torch.FloatTensor(X_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    model = Autoencoder(input_dim, LATENT_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    print(f"\nTraining {name} Autoencoder...")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch in train_loader:
            x = batch[0].to(DEVICE)
            optimizer.zero_grad()
            recon, _ = model(x)
            loss = criterion(recon, x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(DEVICE)
                recon, _ = model(x)
                loss = criterion(recon, x)
                val_loss += loss.item()
                
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        
        if (epoch+1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} | Train MSE: {avg_train:.4f} | Val MSE: {avg_val:.4f}")
    
    return model

def train_dual_autoencoders():
    print("=" * 80)
    print("DUAL AUTOENCODER APPROACH (Separate Models for Weekday/Weekend)")
    print("=" * 80)
    
    data = load_data()
    
    # Train weekday model
    weekday_model = train_model(data['weekday'][0], data['weekday'][1], "WEEKDAY")
    
    # Train weekend model
    weekend_model = train_model(data['weekend'][0], data['weekend'][1], "WEEKEND")
    
    # Evaluate both models
    print("\n" + "=" * 80)
    print("EVALUATION")
    print("=" * 80)
    
    # Weekday evaluation
    weekday_model.eval()
    X_val_weekday = torch.FloatTensor(data['weekday'][1]).to(DEVICE)
    with torch.no_grad():
        recon_weekday, _ = weekday_model(X_val_weekday)
        weekday_errors = torch.mean((X_val_weekday - recon_weekday)**2, dim=1).cpu().numpy()
    
    # Weekend evaluation
    weekend_model.eval()
    X_val_weekend = torch.FloatTensor(data['weekend'][1]).to(DEVICE)
    with torch.no_grad():
        recon_weekend, _ = weekend_model(X_val_weekend)
        weekend_errors = torch.mean((X_val_weekend - recon_weekend)**2, dim=1).cpu().numpy()
    
    # Compute thresholds
    weekday_threshold = np.percentile(weekday_errors, 95)
    weekend_threshold = np.percentile(weekend_errors, 95)
    
    print(f"\nWeekday Threshold (95th percentile): {weekday_threshold:.4f}")
    print(f"Weekend Threshold (95th percentile): {weekend_threshold:.4f}")
    
    # Flag anomalies
    data['weekday'][2]['reconstruction_error'] = weekday_errors
    data['weekday'][2]['is_anomaly'] = weekday_errors > weekday_threshold
    
    data['weekend'][2]['reconstruction_error'] = weekend_errors
    data['weekend'][2]['is_anomaly'] = weekend_errors > weekend_threshold
    
    # Merge results
    val_df = data['val_df'].copy()
    val_df.loc[val_df['is_weekend'] == 0, 'reconstruction_error'] = weekday_errors
    val_df.loc[val_df['is_weekend'] == 1, 'reconstruction_error'] = weekend_errors
    val_df.loc[val_df['is_weekend'] == 0, 'threshold'] = weekday_threshold
    val_df.loc[val_df['is_weekend'] == 1, 'threshold'] = weekend_threshold
    val_df['is_anomaly'] = val_df['reconstruction_error'] > val_df['threshold']
    
    # Statistics
    total_anomalies = val_df['is_anomaly'].sum()
    weekday_anomalies = data['weekday'][2]['is_anomaly'].sum()
    weekend_anomalies = data['weekend'][2]['is_anomaly'].sum()
    
    print(f"\n=== Anomaly Detection Results (Dual Model) ===")
    print(f"Total Anomalies: {total_anomalies} / {len(val_df)} ({100*total_anomalies/len(val_df):.1f}%)")
    print(f"  Weekday Anomalies: {weekday_anomalies} / {len(weekday_errors)} ({100*weekday_anomalies/len(weekday_errors):.1f}%)")
    print(f"  Weekend Anomalies: {weekend_anomalies} / {len(weekend_errors)} ({100*weekend_anomalies/len(weekend_errors):.1f}%)")
    
    # Save models
    models_dir = Path('models')
    models_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(weekday_model.state_dict(), models_dir / 'autoencoder_weekday.pth')
    torch.save(weekend_model.state_dict(), models_dir / 'autoencoder_weekend.pth')
    print(f"\nModels saved:")
    print(f"  - {models_dir / 'autoencoder_weekday.pth'}")
    print(f"  - {models_dir / 'autoencoder_weekend.pth'}")
    
    # Save thresholds
    import json
    thresholds = {
        'approach': 'dual_model',
        'weekday_threshold': float(weekday_threshold),
        'weekend_threshold': float(weekend_threshold)
    }
    with open(models_dir / 'anomaly_thresholds_dual.json', 'w') as f:
        json.dump(thresholds, f, indent=2)
    
    # Save results
    res_dir = Path('reports/results')
    res_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        'approach': 'dual_model',
        'total_samples': len(val_df),
        'weekday_samples': int(len(weekday_errors)),
        'weekend_samples': int(len(weekend_errors)),
        'total_anomalies': int(total_anomalies),
        'weekday_anomalies': int(weekday_anomalies),
        'weekend_anomalies': int(weekend_anomalies),
        'weekday_threshold': float(weekday_threshold),
        'weekend_threshold': float(weekend_threshold)
    }
    
    with open(res_dir / 'anomaly_summary_dual.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Dual Model Training Complete!")
    print(f"   - Results saved to {res_dir}/anomaly_summary_dual.json")
    
    return summary

if __name__ == "__main__":
    try:
        train_dual_autoencoders()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
