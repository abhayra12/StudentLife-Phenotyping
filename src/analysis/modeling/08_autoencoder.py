"""
Modeling Analysis: Behavioral Autoencoder (Task 7.1)

Goal: Unsupervised learning of behavioral patterns.
1. dimensionality Reduction: Compress behavior into 3 latent dimensions.
2. Anomaly Detection: Identify days that don't fit the "normal" patterns (high reconstruction error).

Model: PyTorch Autoencoder (Linear).
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
            nn.Linear(8, latent_dim) # bottleneck
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
    """Load and scale data (no time shift needed for unsupervised)."""
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

    # Features only
    exclude = ['timestamp', 'participant_id', 'date', 'target_reg', 'target_clf']
    features = [c for c in train_df.columns if train_df[c].dtype in ['float64', 'int64'] and c not in exclude]
    
    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[features].fillna(0))
    X_val = scaler.transform(val_df[features].fillna(0))
    
    return X_train, X_val, features, train_df, val_df

def train_autoencoder():
    print(f"--- Task 7.1: Autoencoder Analysis (Device: {DEVICE}) ---")
    
    X_train, X_val, feature_names, df_train, df_val = load_data()
    input_dim = X_train.shape[1]
    
    # DataLoaders
    train_ds = TensorDataset(torch.FloatTensor(X_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model
    model = Autoencoder(input_dim, LATENT_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    # Train
    print("Training Autoencoder...")
    history = []
    
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
            
        # Validation
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
        history.append((avg_train, avg_val))
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Train MSE: {avg_train:.4f} | Val MSE: {avg_val:.4f}")
            
    # Analyze Reconstruction Distribution
    model.eval()
    all_x = torch.FloatTensor(X_val).to(DEVICE)
    with torch.no_grad():
        recon_val, latent_val = model(all_x)
        # MSE per sample
        mse_per_sample = torch.mean((all_x - recon_val)**2, dim=1).cpu().numpy()
        latent_val = latent_val.cpu().numpy()
        
    # Find Anomalies (Top 5% error)
    threshold = np.percentile(mse_per_sample, 95)
    print(f"\nAnomaly Threshold (95th percentile MSE): {threshold:.4f}")
    
    df_val['reconstruction_error'] = mse_per_sample
    df_val['is_anomaly'] = df_val['reconstruction_error'] > threshold
    
    anomalies = df_val[df_val['is_anomaly']]
    print(f"Found {len(anomalies)} anomalous days out of {len(df_val)}")
    
    # Save Results
    out_dir = Path('reports/figures/modeling')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save Model Weights
    models_dir = Path('models')
    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), models_dir / 'autoencoder.pth')
    print(f"Model saved to {models_dir / 'autoencoder.pth'}")
    
    # 1. Error Distribution Plot
    plt.figure(figsize=(10, 6))
    plt.hist(mse_per_sample, bins=50, alpha=0.7, color='blue', label='Normal')
    plt.axvline(threshold, color='red', linestyle='--', label='Anomaly Threshold')
    plt.title("Reconstruction Error Distribution")
    plt.xlabel("MSE Loss")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig(out_dir / 'reconstruction_error.png')
    
    # 2. Latent Space Plot (3D)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Color by activity level if available
    colors = df_val['activity_active_minutes'] if 'activity_active_minutes' in df_val else 'blue'
    sc = ax.scatter(latent_val[:, 0], latent_val[:, 1], latent_val[:, 2], c=colors, cmap='viridis', s=5, alpha=0.6)
    plt.colorbar(sc, label='Activity Minutes')
    ax.set_title("Latent Space (Compressed Behavior)")
    ax.set_xlabel("Latent 1")
    ax.set_ylabel("Latent 2")
    ax.set_zlabel("Latent 3")
    plt.savefig(out_dir / 'latent_space.png')
    
    # Save CSV
    res_dir = Path('reports/results')
    res_dir.mkdir(parents=True, exist_ok=True)
    anomalies.to_csv(res_dir / 'anomalies.csv') # Save full info for manual inspection
    
    print("Analysis Complete. Artifacts saved.")

if __name__ == "__main__":
    try:
        train_autoencoder()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
