"""
Modeling Analysis: Behavioral Autoencoder (Task 7.1)

Goal: Unsupervised learning of behavioral patterns.
1. Dimensionality Reduction: Compress behavior into 3 latent dimensions.
2. Anomaly Detection: Identify days that don't fit the "normal" patterns (high reconstruction error).
3. Weekend Normalization: Use separate thresholds for weekdays vs weekends to reduce false positives.

Model: PyTorch Autoencoder (Linear) with Day-of-Week Aware Anomaly Detection.
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
        
        # Add day of week (0=Monday, 6=Sunday)
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)  # Saturday=5, Sunday=6

    # Features only
    exclude = ['timestamp', 'participant_id', 'date', 'target_reg', 'target_clf', 'day_of_week', 'is_weekend']
    features = [c for c in train_df.columns if train_df[c].dtype in ['float64', 'int64'] and c not in exclude]
    
    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[features].fillna(0))
    X_val = scaler.transform(val_df[features].fillna(0))
    
    return X_train, X_val, features, train_df, val_df

def train_autoencoder():
    print(f"--- Task 7.1: Autoencoder Analysis with Weekend Normalization (Device: {DEVICE}) ---")
    
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
        
    df_val['reconstruction_error'] = mse_per_sample
    
    # ==================== WEEKEND NORMALIZATION ====================
    print("\n=== Weekend Normalization Analysis ===")
    
    # Separate weekday and weekend data
    weekday_errors = df_val[df_val['is_weekend'] == 0]['reconstruction_error']
    weekend_errors = df_val[df_val['is_weekend'] == 1]['reconstruction_error']
    
    # Compute separate thresholds (95th percentile)
    weekday_threshold = np.percentile(weekday_errors, 95)
    weekend_threshold = np.percentile(weekend_errors, 95)
    
    print(f"Weekday (Mon-Fri) Threshold: {weekday_threshold:.4f}")
    print(f"Weekend (Sat-Sun) Threshold: {weekend_threshold:.4f}")
    print(f"Threshold Ratio (Weekend/Weekday): {weekend_threshold/weekday_threshold:.2f}x")
    
    # Apply day-specific thresholds
    df_val['threshold'] = df_val['is_weekend'].map({0: weekday_threshold, 1: weekend_threshold})
    df_val['is_anomaly'] = df_val['reconstruction_error'] > df_val['threshold']
    
    # Statistics
    total_anomalies = df_val['is_anomaly'].sum()
    weekday_anomalies = df_val[(df_val['is_weekend'] == 0) & (df_val['is_anomaly'])].shape[0]
    weekend_anomalies = df_val[(df_val['is_weekend'] == 1) & (df_val['is_anomaly'])].shape[0]
    
    print(f"\n=== Anomaly Detection Results ===")
    print(f"Total Anomalies: {total_anomalies} / {len(df_val)} ({100*total_anomalies/len(df_val):.1f}%)")
    print(f"  Weekday Anomalies: {weekday_anomalies} / {len(weekday_errors)} ({100*weekday_anomalies/len(weekday_errors):.1f}%)")
    print(f"  Weekend Anomalies: {weekend_anomalies} / {len(weekend_errors)} ({100*weekend_anomalies/len(weekend_errors):.1f}%)")
    
    # Compare with naive (single threshold) approach
    naive_threshold = np.percentile(mse_per_sample, 95)
    df_val['is_anomaly_naive'] = df_val['reconstruction_error'] > naive_threshold
    naive_total = df_val['is_anomaly_naive'].sum()
    naive_weekend = df_val[(df_val['is_weekend'] == 1) & (df_val['is_anomaly_naive'])].shape[0]
    
    print(f"\n=== Comparison with Naive (Single Threshold) ===")
    print(f"Naive Total Anomalies: {naive_total}")
    print(f"Naive Weekend Anomalies: {naive_weekend} ({100*naive_weekend/len(weekend_errors):.1f}% of weekends)")
    print(f"Improvement: {naive_weekend - weekend_anomalies} fewer weekend false positives")
    
    # ===============================================================
    
    anomalies = df_val[df_val['is_anomaly']]
    
    # Save Results
    out_dir = Path('reports/figures/modeling')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save Model Weights
    models_dir = Path('models')
    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), models_dir / 'autoencoder.pth')
    print(f"\nModel saved to {models_dir / 'autoencoder.pth'}")
    
    # Save thresholds for inference
    thresholds = {
        'weekday_threshold': float(weekday_threshold),
        'weekend_threshold': float(weekend_threshold),
        'naive_threshold': float(naive_threshold)
    }
    import json
    with open(models_dir / 'anomaly_thresholds.json', 'w') as f:
        json.dump(thresholds, f, indent=2)
    print(f"Thresholds saved to {models_dir / 'anomaly_thresholds.json'}")
    
    # 1. Error Distribution Plot (with weekend split)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Weekday distribution
    axes[0].hist(weekday_errors, bins=50, alpha=0.7, color='blue', label='Weekday')
    axes[0].axvline(weekday_threshold, color='red', linestyle='--', linewidth=2, label=f'Weekday Threshold ({weekday_threshold:.4f})')
    axes[0].set_title("Weekday Reconstruction Error Distribution")
    axes[0].set_xlabel("MSE Loss")
    axes[0].set_ylabel("Count")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Weekend distribution
    axes[1].hist(weekend_errors, bins=50, alpha=0.7, color='green', label='Weekend')
    axes[1].axvline(weekend_threshold, color='red', linestyle='--', linewidth=2, label=f'Weekend Threshold ({weekend_threshold:.4f})')
    axes[1].set_title("Weekend Reconstruction Error Distribution")
    axes[1].set_xlabel("MSE Loss")
    axes[1].set_ylabel("Count")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'reconstruction_error_weekend_split.png', dpi=150)
    print(f"Saved: {out_dir / 'reconstruction_error_weekend_split.png'}")
    
    # 2. Combined Error Distribution (original plot)
    plt.figure(figsize=(10, 6))
    plt.hist(weekday_errors, bins=50, alpha=0.5, color='blue', label='Weekday')
    plt.hist(weekend_errors, bins=50, alpha=0.5, color='green', label='Weekend')
    plt.axvline(weekday_threshold, color='blue', linestyle='--', label='Weekday Threshold')
    plt.axvline(weekend_threshold, color='green', linestyle='--', label='Weekend Threshold')
    plt.axvline(naive_threshold, color='red', linestyle=':', linewidth=2, label='Naive (Single) Threshold')
    plt.title("Reconstruction Error Distribution: Weekday vs Weekend")
    plt.xlabel("MSE Loss")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(out_dir / 'reconstruction_error.png', dpi=150)
    print(f"Saved: {out_dir / 'reconstruction_error.png'}")
    
    # 3. Latent Space Plot (3D, colored by weekend)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Separate weekday and weekend points
    weekday_mask = df_val['is_weekend'] == 0
    weekend_mask = df_val['is_weekend'] == 1
    
    ax.scatter(latent_val[weekday_mask, 0], latent_val[weekday_mask, 1], latent_val[weekday_mask, 2], 
               c='blue', s=10, alpha=0.5, label='Weekday')
    ax.scatter(latent_val[weekend_mask, 0], latent_val[weekend_mask, 1], latent_val[weekend_mask, 2], 
               c='orange', s=10, alpha=0.5, label='Weekend')
    
    ax.set_title("Latent Space: Weekday vs Weekend Behavior")
    ax.set_xlabel("Latent Dim 1")
    ax.set_ylabel("Latent Dim 2")
    ax.set_zlabel("Latent Dim 3")
    ax.legend()
    plt.savefig(out_dir / 'latent_space.png', dpi=150)
    print(f"Saved: {out_dir / 'latent_space.png'}")
    
    # Save CSV
    res_dir = Path('reports/results')
    res_dir.mkdir(parents=True, exist_ok=True)
    
    # Save anomalies with day-of-week info
    anomalies_enhanced = anomalies[['participant_id', 'timestamp', 'day_of_week', 'is_weekend', 
                                     'reconstruction_error', 'threshold', 'activity_active_minutes']].copy()
    anomalies_enhanced['day_name'] = anomalies_enhanced['timestamp'].dt.day_name()
    anomalies_enhanced = anomalies_enhanced.sort_values('reconstruction_error', ascending=False)
    anomalies_enhanced.to_csv(res_dir / 'anomalies.csv', index=False)
    print(f"Saved: {res_dir / 'anomalies.csv'}")
    
    # Save summary statistics
    summary = {
        'total_samples': len(df_val),
        'weekday_samples': int(len(weekday_errors)),
        'weekend_samples': int(len(weekend_errors)),
        'total_anomalies': int(total_anomalies),
        'weekday_anomalies': int(weekday_anomalies),
        'weekend_anomalies': int(weekend_anomalies),
        'weekday_threshold': float(weekday_threshold),
        'weekend_threshold': float(weekend_threshold),
        'naive_threshold': float(naive_threshold),
        'naive_total_anomalies': int(naive_total),
        'naive_weekend_anomalies': int(naive_weekend),
        'false_positive_reduction': int(naive_weekend - weekend_anomalies)
    }
    
    with open(res_dir / 'anomaly_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {res_dir / 'anomaly_summary.json'}")
    
    print("\n✅ Analysis Complete. Weekend normalization implemented successfully!")
    print(f"   - Reduced weekend false positives by {naive_weekend - weekend_anomalies} cases")
    print(f"   - Visualizations saved to {out_dir}/")
    print(f"   - Results saved to {res_dir}/")

if __name__ == "__main__":
    try:
        train_autoencoder()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
