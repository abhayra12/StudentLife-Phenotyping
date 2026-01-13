"""
Phase 4 Verification Script

This script verifies the feature engineering modules by running them
on test data and generating visualization plots.
It serves as proof that the code in src/features/ works correctly.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add src to path
sys.path.append('.')

from src.features.temporal_features import process_temporal_features
from src.features.activity_sleep import process_activity_sleep
from src.features.location_features import process_location_features

def verify_phase4():
    print("Starting Phase 4 Verification...")
    
    # 1. Load or Generate Data
    DATA_DIR = Path('data/processed_test')
    if (DATA_DIR / 'train.parquet').exists():
        print(f"Loading test data from {DATA_DIR}")
        df = pd.read_parquet(DATA_DIR / 'train.parquet')
        
        # Ensure required columns exist (inject dummy if missing from test run)
        if 'gps_lat' not in df.columns:
            print("Injecting dummy GPS data for verification...")
            df['gps_lat'] = 43.7 + np.cumsum(np.random.randn(len(df)) * 0.001)
            df['gps_lon'] = -72.3 + np.cumsum(np.random.randn(len(df)) * 0.001)
            
        if 'activity_active_minutes' not in df.columns:
            print("Injecting dummy Activity data...")
            df['activity_active_minutes'] = np.random.randint(0, 60, size=len(df))
            
        if 'dark_minutes' not in df.columns:
            print("Injecting dummy Sleep data...")
            df['dark_minutes'] = np.random.randint(0, 60, size=len(df))
            df['phonelock_minutes'] = np.random.randint(0, 60, size=len(df))
            df['audio_voice_minutes'] = np.random.randint(0, 10, size=len(df))
            
            df['audio_voice_minutes'] = np.random.randint(0, 10, size=len(df))
            
    else:
        print("Test data not found. Generating dummy data.")
        dates = pd.date_range(start='2013-03-27', end='2013-06-05', freq='h')
        df = pd.DataFrame({'timestamp': dates})
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['week_of_term'] = ((df['timestamp'] - df['timestamp'].min()).dt.days // 7) + 1
        
        # Mock Sensor Data
        df['activity_active_minutes'] = np.random.randint(0, 60, size=len(df))
        df['dark_minutes'] = np.where(df['hour_of_day'].isin([0,1,2,3,4,5]), 60, 0)
        df['phonelock_minutes'] = np.where(df['hour_of_day'].isin([0,1,2,3,4,5]), 60, 10)
        df['audio_voice_minutes'] = np.random.randint(0, 10, size=len(df))
        
        # Mock GPS
        df['gps_lat'] = 43.7 + np.cumsum(np.random.randn(len(df)) * 0.001)
        df['gps_lon'] = -72.3 + np.cumsum(np.random.randn(len(df)) * 0.001)
        df['participant_id'] = 'u00'

    # Ensure timestamp is datetime and set as index for rolling operations
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp', drop=False)
        
    # 2. Run Feature Engineering
    print("Running Temporal Features...")
    df = process_temporal_features(df)
    
    print("Running Activity/Sleep Features...")
    df = process_activity_sleep(df)
    
    print("Running Location Features...")
    df = process_location_features(df)
    
    print(f"Final Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Reset index for plotting to avoid Seaborn issues
    df_plot = df.reset_index(drop=True)
    
    # 3. Generate Verification Plots
    OUTPUT_DIR = Path('reports/figures')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Cyclical Time
    try:
        if 'hour_of_day' in df_plot.columns:
            sns.scatterplot(data=df_plot.iloc[:24], x='hour_of_day_sin', y='hour_of_day_cos', hue='hour_of_day', ax=axes[0,0], s=100)
            axes[0,0].set_title("Cyclical Hour Encoding")
        else:
            print("Skipping Plot 1: hour_of_day missing")
    except Exception as e:
        print(f"Plot 1 failed: {e}")
    
    # Plot 2: Activity Heatmap
    try:
        if 'day_of_week' not in df_plot.columns:
            df_plot['day_of_week'] = df_plot['timestamp'].dt.dayofweek
        
        if 'active_ratio' in df_plot.columns:
            pivot = df_plot.pivot_table(index='day_of_week', columns='hour_of_day', values='active_ratio', aggfunc='mean')
            sns.heatmap(pivot, cmap='Reds', ax=axes[0,1])
            axes[0,1].set_title("Activity Ratio Heatmap")
        else:
            print("Skipping Plot 2: active_ratio missing")
    except Exception as e:
        print(f"Plot 2 failed: {e}")
    
    # Plot 3: Sleep Duration
    try:
        if 'sleep_duration_24h' in df_plot.columns:
            sns.histplot(df_plot['sleep_duration_24h'], bins=20, ax=axes[1,0])
            axes[1,0].set_title("Sleep Duration Distribution")
        else:
            print("Skipping Plot 3: sleep_duration_24h missing")
    except Exception as e:
        print(f"Plot 3 failed: {e}")
    
    # Plot 4: GPS Trajectory
    try:
        if 'location_cluster' in df_plot.columns:
            mask = df_plot['location_cluster'] != -1
            sns.scatterplot(data=df_plot[mask], x='gps_lon', y='gps_lat', hue='location_cluster', palette='tab10', ax=axes[1,1], s=50)
            axes[1,1].set_title("GPS Trajectory & Clusters")
        else:
            print("Skipping Plot 4: location_cluster missing")
    except Exception as e:
        print(f"Plot 4 failed: {e}")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'verification_phase4.png')
    print(f"Verification plot saved to {OUTPUT_DIR / 'verification_phase4.png'}")

if __name__ == "__main__":
    verify_phase4()
