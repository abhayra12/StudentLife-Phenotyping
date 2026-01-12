"""
Analysis Script: Location & Mobility Features

Goal: Visualize and verify spatial behavioral features.
This script replaces 'notebooks/04_features/03_location_features.ipynb'.

Methodology:
1. Load the processed training dataset.
2. Apply location feature engineering:
   - Calculate mobility metrics (Distance Traveled, Location Variance).
   - Identify Significant Places using DBSCAN clustering.
3. Generate visualizations:
   - GPS Trajectory: Scatter plot of Lat/Lon, colored by Cluster ID.
     Expected pattern: Dense clusters (Home, Work) and sparse points (Transit).
   - Distance Distribution: Histogram of hourly distance traveled.
     Expected pattern: Long tail (mostly 0 movement, occasional long trips).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add project root to path
sys.path.append('.')

from src.features.location_features import process_location_features

def analyze_location_features():
    print("--- Starting Location & Mobility Analysis ---")
    
    # 1. Load Data
    DATA_PATH = Path('data/processed/train.parquet')
    TEST_DATA_PATH = Path('data/processed_test/train.parquet')
    
    if DATA_PATH.exists():
        print(f"Loading full dataset from {DATA_PATH}...")
        df = pd.read_parquet(DATA_PATH)
    elif TEST_DATA_PATH.exists():
        print(f"Loading test dataset from {TEST_DATA_PATH}...")
        df = pd.read_parquet(TEST_DATA_PATH)
    else:
        print("Error: No data found.")
        return

    # 2. Feature Engineering
    print("Applying location feature engineering...")
    # Inject dummy GPS if missing (for verification robustness)
    if 'gps_lat' not in df.columns:
        print("Warning: GPS data missing. Injecting dummy random walk for demonstration.")
        df['gps_lat'] = 43.7 + np.cumsum(np.random.randn(len(df)) * 0.001)
        df['gps_lon'] = -72.3 + np.cumsum(np.random.randn(len(df)) * 0.001)
        
    df_features = process_location_features(df)
    
    # 3. Visualization
    OUTPUT_DIR = Path('reports/figures/analysis')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: GPS Trajectory & Clusters
    # Visualizes the spatial distribution of the participant.
    # Points with the same color belong to the same "Significant Place".
    # Grey points (-1) are noise/transit.
    print("Generating GPS Trajectory Plot...")
    try:
        if 'location_cluster' in df_features.columns:
            plt.figure(figsize=(10, 8))
            
            # Plot Noise first (grey, small, transparent)
            noise_mask = df_features['location_cluster'] == -1
            if noise_mask.any():
                sns.scatterplot(
                    data=df_features[noise_mask],
                    x='gps_lon', y='gps_lat',
                    color='grey', s=10, alpha=0.3, label='Noise/Transit'
                )
            
            # Plot Clusters (colored, larger)
            cluster_mask = ~noise_mask
            if cluster_mask.any():
                sns.scatterplot(
                    data=df_features[cluster_mask],
                    x='gps_lon', y='gps_lat',
                    hue='location_cluster', palette='tab10', s=50,
                    legend='full'
                )
                
            plt.title("GPS Trajectory & Significant Places (DBSCAN Clusters)")
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.savefig(OUTPUT_DIR / '03_location_trajectory.png')
            plt.close()
        else:
            print("Skipping Trajectory: 'location_cluster' missing.")
    except Exception as e:
        print(f"Trajectory plot failed: {e}")
        
    # Plot 2: Distance Traveled Distribution
    # Verifies the mobility metric.
    print("Generating Distance Distribution Plot...")
    try:
        if 'distance_traveled' in df_features.columns:
            plt.figure(figsize=(10, 6))
            # Use log scale because most values are small (0-1km) but some are large
            sns.histplot(df_features['distance_traveled'].dropna(), bins=50, log_scale=(False, True))
            plt.title("Hourly Distance Traveled (km) - Log Frequency")
            plt.xlabel("Distance (km)")
            plt.ylabel("Log Count")
            plt.savefig(OUTPUT_DIR / '03_mobility_distance.png')
            plt.close()
        else:
            print("Skipping Distance Plot: 'distance_traveled' missing.")
    except Exception as e:
        print(f"Distance plot failed: {e}")
        
    print(f"Analysis complete. Plots saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    analyze_location_features()
