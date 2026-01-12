"""
Analysis Script: Activity & Sleep Features

Goal: Visualize and verify behavioral features related to physical activity and sleep.
This script replaces 'notebooks/04_features/02_activity_sleep.ipynb'.

Methodology:
1. Load the processed training dataset.
2. Apply activity and sleep feature engineering:
   - Calculate active/sedentary ratios per hour.
   - Infer sleep states using the heuristic (Dark + Lock + Quiet + Night).
3. Generate visualizations:
   - Activity Heatmap: Shows average activity levels by Day of Week and Hour of Day.
     Expected pattern: High activity during day, low at night.
   - Sleep Duration: Histogram of inferred sleep duration (rolling 24h sum).
     Expected pattern: Normal distribution centered around 6-8 hours.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add project root to path
sys.path.append('.')

from src.features.activity_sleep import process_activity_sleep

def analyze_activity_sleep():
    print("--- Starting Activity & Sleep Analysis ---")
    
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
    print("Applying activity/sleep feature engineering...")
    # Ensure timestamp is datetime for rolling operations
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        # Set index for rolling but keep column
        df = df.set_index('timestamp', drop=False)
        
    df_features = process_activity_sleep(df)
    
    # 3. Visualization
    OUTPUT_DIR = Path('reports/figures/analysis')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Activity Heatmap
    # We want to see WHEN students are active.
    # X-axis: Hour of Day (0-23)
    # Y-axis: Day of Week (Mon-Sun)
    # Color: Average Active Ratio (0.0 to 1.0)
    print("Generating Activity Heatmap...")
    try:
        # Ensure day_of_week exists
        if 'day_of_week' not in df_features.columns:
            df_features['day_of_week'] = df_features['timestamp'].dt.dayofweek
            
        if 'active_ratio' in df_features.columns:
            pivot = df_features.pivot_table(
                index='day_of_week', 
                columns='hour_of_day', 
                values='active_ratio', 
                aggfunc='mean'
            )
            plt.figure(figsize=(12, 6))
            sns.heatmap(pivot, cmap='Reds', annot=False) # annot=True can be messy with many cells
            plt.title("Average Activity Ratio (0=Sedentary, 1=Active)")
            plt.ylabel("Day of Week (0=Mon, 6=Sun)")
            plt.xlabel("Hour of Day")
            plt.savefig(OUTPUT_DIR / '02_activity_heatmap.png')
            plt.close()
        else:
            print("Skipping Heatmap: 'active_ratio' missing.")
    except Exception as e:
        print(f"Heatmap failed: {e}")

    # Plot 2: Sleep Duration Distribution
    # We want to verify if our sleep heuristic produces reasonable numbers.
    # If the peak is at 2 hours or 14 hours, our heuristic thresholds might be wrong.
    print("Generating Sleep Duration Histogram...")
    try:
        if 'sleep_duration_24h' in df_features.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(df_features['sleep_duration_24h'].dropna(), bins=24, kde=True)
            plt.title("Distribution of Inferred Sleep Duration (Rolling 24h)")
            plt.xlabel("Hours of Sleep")
            plt.ylabel("Frequency")
            plt.axvline(x=7, color='r', linestyle='--', label='7h Target')
            plt.legend()
            plt.savefig(OUTPUT_DIR / '02_sleep_distribution.png')
            plt.close()
        else:
            print("Skipping Sleep Hist: 'sleep_duration_24h' missing.")
    except Exception as e:
        print(f"Sleep Hist failed: {e}")
        
    print(f"Analysis complete. Plots saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    analyze_activity_sleep()
