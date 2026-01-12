"""
Analysis Script: Temporal & Circadian Features

Goal: Visualize and verify time-based features engineered in Phase 4.
This script replaces 'notebooks/04_features/01_temporal_features.ipynb'.

Methodology:
1. Load the processed training dataset.
2. Apply temporal feature engineering (Sin/Cos encoding, Day Parts).
3. Generate visualizations to confirm the logic:
   - Cyclical Encoding: Should form a circle (clock face).
   - Day Parts: Check distribution of Morning/Afternoon/Evening/Night.
   - Term Phase: Check distribution of Early/Midterm/Finals.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add project root to path to import src modules
sys.path.append('.')

from src.features.temporal_features import process_temporal_features

def analyze_temporal_features():
    print("--- Starting Temporal Feature Analysis ---")
    
    # 1. Load Data
    # We try to load the full dataset first, then fall back to test data
    DATA_PATH = Path('data/processed/train.parquet')
    TEST_DATA_PATH = Path('data/processed_test/train.parquet')
    
    if DATA_PATH.exists():
        print(f"Loading full dataset from {DATA_PATH}...")
        df = pd.read_parquet(DATA_PATH)
    elif TEST_DATA_PATH.exists():
        print(f"Loading test dataset from {TEST_DATA_PATH}...")
        df = pd.read_parquet(TEST_DATA_PATH)
    else:
        print("Error: No data found. Please run the data pipeline first.")
        return

    # 2. Apply Feature Engineering
    # This adds columns like 'hour_sin', 'hour_cos', 'day_part', 'term_phase'
    print("Applying temporal feature engineering...")
    df_features = process_temporal_features(df)
    
    # 3. Visualization
    OUTPUT_DIR = Path('reports/figures/analysis')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Cyclical Encoding Check
    # We plot Hour Sin vs Hour Cos. 
    # Since time is cyclical (23:00 is close to 00:00), this representation 
    # allows machine learning models to understand this proximity.
    # The plot should look like a perfect clock face.
    print("Generating Cyclical Encoding Plot...")
    plt.figure(figsize=(8, 8))
    # Take first 24 unique hours to verify the circle
    sample = df_features[['hour_of_day', 'hour_sin', 'hour_cos']].drop_duplicates().sort_values('hour_of_day')
    sns.scatterplot(data=sample, x='hour_sin', y='hour_cos', hue='hour_of_day', palette='viridis', s=200)
    
    # Add labels for clarity
    for i, row in sample.iterrows():
        plt.text(row['hour_sin']+0.05, row['hour_cos']+0.05, str(int(row['hour_of_day'])))
        
    plt.title("Cyclical Hour Encoding (24h Clock Representation)")
    plt.xlabel("Sin(Hour)")
    plt.ylabel("Cos(Hour)")
    plt.grid(True)
    plt.savefig(OUTPUT_DIR / '01_temporal_cyclical_encoding.png')
    plt.close()
    
    # Plot 2: Day Parts Distribution
    # Verify that we have a reasonable distribution of data across day parts.
    # If 'Night' is empty, we might be missing sleep data.
    print("Generating Day Parts Distribution Plot...")
    plt.figure(figsize=(10, 6))
    order = ['Morning', 'Afternoon', 'Evening', 'Night']
    sns.countplot(data=df_features, x='day_part', order=order, palette='Blues_d')
    plt.title("Distribution of Data Samples by Day Part")
    plt.xlabel("Day Part")
    plt.ylabel("Count of Hourly Samples")
    plt.savefig(OUTPUT_DIR / '01_temporal_day_parts.png')
    plt.close()
    
    print(f"Analysis complete. Plots saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    analyze_temporal_features()
