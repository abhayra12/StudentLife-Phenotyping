"""
EDA Script: Sensor Deep Dive

Goal: Analyze all 10 StudentLife sensor types to verify dataset characteristics.
This script replaces 'notebooks/02_eda/01_sensor_deep_dive.ipynb'.

Key Questions:
- How many participants?
- How many weeks of data?
- What are the data formats and sampling rates?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import sys

# Add project root to path
sys.path.append('.')

def analyze_sensor_deep_dive():
    print("--- Starting Sensor Deep Dive Analysis ---")
    
    # Paths
    RAW_DATA_PATH = Path('data/raw/dataset/sensing')
    MANIFEST_PATH = Path('data/raw/sensing_manifest.json')
    OUTPUT_DIR = Path('reports/figures/eda')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Manifest
    if not MANIFEST_PATH.exists():
        print("Manifest not found. Please run data acquisition first.")
        return
        
    with open(MANIFEST_PATH, 'r') as f:
        manifest = json.load(f)
        
    print("Manifest loaded successfully.")
    print(f"Participants: {manifest['global_stats']['total_participants']}")
    print(f"Sensors: {manifest['global_stats']['total_sensor_types']}")
    
    # 2. Verify Participant Count
    participants = manifest['participants']
    print(f"Actual Participant Count: {len(participants)}")
    
    # 3. Analyze Study Duration (using Activity sensor as proxy)
    print("Analyzing Study Duration...")
    durations = []
    activity_path = RAW_DATA_PATH / 'activity'
    
    if activity_path.exists():
        for p in participants:
            files = list(activity_path.glob(f"*{p}*.csv"))
            if files:
                try:
                    df = pd.read_csv(files[0])
                    if 'timestamp' in df.columns:
                        start = pd.to_datetime(df['timestamp'].min(), unit='s')
                        end = pd.to_datetime(df['timestamp'].max(), unit='s')
                        days = (end - start).days
                        durations.append({'participant': p, 'weeks': days/7})
                except:
                    pass
    
    if durations:
        durations_df = pd.DataFrame(durations)
        avg_weeks = durations_df['weeks'].mean()
        print(f"Average Study Duration: {avg_weeks:.1f} weeks")
        
        # Plot Duration
        plt.figure(figsize=(10, 5))
        sns.histplot(durations_df['weeks'], bins=20)
        plt.axvline(10, color='r', linestyle='--', label='Target: 10 weeks')
        plt.title("Distribution of Study Duration")
        plt.xlabel("Weeks")
        plt.savefig(OUTPUT_DIR / '01_study_duration.png')
        plt.close()
    
    # 4. Sensor Size Summary
    print("Generating Sensor Size Summary...")
    summary_data = []
    for sensor, stats in manifest['per_sensor_stats'].items():
        summary_data.append({
            'Sensor': sensor,
            'Size (MB)': stats['total_size_mb']
        })
    
    summary_df = pd.DataFrame(summary_data).sort_values('Size (MB)', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=summary_df, x='Sensor', y='Size (MB)', palette='viridis')
    plt.xticks(rotation=45)
    plt.title("Data Size by Sensor Type")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_sensor_sizes.png')
    plt.close()
    
    print(f"Analysis complete. Plots saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    analyze_sensor_deep_dive()
