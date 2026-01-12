"""
Run Cleaning Pipeline

This script executes the data cleaning pipeline for all Tier 1 participants.
It serves as a programmatic alternative to running the notebook.
"""

import pandas as pd
from pathlib import Path
import sys
import shutil

# Add src to path
sys.path.append('.')

from src.data.cleaning import (
    validate_timestamps,
    validate_values,
    detect_outliers,
    handle_missing_data
)

def run_cleaning():
    # Paths
    RAW_PATH = Path('data/raw/dataset/sensing')
    PROCESSED_PATH = Path('data/processed')
    CLEANED_PATH = PROCESSED_PATH / 'cleaned'
    CLEANED_PATH.mkdir(parents=True, exist_ok=True)
    
    # Load Tier 1 participants
    tiers_file = PROCESSED_PATH / 'participant_tiers.csv'
    if not tiers_file.exists():
        print("Error: participant_tiers.csv not found")
        return
        
    tiers_df = pd.read_csv(tiers_file)
    tier1_participants = tiers_df[tiers_df['quality_tier'] == 'Tier 1: Excellent']['participant'].tolist()
    
    print(f"Cleaning data for {len(tier1_participants)} Tier 1 participants...")
    
    sensors = ['activity', 'audio', 'bluetooth', 'conversation', 'dark', 'gps', 'phonecharge', 'phonelock', 'wifi', 'wifi_location']
    
    for sensor in sensors:
        print(f"Processing {sensor}...")
        sensor_out_dir = CLEANED_PATH / sensor
        sensor_out_dir.mkdir(exist_ok=True)
        
        for pid in tier1_participants:
            # Try both naming conventions
            f1 = RAW_PATH / sensor / f"{sensor}_{pid}.csv"
            f2 = RAW_PATH / sensor / f"{pid}.{sensor}.csv"
            
            input_file = f1 if f1.exists() else (f2 if f2.exists() else None)
            
            if not input_file:
                continue
                
            try:
                df = pd.read_csv(input_file)
                
                # 1. Validate Timestamps
                time_col = 'timestamp' if 'timestamp' in df.columns else ('time' if 'time' in df.columns else None)
                if not time_col:
                    # Some sensors like conversation use start_timestamp
                    if 'start_timestamp' in df.columns:
                        time_col = 'start_timestamp'
                    elif 'start' in df.columns:
                        time_col = 'start'
                        
                if time_col:
                    df, _ = validate_timestamps(df, time_col)
                    
                # 2. Validate Values
                df, _ = validate_values(df, sensor)
                
                # 3. Handle Missing (simple drop for now to be safe)
                # df = handle_missing_data(df, strategy='drop')
                
                # Save
                out_file = sensor_out_dir / f"{sensor}_{pid}.csv"
                df.to_csv(out_file, index=False)
                
            except Exception as e:
                print(f"  Error cleaning {sensor} for {pid}: {e}")

if __name__ == "__main__":
    run_cleaning()
