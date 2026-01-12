"""
Run Alignment Pipeline

This script executes the alignment logic for all Tier 1 participants.
It serves as a programmatic alternative to running the notebook.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import sys

# Add src to path
sys.path.append('.')

from src.data.alignment import align_participant

def run_alignment():
    # Paths
    PROCESSED_PATH = Path('data/processed')
    CLEANED_PATH = PROCESSED_PATH / 'cleaned'
    ALIGNED_PATH = PROCESSED_PATH / 'aligned'
    ALIGNED_PATH.mkdir(parents=True, exist_ok=True)
    
    # Study Dates
    START_DATE = datetime(2013, 3, 27)
    END_DATE = datetime(2013, 6, 5)
    
    # Load Tier 1 participants
    tiers_file = PROCESSED_PATH / 'participant_tiers.csv'
    if not tiers_file.exists():
        print("Error: participant_tiers.csv not found")
        return
        
    tiers_df = pd.read_csv(tiers_file)
    tier1_participants = tiers_df[tiers_df['quality_tier'] == 'Tier 1: Excellent']['participant'].tolist()
    
    print(f"Aligning data for {len(tier1_participants)} Tier 1 participants...")
    
    sensors = ['activity', 'conversation', 'gps', 'bluetooth', 'wifi', 'dark', 'phonelock', 'phonecharge', 'audio', 'wifi_location']
    
    for pid in tier1_participants:
        print(f"Processing {pid}...")
        
        # Load sensors
        sensor_dfs = {}
        for sensor in sensors:
            file_path = CLEANED_PATH / sensor / f"{sensor}_{pid}.csv"
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    sensor_dfs[sensor] = df
                except Exception as e:
                    print(f"  Error loading {sensor}: {e}")
        
        if not sensor_dfs:
            print(f"  No data found for {pid}")
            continue
            
        # Align
        try:
            aligned_df = align_participant(pid, sensor_dfs, START_DATE, END_DATE)
            
            # Save
            out_file = ALIGNED_PATH / f"aligned_{pid}.csv"
            aligned_df.to_csv(out_file)
            print(f"  Saved {len(aligned_df)} rows")
            
        except Exception as e:
            print(f"  Error aligning {pid}: {e}")

if __name__ == "__main__":
    run_alignment()
