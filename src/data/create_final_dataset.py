"""
Create Final Dataset

This script combines aligned sensor data from all Tier 1 participants,
performs a time-based train/val/test split, and saves the final
datasets in Parquet format.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import shutil

# Constants
PROCESSED_PATH = Path('data/processed')
ALIGNED_PATH = PROCESSED_PATH / 'aligned'
CLEANED_PATH = PROCESSED_PATH / 'cleaned'
OUTPUT_PATH = PROCESSED_PATH

# Study Dates (Spring 2013)
STUDY_START = datetime(2013, 3, 27)
STUDY_END = datetime(2013, 6, 5)

def get_tier1_participants():
    """Load Tier 1 participant IDs."""
    tiers_file = PROCESSED_PATH / 'participant_tiers.csv'
    if not tiers_file.exists():
        raise FileNotFoundError(f"Tiers file not found: {tiers_file}")
        
    df = pd.read_csv(tiers_file)
    return df[df['quality_tier'] == 'Tier 1: Excellent']['participant'].tolist()

def build_full_dataset(participant_ids):
    """
    Stack all participant hourly data.
    
    Args:
        participant_ids: List of participant IDs
        
    Returns:
        DataFrame with all participants and time features
    """
    all_dfs = []
    
    print(f"Loading data for {len(participant_ids)} participants...")
    
    for pid in participant_ids:
        file_path = ALIGNED_PATH / f"aligned_{pid}.csv"
        if not file_path.exists():
            print(f"Warning: Aligned file not found for {pid}")
            continue
            
        df = pd.read_csv(file_path)
        
        # Ensure timestamp is datetime
        if 'Unnamed: 0' in df.columns:
            df = df.rename(columns={'Unnamed: 0': 'timestamp'})
            
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Add participant ID if missing (it should be there from alignment)
        if 'participant_id' not in df.columns:
            df['participant_id'] = pid
            
        all_dfs.append(df)
        
    if not all_dfs:
        raise ValueError("No data loaded!")
        
    full_df = pd.concat(all_dfs, ignore_index=True)
    
    # Sort by time
    full_df = full_df.sort_values('timestamp')
    
    # Add time features
    print("Adding time features...")
    full_df['hour_of_day'] = full_df['timestamp'].dt.hour
    full_df['day_of_week'] = full_df['timestamp'].dt.dayofweek
    
    # Calculate week of term (1-10)
    # Term starts March 27 (Week 1)
    days_since_start = (full_df['timestamp'] - STUDY_START).dt.days
    full_df['week_of_term'] = (days_since_start // 7) + 1
    
    # Clip weeks to 1-10 range (handle pre/post term data if any)
    full_df = full_df[full_df['week_of_term'].between(1, 10)]
    
    return full_df

def split_by_time(df):
    """
    Split chronologically:
    - Train: Week 1-6
    - Val: Week 7-8
    - Test: Week 9-10
    """
    print("Splitting dataset...")
    
    train_mask = df['week_of_term'].between(1, 6)
    val_mask = df['week_of_term'].between(7, 8)
    test_mask = df['week_of_term'].between(9, 10)
    
    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()
    
    print(f"Train: {len(train_df)} rows (Weeks 1-6)")
    print(f"Val:   {len(val_df)} rows (Weeks 7-8)")
    print(f"Test:  {len(test_df)} rows (Weeks 9-10)")
    
    return train_df, val_df, test_df

def save_datasets(train, val, test, output_dir):
    """Save datasets as Parquet and create metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving to {output_dir}...")
    
    # Save Parquet
    train.to_parquet(output_dir / 'train.parquet', index=False)
    val.to_parquet(output_dir / 'val.parquet', index=False)
    test.to_parquet(output_dir / 'test.parquet', index=False)
    
    # Create Metadata
    metadata = {
        "dataset_version": "1.0",
        "created_at": datetime.now().isoformat(),
        "participants": {
            "total_rows": len(train) + len(val) + len(test),
            "train_rows": len(train),
            "val_rows": len(val),
            "test_rows": len(test),
            "unique_participants": train['participant_id'].nunique()
        },
        "time_range": {
            "start": str(STUDY_START.date()),
            "end": str(STUDY_END.date())
        },
        "splits": {
            "train": {"weeks": [1, 2, 3, 4, 5, 6]},
            "val": {"weeks": [7, 8]},
            "test": {"weeks": [9, 10]}
        },
        "features": sorted(train.columns.tolist())
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
        
    print("Metadata saved.")

if __name__ == "__main__":
    try:
        # 1. Get Participants
        pids = get_tier1_participants()
        
        # 2. Build Dataset
        full_df = build_full_dataset(pids)
        
        # 3. Split
        train, val, test = split_by_time(full_df)
        
        # 4. Save
        save_datasets(train, val, test, OUTPUT_PATH)
        
        print("\nDataset creation complete!")
        
    except Exception as e:
        print(f"\nError: {e}")
        exit(1)
