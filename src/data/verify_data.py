"""
Verify Dataset Integrity
-----------------------
Checks if the extracted dataset is readable and has the expected structure.
"""

import pandas as pd
from pathlib import Path
import os

DATA_DIR = Path("data/raw/dataset")

def check_file(path, name):
    print(f"Checking {name}...")
    if not path.exists():
        print(f"❌ MISSING: {path}")
        return False
    
    try:
        df = pd.read_csv(path)
        print(f"✅ LOADED: {name} | Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)[:5]}...")
        return True
    except Exception as e:
        print(f"❌ ERROR: Could not read {name}: {e}")
        return False

def main():
    print("🔍 Verifying StudentLife Dataset Integrity...\n")
    
    # 1. Check Sensing Data (Activity)
    # Note: Sensing data is often per-participant or in a specific folder structure
    # Let's check one participant's activity file if it exists, or the folder
    sensing_dir = DATA_DIR / "sensing"
    if sensing_dir.exists():
        # Activity is usually in 'activity' folder, with files like 'activity_u00.csv'
        activity_dir = sensing_dir / "activity"
        if activity_dir.exists():
            files = list(activity_dir.glob("*.csv"))
            if files:
                check_file(files[0], f"Activity Data ({files[0].name})")
            else:
                print("❌ No CSV files found in sensing/activity")
        else:
             print(f"❌ Missing directory: {activity_dir}")
    else:
        print(f"❌ Missing directory: {sensing_dir}")

    # 2. Check EMA Data
    ema_dir = DATA_DIR / "EMA/response"
    if ema_dir.exists():
        # Check Stress EMA if available
        stress_dir = ema_dir / "Stress" # Capitalization might vary
        if not stress_dir.exists():
             stress_dir = ema_dir / "stress"
        
        if stress_dir.exists():
             files = list(stress_dir.glob("*.csv"))
             if files:
                 check_file(files[0], f"Stress EMA ({files[0].name})")
        else:
            print(f"⚠️ Could not find Stress EMA folder in {ema_dir}")
    else:
        print(f"❌ Missing directory: {ema_dir}")

    # 3. Check Survey Data
    survey_dir = DATA_DIR / "survey"
    if survey_dir.exists():
        # PHQ-9.csv should be here
        phq9_path = survey_dir / "PHQ-9.csv"
        if not phq9_path.exists():
             phq9_path = survey_dir / "phq-9.csv"
        
        if phq9_path.exists():
            check_file(phq9_path, "PHQ-9 Survey")
        else:
             print(f"❌ Missing PHQ-9 file in {survey_dir}")
    else:
        print(f"❌ Missing directory: {survey_dir}")

if __name__ == "__main__":
    main()
