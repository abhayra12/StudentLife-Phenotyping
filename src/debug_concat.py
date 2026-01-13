
import pandas as pd
from pathlib import Path

def debug():
    p0 = Path('data/processed/aligned/aligned_u00.csv')
    p2 = Path('data/processed/aligned/aligned_u02.csv')
    
    print(f"Loading {p0}...")
    df0 = pd.read_csv(p0)
    if 'Unnamed: 0' in df0.columns: df0.rename(columns={'Unnamed: 0': 'timestamp'}, inplace=True)
    print(f"u00 columns: {list(df0.columns)}")
    
    print(f"Loading {p2}...")
    df2 = pd.read_csv(p2)
    if 'Unnamed: 0' in df2.columns: df2.rename(columns={'Unnamed: 0': 'timestamp'}, inplace=True)
    print(f"u02 columns: {list(df2.columns)}")
    
    print("Concatenating...")
    full = pd.concat([df0, df2], ignore_index=True)
    print(f"Full columns: {list(full.columns)}")
    
    if 'activity_active_minutes' in full.columns:
        print("SUCCESS: activity_active_minutes found.")
        print(f"Count non-null: {full['activity_active_minutes'].count()}")
    else:
        print("FAILURE: activity_active_minutes NOT found.")

if __name__ == '__main__':
    debug()
