"""
EDA Script: Term Lifecycle Analysis

Goal: Analyze trends over the academic term.
This script replaces 'notebooks/02_eda/03_term_lifecycle.ipynb'.

Methodology:
1. Aggregate data by 'Week of Term'.
2. Visualize trends in activity, sleep, and conversation.
3. Identify mid-term and finals effects.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add project root to path
sys.path.append('.')

def analyze_term_lifecycle():
    print("--- Starting Term Lifecycle Analysis ---")
    
    # Load processed data if available (best for term analysis)
    DATA_PATH = Path('data/processed/train.parquet')
    OUTPUT_DIR = Path('reports/figures/eda')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not DATA_PATH.exists():
        print("Processed data not found. Skipping term analysis.")
        return
        
    df = pd.read_parquet(DATA_PATH)
    
    # Ensure week_of_term exists
    if 'timestamp' in df.columns:
        start_date = pd.to_datetime(df['timestamp']).min()
        df['week_of_term'] = ((pd.to_datetime(df['timestamp']) - start_date).dt.days // 7) + 1
        
    # Aggregate by week
    print("Aggregating by Week of Term...")
    weekly_stats = df.groupby('week_of_term').agg({
        'activity_active_minutes': 'mean',
        'conversation_minutes': 'mean',
        'dark_minutes': 'mean' # Proxy for sleep
    }).reset_index()
    
    # Plot Trends
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=weekly_stats, x='week_of_term', y='activity_active_minutes', marker='o', label='Activity')
    sns.lineplot(data=weekly_stats, x='week_of_term', y='conversation_minutes', marker='o', label='Conversation')
    plt.title("Behavioral Trends over Academic Term")
    plt.xlabel("Week of Term")
    plt.ylabel("Average Minutes/Hour")
    plt.grid(True)
    plt.legend()
    plt.savefig(OUTPUT_DIR / '03_term_trends.png')
    plt.close()
    
    print(f"Analysis complete. Plots saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    analyze_term_lifecycle()
